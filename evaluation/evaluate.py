"""
Evaluation Script for Comprehensive Model Assessment

Features:
- Quantitative metrics (PSNR, SSIM, NIQE)
- Visual comparisons
- Histogram analysis
- Ablation studies
- Failure case analysis
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloader
from models import LightweightUNet, DCENet, EnhancementCNN, HybridRetinexNet
from evaluation.metrics import evaluate_enhancement, MetricTracker
from retinex import RetinexEnhancer, apply_retinex_to_tensor
from utils import (
    get_device, load_checkpoint, save_comparison_grid,
    save_retinex_decomposition, plot_histogram_comparison,
    create_ablation_table
)


class Evaluator:
    """
    Comprehensive evaluation framework
    """
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = get_device()
        
        # Create results directory
        self.results_dir = Path(config.get('results_dir', 'evaluation_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        if checkpoint_path:
            load_checkpoint(checkpoint_path, self.model)
        
        # Load test data
        self.test_loader = self._build_dataloader()
        
        print(f"Evaluator initialized")
        print(f"Results will be saved to: {self.results_dir}")
    
    def _build_model(self) -> nn.Module:
        """Build model from config"""
        model_config = self.config['model']
        model_name = model_config['name']
        
        if model_name == 'LightweightUNet':
            model = LightweightUNet(base_features=model_config.get('base_features', 32))
        elif model_name == 'DCENet':
            model = DCENet(
                num_iterations=model_config.get('num_iterations', 8),
                base_channels=model_config.get('base_channels', 32)
            )
        elif model_name == 'EnhancementCNN':
            model = EnhancementCNN(
                base_channels=model_config.get('base_channels', 64),
                num_residual_blocks=model_config.get('num_residual_blocks', 6)
            )
        elif model_name == 'HybridRetinexNet':
            model = HybridRetinexNet(
                use_retinex_input=model_config.get('use_retinex_input', True)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def _build_dataloader(self):
        """Build test dataloader"""
        data_config = self.config['data']
        
        return get_dataloader(
            root_dir=data_config['root_dir'],
            split='test',
            batch_size=1,
            image_size=data_config['image_size'],
            augment=False,
            num_workers=0,
            paired=data_config.get('paired', True)
        )
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run comprehensive evaluation
        
        Returns:
            Dictionary of aggregate metrics
        """
        print("\n" + "="*50)
        print("Starting Comprehensive Evaluation")
        print("="*50 + "\n")
        
        metric_tracker = MetricTracker()
        all_metrics = []
        
        # Retinex-only baseline
        retinex_enhancer = RetinexEnhancer(method='MSR', scales=[15, 80, 250])
        retinex_metrics = []
        
        for idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
            low_light = batch['low'].to(self.device)
            filename = batch['filename'][0]
            
            if 'high' in batch:
                target = batch['high'].to(self.device)
            else:
                target = None
            
            # 1. Raw low-light input (baseline)
            raw_input = low_light
            
            # 2. Classical Retinex only
            retinex_enhanced, retinex_reflectance = apply_retinex_to_tensor(
                low_light, method='MSR', scales=[15, 80, 250], gamma=1.2
            )
            
            # 3. Deep learning enhancement
            if self.config['model']['name'] == 'HybridRetinexNet' and \
               self.config['model'].get('use_retinex_input', False):
                dl_enhanced = self.model(low_light, retinex_reflectance)
            elif self.config['model']['name'] == 'DCENet':
                dl_enhanced, _ = self.model(low_light)
            else:
                dl_enhanced = self.model(low_light)
            
            # Compute metrics
            if target is not None:
                # Metrics for DL model
                metrics = evaluate_enhancement(dl_enhanced, target, compute_niqe=True)
                all_metrics.append(metrics)
                metric_tracker.update(metrics)
                
                # Metrics for Retinex baseline
                retinex_met = evaluate_enhancement(retinex_enhanced, target, compute_niqe=False)
                retinex_metrics.append(retinex_met)
            
            # Save visual comparisons for first 10 images
            if idx < 10:
                save_path = self.results_dir / f'comparison_{idx:03d}_{filename}'
                save_comparison_grid(
                    low_light=low_light,
                    enhanced=dl_enhanced,
                    target=target,
                    save_path=str(save_path)
                )
                
                # Save Retinex decomposition
                if self.config['model'].get('use_retinex_input', False):
                    # Get illumination (simplified: input / reflectance)
                    illum = low_light / (retinex_reflectance + 1e-6)
                    illum = torch.clamp(illum, 0, 1)
                    
                    decomp_path = self.results_dir / f'retinex_decomp_{idx:03d}.png'
                    save_retinex_decomposition(
                        input_image=low_light[0],
                        reflectance=retinex_reflectance[0],
                        illumination=illum[0],
                        enhanced=dl_enhanced[0],
                        save_path=str(decomp_path)
                    )
                
                # Save histograms
                hist_path = self.results_dir / f'histogram_{idx:03d}.png'
                plot_histogram_comparison(
                    images=[low_light[0], dl_enhanced[0]] + ([target[0]] if target is not None else []),
                    labels=['Low-Light', 'Enhanced'] + (['Target'] if target is not None else []),
                    save_path=str(hist_path)
                )
        
        # Aggregate results
        avg_metrics = metric_tracker.get_averages()
        
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"\nDeep Learning Model ({self.config['model']['name']}):")
        print(f"  PSNR: {avg_metrics.get('psnr', 0):.2f} dB")
        print(f"  SSIM: {avg_metrics.get('ssim', 0):.4f}")
        print(f"  NIQE: {avg_metrics.get('niqe', 0):.4f}")
        
        if retinex_metrics:
            retinex_avg = {
                'psnr': np.mean([m['psnr'] for m in retinex_metrics]),
                'ssim': np.mean([m['ssim'] for m in retinex_metrics])
            }
            print(f"\nRetinex Baseline (MSR):")
            print(f"  PSNR: {retinex_avg['psnr']:.2f} dB")
            print(f"  SSIM: {retinex_avg['ssim']:.4f}")
        
        # Save results to JSON
        results = {
            'model': self.config['model']['name'],
            'average_metrics': avg_metrics,
            'retinex_baseline': retinex_avg if retinex_metrics else None,
            'per_image_metrics': all_metrics
        }
        
        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {self.results_dir}")
        
        return avg_metrics
    
    def ablation_study(self):
        """
        Perform ablation study comparing different components
        """
        print("\n" + "="*50)
        print("Ablation Study")
        print("="*50 + "\n")
        
        ablation_results = {}
        
        # Test configurations
        configs = {
            'Raw Input': {'method': 'none'},
            'SSR (σ=80)': {'method': 'ssr', 'scales': [80]},
            'MSR': {'method': 'msr', 'scales': [15, 80, 250]},
            'DL Only': {'method': 'dl_only'},
            'Hybrid (SSR + DL)': {'method': 'hybrid_ssr'},
            'Hybrid (MSR + DL)': {'method': 'hybrid_msr'}
        }
        
        for config_name, config in configs.items():
            print(f"Testing: {config_name}...")
            metrics_list = []
            
            for batch in tqdm(self.test_loader, desc=config_name, leave=False):
                low_light = batch['low'].to(self.device)
                target = batch.get('high')
                
                if target is None:
                    continue
                
                target = target.to(self.device)
                
                # Apply enhancement based on config
                if config['method'] == 'none':
                    enhanced = low_light
                elif config['method'] == 'ssr':
                    enhanced, _ = apply_retinex_to_tensor(
                        low_light, method='SSR', scales=config['scales'], gamma=1.2
                    )
                elif config['method'] == 'msr':
                    enhanced, _ = apply_retinex_to_tensor(
                        low_light, method='MSR', scales=config['scales'], gamma=1.2
                    )
                elif config['method'] == 'dl_only':
                    enhanced = self.model(low_light)
                elif config['method'] == 'hybrid_ssr':
                    _, reflectance = apply_retinex_to_tensor(
                        low_light, method='SSR', scales=[80], gamma=1.2
                    )
                    enhanced = self.model(low_light, reflectance)
                elif config['method'] == 'hybrid_msr':
                    _, reflectance = apply_retinex_to_tensor(
                        low_light, method='MSR', scales=[15, 80, 250], gamma=1.2
                    )
                    enhanced = self.model(low_light, reflectance)
                
                # Compute metrics
                metrics = evaluate_enhancement(enhanced, target, compute_niqe=False)
                metrics_list.append(metrics)
            
            # Average metrics
            avg_metrics = {
                'psnr': np.mean([m['psnr'] for m in metrics_list]),
                'ssim': np.mean([m['ssim'] for m in metrics_list])
            }
            
            ablation_results[config_name] = avg_metrics
        
        # Create table
        create_ablation_table(
            ablation_results,
            save_path=str(self.results_dir / 'ablation_table.txt')
        )
        
        print("\n✓ Ablation study complete")
        return ablation_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Low-Light Enhancement Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['results_dir'] = args.results_dir
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Run evaluation
    evaluator.evaluate()
    
    # Run ablation study if requested
    if args.ablation:
        evaluator.ablation_study()


if __name__ == '__main__':
    main()
