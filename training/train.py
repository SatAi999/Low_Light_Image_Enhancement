"""
Training Script for Low-Light Image Enhancement

Supports multiple training modes:
1. Supervised training (with paired data)
2. Self-supervised training (Zero-DCE style)
3. Hybrid Retinex + CNN training
4. Ablation studies

Features:
- Multi-GPU training
- Automatic checkpointing
- TensorBoard logging
- Configurable loss weights
- Learning rate scheduling
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloader
from models import LightweightUNet, DCENet, EnhancementCNN, HybridRetinexNet, count_parameters
from training.losses import EnhancementLoss
from evaluation.metrics import evaluate_enhancement, MetricTracker
from retinex import apply_retinex_to_tensor
from utils.visualization import save_comparison_grid, plot_training_curves
from utils.common import set_seed, save_checkpoint, load_checkpoint, get_device


class Trainer:
    """
    Main trainer class for low-light image enhancement
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Create output directories
        self.exp_dir = Path(config['experiment_dir'])
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.results_dir = self.exp_dir / 'results'
        self.logs_dir = self.exp_dir / 'logs'
        
        for dir_path in [self.checkpoint_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Setup model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        print(f"\nModel: {config['model']['name']}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}\n")
        
        # Setup loss
        self.criterion = self._build_criterion()
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup scheduler
        self.scheduler = self._build_scheduler()
        
        # Setup data loaders
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val')
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.metric_tracker = MetricTracker()
    
    def _build_model(self) -> nn.Module:
        """Build model based on config"""
        model_config = self.config['model']
        model_name = model_config['name']
        
        if model_name == 'LightweightUNet':
            model = LightweightUNet(
                in_channels=3,
                out_channels=3,
                base_features=model_config.get('base_features', 32)
            )
        elif model_name == 'DCENet':
            model = DCENet(
                in_channels=3,
                num_iterations=model_config.get('num_iterations', 8),
                base_channels=model_config.get('base_channels', 32)
            )
        elif model_name == 'EnhancementCNN':
            model = EnhancementCNN(
                in_channels=3,
                out_channels=3,
                base_channels=model_config.get('base_channels', 64),
                num_residual_blocks=model_config.get('num_residual_blocks', 6)
            )
        elif model_name == 'HybridRetinexNet':
            model = HybridRetinexNet(
                in_channels=3,
                out_channels=3,
                use_retinex_input=model_config.get('use_retinex_input', True)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def _build_criterion(self) -> nn.Module:
        """Build loss function based on config"""
        loss_config = self.config['loss']
        
        criterion = EnhancementLoss(
            use_perceptual=loss_config.get('use_perceptual', False),
            use_reconstruction=loss_config.get('use_reconstruction', True),
            lambda_recon=loss_config.get('lambda_recon', 1.0),
            lambda_perceptual=loss_config.get('lambda_perceptual', 0.1),
            lambda_exposure=loss_config.get('lambda_exposure', 1.0),
            lambda_color=loss_config.get('lambda_color', 5.0),
            lambda_spatial=loss_config.get('lambda_spatial', 10.0),
            lambda_illum=loss_config.get('lambda_illum', 1.0),
            target_exposure=loss_config.get('target_exposure', 0.6)
        )
        
        return criterion.to(self.device)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['name'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        
        if not sched_config or sched_config.get('name') == 'None':
            return None
        
        if sched_config['name'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_config['name'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        else:
            return None
        
        return scheduler
    
    def _build_dataloader(self, split: str) -> DataLoader:
        """Build data loader"""
        data_config = self.config['data']
        
        loader = get_dataloader(
            root_dir=data_config['root_dir'],
            split=split,
            batch_size=data_config['batch_size'],
            image_size=data_config['image_size'],
            augment=(split == 'train'),
            num_workers=data_config.get('num_workers', 4),
            paired=data_config.get('paired', True)
        )
        
        return loader
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        self.metric_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            low_light = batch['low'].to(self.device)
            
            # Get ground truth if available
            if 'high' in batch:
                target = batch['high'].to(self.device)
            else:
                target = None
            
            # Forward pass
            if self.config['model']['name'] == 'HybridRetinexNet' and \
               self.config['model'].get('use_retinex_input', False):
                # Apply Retinex preprocessing
                with torch.no_grad():
                    _, reflectance = apply_retinex_to_tensor(
                        low_light,
                        method='MSR',
                        scales=[15, 80, 250],
                        gamma=1.2
                    )
                enhanced = self.model(low_light, reflectance)
            elif self.config['model']['name'] == 'DCENet':
                enhanced, curves = self.model(low_light)
            else:
                enhanced = self.model(low_light)
            
            # Compute loss
            total_loss, losses = self.criterion(
                enhanced=enhanced,
                input_low=low_light,
                target=target
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.metric_tracker.update(losses)
            
            # Update progress bar
            pbar.set_postfix({'loss': losses['total']})
        
        return self.metric_tracker.get_averages()
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set"""
        self.model.eval()
        self.metric_tracker.reset()
        
        all_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            low_light = batch['low'].to(self.device)
            
            if 'high' in batch:
                target = batch['high'].to(self.device)
            else:
                target = None
            
            # Forward pass
            if self.config['model']['name'] == 'HybridRetinexNet' and \
               self.config['model'].get('use_retinex_input', False):
                _, reflectance = apply_retinex_to_tensor(
                    low_light,
                    method='MSR',
                    scales=[15, 80, 250],
                    gamma=1.2
                )
                enhanced = self.model(low_light, reflectance)
            elif self.config['model']['name'] == 'DCENet':
                enhanced, _ = self.model(low_light)
            else:
                enhanced = self.model(low_light)
            
            # Compute metrics
            if target is not None:
                metrics = evaluate_enhancement(enhanced, target, compute_niqe=False)
                all_metrics.append(metrics)
            
            # Save sample visualizations
            if batch_idx == 0:
                save_comparison_grid(
                    low_light=low_light[:4],
                    enhanced=enhanced[:4],
                    target=target[:4] if target is not None else None,
                    save_path=self.results_dir / f'epoch_{self.current_epoch}.png'
                )
        
        # Average metrics
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            return avg_metrics
        else:
            return {}
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['epochs']} epochs...\n")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log to TensorBoard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics.get('total', 0):.4f}")
            if val_metrics:
                print(f"  Val PSNR: {val_metrics.get('psnr', 0):.2f} dB")
                print(f"  Val SSIM: {val_metrics.get('ssim', 0):.4f}")
            
            # Save checkpoint
            is_best = val_metrics.get('psnr', 0) > self.best_metric
            if is_best:
                self.best_metric = val_metrics.get('psnr', 0)
            
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_metrics,
                save_dir=self.checkpoint_dir,
                is_best=is_best
            )
        
        print(f"\nTraining completed!")
        print(f"Best PSNR: {self.best_metric:.2f} dB")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Low-Light Enhancement Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        load_checkpoint(args.resume, trainer.model, trainer.optimizer, trainer.scheduler)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
