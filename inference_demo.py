"""
Inference Demo Script

Run low-light enhancement on custom images using trained models.
Supports:
- Single image inference
- Batch processing
- Different model types
- Side-by-side comparisons
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LightweightUNet, DCENet, EnhancementCNN, HybridRetinexNet
from retinex import RetinexEnhancer, apply_retinex_to_tensor
from utils import get_device, load_checkpoint, save_comparison_grid


class ImageEnhancer:
    """
    Inference wrapper for low-light image enhancement
    """
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Args:
            config_path: Path to model config YAML
            checkpoint_path: Path to trained model checkpoint
        """
        self.device = get_device()
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        load_checkpoint(checkpoint_path, self.model)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.model_name = self.config['model']['name']
        print(f"✓ Model loaded: {self.model_name}")
    
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
    
    @torch.no_grad()
    def enhance_image(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        Enhance a single low-light image
        
        Args:
            image_path: Path to input image
            save_path: Path to save enhanced image (optional)
        
        Returns:
            Enhanced image as numpy array [H, W, C] in [0, 255]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform to tensor
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Apply enhancement
        if self.model_name == 'HybridRetinexNet' and \
           self.config['model'].get('use_retinex_input', False):
            # Apply Retinex preprocessing
            _, reflectance = apply_retinex_to_tensor(
                img_tensor,
                method='MSR',
                scales=[15, 80, 250],
                gamma=1.2
            )
            enhanced_tensor = self.model(img_tensor, reflectance)
        elif self.model_name == 'DCENet':
            enhanced_tensor, _ = self.model(img_tensor)
        else:
            enhanced_tensor = self.model(img_tensor)
        
        # Convert back to numpy
        enhanced_np = enhanced_tensor[0].cpu().permute(1, 2, 0).numpy()
        enhanced_np = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
        
        # Resize back to original size if needed
        if enhanced_np.shape[:2] != (original_size[1], original_size[0]):
            enhanced_np = cv2.resize(enhanced_np, original_size, interpolation=cv2.INTER_CUBIC)
        
        # Save if path provided
        if save_path:
            enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, enhanced_bgr)
            print(f"✓ Saved enhanced image to: {save_path}")
        
        return enhanced_np
    
    def enhance_batch(self, input_dir: str, output_dir: str, compare: bool = True):
        """
        Enhance all images in a directory
        
        Args:
            input_dir: Directory with input images
            output_dir: Directory to save enhanced images
            compare: Create side-by-side comparison
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        for img_file in image_files:
            print(f"Processing: {img_file.name}...")
            
            # Enhance image
            enhanced = self.enhance_image(
                str(img_file),
                save_path=str(output_path / f'enhanced_{img_file.name}')
            )
            
            # Create comparison if requested
            if compare:
                # Load original
                original = cv2.imread(str(img_file))
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                
                # Resize to match if needed
                if original.shape[:2] != enhanced.shape[:2]:
                    original = cv2.resize(original, (enhanced.shape[1], enhanced.shape[0]))
                
                # Create side-by-side comparison
                comparison = np.hstack([original, enhanced])
                comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(output_path / f'comparison_{img_file.name}'),
                    comparison_bgr
                )
        
        print(f"\n✓ Processed {len(image_files)} images")
        print(f"✓ Results saved to: {output_dir}")


def demo_retinex_only(image_path: str, output_dir: str = 'retinex_demo'):
    """
    Demo classical Retinex enhancement without deep learning
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Test different Retinex configurations
    configs = [
        ('SSR_sigma15', 'SSR', [15]),
        ('SSR_sigma80', 'SSR', [80]),
        ('SSR_sigma250', 'SSR', [250]),
        ('MSR', 'MSR', [15, 80, 250])
    ]
    
    results = {'Input': image_rgb}
    
    for name, method, scales in configs:
        print(f"Applying {name}...")
        enhancer = RetinexEnhancer(method=method, scales=scales)
        enhanced, components = enhancer.enhance(
            image_rgb,
            gamma=1.2,
            return_components=True
        )
        results[name] = enhanced
        
        # Save individual result
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path / f'{name}.png'), enhanced_bgr)
    
    # Create comparison grid
    n_results = len(results)
    n_cols = 3
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()
    
    for idx, (name, img) in enumerate(results.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=14)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'retinex_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Retinex demo results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Low-Light Image Enhancement Inference')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single image enhancement
    single_parser = subparsers.add_parser('single', help='Enhance a single image')
    single_parser.add_argument('--config', type=str, required=True, help='Model config file')
    single_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    single_parser.add_argument('--input', type=str, required=True, help='Input image path')
    single_parser.add_argument('--output', type=str, required=True, help='Output image path')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Enhance multiple images')
    batch_parser.add_argument('--config', type=str, required=True, help='Model config file')
    batch_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    batch_parser.add_argument('--no-compare', action='store_true', help='Skip comparison images')
    
    # Retinex-only demo
    retinex_parser = subparsers.add_parser('retinex', help='Demo classical Retinex')
    retinex_parser.add_argument('--input', type=str, required=True, help='Input image path')
    retinex_parser.add_argument('--output_dir', type=str, default='retinex_demo', 
                               help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        enhancer = ImageEnhancer(args.config, args.checkpoint)
        enhancer.enhance_image(args.input, args.output)
    
    elif args.command == 'batch':
        enhancer = ImageEnhancer(args.config, args.checkpoint)
        enhancer.enhance_batch(args.input_dir, args.output_dir, compare=not args.no_compare)
    
    elif args.command == 'retinex':
        demo_retinex_only(args.input, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    main()
