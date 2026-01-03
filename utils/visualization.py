"""
Utility functions for visualization, saving results, and plotting
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Optional, List
import cv2


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array for visualization
    
    Args:
        tensor: [C, H, W] or [B, C, H, W] in [0, 1]
    
    Returns:
        numpy array [H, W, C] or [B, H, W, C] in [0, 255] uint8
    """
    if tensor.dim() == 4:
        # Batch: [B, C, H, W] -> [B, H, W, C]
        arr = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    else:
        # Single: [C, H, W] -> [H, W, C]
        arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def save_image(image: torch.Tensor, save_path: str):
    """Save a single image tensor to disk"""
    img_np = tensor_to_numpy(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, img_np)


def save_comparison_grid(
    low_light: torch.Tensor,
    enhanced: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    save_path: str = 'comparison.png',
    max_images: int = 4
):
    """
    Save a grid comparing low-light, enhanced, and target images
    
    Args:
        low_light: Low-light images [B, C, H, W]
        enhanced: Enhanced images [B, C, H, W]
        target: Target images (optional) [B, C, H, W]
        save_path: Where to save the comparison
        max_images: Maximum number of images to show
    """
    batch_size = min(low_light.shape[0], max_images)
    
    # Determine number of columns
    n_cols = 3 if target is not None else 2
    
    # Create figure
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(n_cols * 4, batch_size * 4))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Low-light input
        axes[i, 0].imshow(tensor_to_numpy(low_light[i]))
        axes[i, 0].set_title('Low-Light Input')
        axes[i, 0].axis('off')
        
        # Enhanced output
        axes[i, 1].imshow(tensor_to_numpy(enhanced[i]))
        axes[i, 1].set_title('Enhanced')
        axes[i, 1].axis('off')
        
        # Target (if available)
        if target is not None:
            axes[i, 2].imshow(tensor_to_numpy(target[i]))
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_retinex_decomposition(
    input_image: torch.Tensor,
    reflectance: torch.Tensor,
    illumination: torch.Tensor,
    enhanced: torch.Tensor,
    save_path: str = 'retinex_decomposition.png'
):
    """
    Visualize Retinex decomposition components
    
    Shows: Input, Reflectance, Illumination, Enhanced
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(tensor_to_numpy(input_image))
    axes[0].set_title('Input (Low-Light)')
    axes[0].axis('off')
    
    axes[1].imshow(tensor_to_numpy(reflectance))
    axes[1].set_title('Reflectance (R)')
    axes[1].axis('off')
    
    axes[2].imshow(tensor_to_numpy(illumination))
    axes[2].set_title('Illumination (L)')
    axes[2].axis('off')
    
    axes[3].imshow(tensor_to_numpy(enhanced))
    axes[3].set_title('Enhanced Output')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_histogram_comparison(
    images: List[torch.Tensor],
    labels: List[str],
    save_path: str = 'histogram.png'
):
    """
    Plot RGB histograms for multiple images
    
    Args:
        images: List of image tensors [C, H, W]
        labels: List of labels for each image
        save_path: Where to save the plot
    """
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 1, figsize=(10, 3 * n_images))
    
    if n_images == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue']
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        img_np = img.detach().cpu().numpy()  # [C, H, W]
        
        for c, color in enumerate(colors):
            channel_data = img_np[c].flatten()
            axes[idx].hist(channel_data, bins=256, range=(0, 1), 
                          alpha=0.5, color=color, label=color.upper())
        
        axes[idx].set_title(f'{label} - RGB Histogram')
        axes[idx].set_xlabel('Pixel Intensity')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(
    metrics_log: dict,
    save_path: str = 'training_curves.png'
):
    """
    Plot training curves (loss, PSNR, SSIM over epochs)
    
    Args:
        metrics_log: Dictionary with lists of metrics per epoch
            Example: {'train_loss': [...], 'val_psnr': [...]}
        save_path: Where to save the plot
    """
    n_metrics = len(metrics_log)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_name, values) in enumerate(metrics_log.items()):
        axes[idx].plot(values, linewidth=2)
        axes[idx].set_title(metric_name.replace('_', ' ').title())
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(metrics_log), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_ablation_table(
    results: dict,
    save_path: str = 'ablation_results.txt'
):
    """
    Create a formatted table for ablation study results
    
    Args:
        results: Dictionary of experiment_name -> metrics
            Example: {'Baseline': {'psnr': 25.0, 'ssim': 0.85}, ...}
        save_path: Where to save the table
    """
    # Determine column widths
    exp_names = list(results.keys())
    metric_names = list(results[exp_names[0]].keys())
    
    # Create header
    header = f"{'Experiment':<30}"
    for metric in metric_names:
        header += f"{metric.upper():>12}"
    
    # Create rows
    rows = []
    for exp_name, metrics in results.items():
        row = f"{exp_name:<30}"
        for metric_name in metric_names:
            value = metrics.get(metric_name, 0)
            row += f"{value:>12.4f}"
        rows.append(row)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for row in rows:
            f.write(row + '\n')
    
    print(f"Ablation table saved to {save_path}")


if __name__ == '__main__':
    """Test visualization functions"""
    print("Testing visualization utilities...")
    
    # Create dummy data
    low_light = torch.rand(4, 3, 256, 256) * 0.3
    enhanced = torch.rand(4, 3, 256, 256)
    target = torch.rand(4, 3, 256, 256)
    
    # Test comparison grid
    save_comparison_grid(low_light, enhanced, target, 'test_comparison.png')
    print("✓ Comparison grid saved")
    
    # Test histogram
    plot_histogram_comparison(
        [low_light[0], enhanced[0], target[0]],
        ['Low-Light', 'Enhanced', 'Target'],
        'test_histogram.png'
    )
    print("✓ Histogram saved")
    
    # Test training curves
    metrics_log = {
        'train_loss': np.linspace(1.0, 0.1, 50),
        'val_psnr': np.linspace(20, 30, 50),
        'val_ssim': np.linspace(0.7, 0.9, 50)
    }
    plot_training_curves(metrics_log, 'test_curves.png')
    print("✓ Training curves saved")
    
    # Clean up test files
    for f in ['test_comparison.png', 'test_histogram.png', 'test_curves.png']:
        if os.path.exists(f):
            os.remove(f)
    
    print("✓ All visualization utilities working!")
