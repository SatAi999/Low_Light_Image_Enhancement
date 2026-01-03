"""
Create showcase comparison images for portfolio/recruiter presentation
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def create_comparison_grid(input_dir, enhanced_dir, output_dir, num_samples=6):
    """
    Create side-by-side comparison grid for showcase
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    input_images = sorted(list(Path(input_dir).glob('*.png')))[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 4))
    
    for idx, img_path in enumerate(input_images):
        # Load input image
        input_img = cv2.imread(str(img_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Load enhanced image
        enhanced_path = Path(enhanced_dir) / f"enhanced_{img_path.name}"
        enhanced_img = cv2.imread(str(enhanced_path))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx, 0].imshow(input_img)
        axes[idx, 0].set_title('Low-Light Input', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(enhanced_img)
        axes[idx, 1].set_title('Enhanced Output', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'showcase_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Showcase comparison saved to: {output_path / 'showcase_comparison.png'}")
    
    # Create individual comparisons for top 3
    for idx, img_path in enumerate(input_images[:3]):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Load images
        input_img = cv2.imread(str(img_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        enhanced_path = Path(enhanced_dir) / f"enhanced_{img_path.name}"
        enhanced_img = cv2.imread(str(enhanced_path))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[0].imshow(input_img)
        axes[0].set_title('Low-Light Input', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_img)
        axes[1].set_title('Enhanced with HybridRetinexNet', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f'comparison_{idx+1}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Individual comparison {idx+1} saved")


def create_metrics_report(output_dir):
    """
    Create visual metrics report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training metrics (from the completed training)
    epochs = list(range(25))
    psnr_values = [
        12.72, 12.59, 18.18, 16.62, 18.11, 16.51, 17.04, 17.09, 16.97, 17.26,
        18.01, 18.04, 17.71, 17.41, 17.18, 17.55, 18.29, 18.05, 17.47, 17.52,
        17.66, 17.76, 17.66, 17.79, 17.85
    ]
    
    ssim_values = [
        0.5378, 0.5771, 0.7132, 0.6540, 0.7143, 0.6773, 0.6920, 0.6840, 0.6797, 0.6912,
        0.7239, 0.7153, 0.7119, 0.6995, 0.6901, 0.7013, 0.7183, 0.7148, 0.6948, 0.7020,
        0.7043, 0.7052, 0.7036, 0.7059, 0.7077
    ]
    
    # Create metrics plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR plot
    axes[0].plot(epochs, psnr_values, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].axhline(y=max(psnr_values), color='r', linestyle='--', alpha=0.7, 
                    label=f'Best: {max(psnr_values):.2f} dB')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Peak Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # SSIM plot
    axes[1].plot(epochs, ssim_values, 'g-', linewidth=2, marker='s', markersize=4)
    axes[1].axhline(y=max(ssim_values), color='r', linestyle='--', alpha=0.7,
                    label=f'Best: {max(ssim_values):.4f}')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_title('Structural Similarity Index', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_metrics.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training metrics plot saved to: {output_path / 'training_metrics.png'}")
    
    # Create summary card
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
    HybridRetinexNet - Training Results Summary
    
    Best Performance:
    • PSNR: {max(psnr_values):.2f} dB (Epoch 16)
    • SSIM: {max(ssim_values):.4f} (Epoch 10)
    
    Final Metrics:
    • PSNR: {psnr_values[-1]:.2f} dB
    • SSIM: {ssim_values[-1]:.4f}
    
    Model Configuration:
    • Architecture: Hybrid Retinex + CNN
    • Parameters: 169,299
    • Training Epochs: 25
    • Dataset: LOL (436 train, 49 val)
    • Learning Rate: 0.001
    • Batch Size: 16
    
    Key Features:
    • Physics-informed deep learning approach
    • Balances classical Retinex theory with CNN power
    • Lightweight architecture for efficient inference
    • Multi-scale feature extraction
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path / 'results_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Results summary saved to: {output_path / 'results_summary.png'}")


if __name__ == '__main__':
    # Create showcase comparisons
    create_comparison_grid(
        input_dir='lol_dataset/eval15/low',
        enhanced_dir='results/demo_outputs',
        output_dir='results/showcase'
    )
    
    # Create metrics report
    create_metrics_report('results/showcase')
    
    print("\n" + "="*60)
    print("SHOWCASE MATERIALS CREATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  • results/showcase/showcase_comparison.png - Grid of 6 before/after")
    print("  • results/showcase/comparison_1.png - Individual comparison 1")
    print("  • results/showcase/comparison_2.png - Individual comparison 2")
    print("  • results/showcase/comparison_3.png - Individual comparison 3")
    print("  • results/showcase/training_metrics.png - PSNR & SSIM curves")
    print("  • results/showcase/results_summary.png - Summary card")
    print("\nUse these for:")
    print("  ✓ README.md visual demonstrations")
    print("  ✓ Portfolio presentations")
    print("  ✓ GitHub repository showcase")
    print("  ✓ Recruiter discussions")
