"""
Quick Start Script - Run this to test the project setup

Tests:
1. Dataset loading
2. Retinex algorithms
3. Model architectures
4. Loss functions
5. Metrics
"""

import sys
import os

print("="*60)
print("Low-Light Image Enhancement - Quick Test")
print("="*60)

# Test 1: Dataset
print("\n[1/5] Testing dataset loader...")
try:
    from data import get_dataloader
    
    train_loader = get_dataloader(
        root_dir='lol_dataset',
        split='train',
        batch_size=2,
        image_size=256,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"✓ Dataset loaded: {len(train_loader)} batches")
    print(f"  - Low-light shape: {batch['low'].shape}")
    print(f"  - High-light shape: {batch['high'].shape}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")

# Test 2: Retinex
print("\n[2/5] Testing Retinex algorithms...")
try:
    from retinex import RetinexEnhancer
    import numpy as np
    
    # Test with synthetic image
    test_image = np.random.rand(256, 256, 3).astype(np.float32) * 0.3 * 255
    test_image = test_image.astype(np.uint8)
    
    # SSR
    ssr_enhancer = RetinexEnhancer(method='SSR', scales=[80])
    ssr_result, _ = ssr_enhancer.enhance(test_image, gamma=1.2)
    print(f"✓ SSR working: output shape {ssr_result.shape}")
    
    # MSR
    msr_enhancer = RetinexEnhancer(method='MSR', scales=[15, 80, 250])
    msr_result, _ = msr_enhancer.enhance(test_image, gamma=1.2)
    print(f"✓ MSR working: output shape {msr_result.shape}")
except Exception as e:
    print(f"✗ Retinex failed: {e}")

# Test 3: Models
print("\n[3/5] Testing deep learning models...")
try:
    import torch
    from models import LightweightUNet, DCENet, EnhancementCNN, HybridRetinexNet, count_parameters
    
    device = torch.device('cpu')
    x = torch.rand(1, 3, 256, 256)
    
    models_to_test = [
        ('LightweightUNet', LightweightUNet(base_features=32)),
        ('DCENet', DCENet(num_iterations=8)),
        ('EnhancementCNN', EnhancementCNN(base_channels=64)),
        ('HybridRetinexNet', HybridRetinexNet())
    ]
    
    for name, model in models_to_test:
        model.eval()
        with torch.no_grad():
            if name == 'DCENet':
                out, _ = model(x)
            else:
                out = model(x)
        params = count_parameters(model)
        print(f"✓ {name}: {params:,} parameters, output shape {out.shape}")
except Exception as e:
    print(f"✗ Models failed: {e}")

# Test 4: Loss Functions
print("\n[4/5] Testing loss functions...")
try:
    from training.losses import EnhancementLoss
    
    enhanced = torch.rand(2, 3, 128, 128)
    input_low = torch.rand(2, 3, 128, 128) * 0.3
    target = torch.rand(2, 3, 128, 128)
    
    criterion = EnhancementLoss(
        use_perceptual=False,
        use_reconstruction=True
    )
    
    total_loss, losses = criterion(enhanced, input_low, target)
    print(f"✓ Loss computation working")
    print(f"  - Total loss: {losses['total']:.4f}")
    print(f"  - Components: {list(losses.keys())}")
except Exception as e:
    print(f"✗ Loss functions failed: {e}")

# Test 5: Metrics
print("\n[5/5] Testing evaluation metrics...")
try:
    from evaluation.metrics import evaluate_enhancement
    
    pred = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)
    
    metrics = evaluate_enhancement(pred, target, compute_niqe=False)
    print(f"✓ Metrics computation working")
    print(f"  - PSNR: {metrics['psnr']:.2f} dB")
    print(f"  - SSIM: {metrics['ssim']:.4f}")
except Exception as e:
    print(f"✗ Metrics failed: {e}")

# Summary
print("\n" + "="*60)
print("Quick test completed!")
print("="*60)
print("\nNext steps:")
print("1. Train a model: python training/train.py --config configs/hybrid_retinex.yaml")
print("2. Test Retinex: python inference_demo.py retinex --input <image>")
print("3. Run inference: python inference_demo.py single --config <config> --checkpoint <ckpt> --input <image> --output <out>")
print("\nFor detailed instructions, see README.md")
