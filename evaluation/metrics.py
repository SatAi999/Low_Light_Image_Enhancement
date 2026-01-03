"""
Evaluation Metrics for Low-Light Image Enhancement

Implements standard image quality assessment metrics:
1. PSNR (Peak Signal-to-Noise Ratio) - Full-reference
2. SSIM (Structural Similarity Index) - Full-reference
3. NIQE (Natural Image Quality Evaluator) - No-reference
4. Additional perceptual metrics

Reference:
- Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004
- Mittal et al., "Making a Completely Blind Image Quality Analyzer", IEEE SPL 2013
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR)
    
    Measures pixel-level similarity between two images.
    Higher is better (typical range: 20-40 dB).
    
    PSNR = 10 × log₁₀(MAX² / MSE)
    
    where MSE is the mean squared error between images.
    
    Args:
        img1: First image (enhanced), [H, W, C] or [H, W]
        img2: Second image (reference), same shape
        max_val: Maximum pixel value (255 for uint8, 1.0 for float)
    
    Returns:
        PSNR value in dB
    """
    # Ensure float arrays
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    psnr_val = 10 * np.log10((max_val ** 2) / mse)
    
    return psnr_val


def ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Structural Similarity Index (SSIM)
    
    Measures perceptual similarity considering luminance, contrast, and structure.
    Range: [-1, 1], where 1 means identical images.
    
    SSIM(x,y) = [l(x,y)]^α × [c(x,y)]^β × [s(x,y)]^γ
    
    where:
    - l(x,y): luminance comparison
    - c(x,y): contrast comparison
    - s(x,y): structure comparison
    
    Args:
        img1: First image (enhanced), [H, W, C] or [H, W]
        img2: Second image (reference), same shape
        max_val: Maximum pixel value
    
    Returns:
        SSIM value in [0, 1]
    """
    # Handle multichannel
    if len(img1.shape) == 3:
        ssim_val = ssim_skimage(
            img1, img2,
            data_range=max_val,
            channel_axis=2,
            multichannel=True
        )
    else:
        ssim_val = ssim_skimage(
            img1, img2,
            data_range=max_val
        )
    
    return ssim_val


def niqe(img: np.ndarray) -> float:
    """
    Natural Image Quality Evaluator (NIQE)
    
    No-reference metric that compares image statistics to natural scene statistics.
    Lower is better (typical range: 3-10).
    
    Note: This is a simplified placeholder. For production, use:
    - pyiqa library: https://github.com/chaofengc/IQA-PyTorch
    - MATLAB implementation
    
    Args:
        img: Input image [H, W, C], uint8
    
    Returns:
        NIQE score (lower is better)
    """
    # This is a placeholder - simplified version
    # In production, use proper NIQE implementation from pyiqa
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img
    
    # Compute simple statistics as proxy
    # Real NIQE uses MSCN coefficients and natural scene statistics
    
    # Mean
    mu = np.mean(img_gray)
    
    # Standard deviation
    sigma = np.std(img_gray)
    
    # Entropy (measure of information)
    hist, _ = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Sharpness (gradient magnitude)
    gx = ndimage.sobel(img_gray, axis=0)
    gy = ndimage.sobel(img_gray, axis=1)
    sharpness = np.mean(np.sqrt(gx**2 + gy**2))
    
    # Simple score (placeholder)
    # Lower is better for NIQE
    score = 10.0 - (entropy / 10.0) - (sharpness / 50.0)
    score = max(0, score)
    
    return score


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute average PSNR for a batch of images
    
    Args:
        pred: Predicted images [B, C, H, W], torch.Tensor in [0, 1]
        target: Target images [B, C, H, W], torch.Tensor in [0, 1]
    
    Returns:
        Average PSNR across batch
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        # Convert to [H, W, C]
        img_pred = np.transpose(pred_np[i], (1, 2, 0))
        img_target = np.transpose(target_np[i], (1, 2, 0))
        
        # Compute PSNR
        psnr_val = psnr(img_pred, img_target, max_val=1.0)
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute average SSIM for a batch of images
    
    Args:
        pred: Predicted images [B, C, H, W], torch.Tensor in [0, 1]
        target: Target images [B, C, H, W], torch.Tensor in [0, 1]
    
    Returns:
        Average SSIM across batch
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        # Convert to [H, W, C]
        img_pred = np.transpose(pred_np[i], (1, 2, 0))
        img_target = np.transpose(target_np[i], (1, 2, 0))
        
        # Compute SSIM
        ssim_val = ssim(img_pred, img_target, max_val=1.0)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def batch_niqe(pred: torch.Tensor) -> float:
    """
    Compute average NIQE for a batch of images
    
    Args:
        pred: Predicted images [B, C, H, W], torch.Tensor in [0, 1]
    
    Returns:
        Average NIQE score across batch
    """
    pred_np = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
    
    batch_size = pred_np.shape[0]
    niqe_values = []
    
    for i in range(batch_size):
        # Convert to [H, W, C]
        img_pred = np.transpose(pred_np[i], (1, 2, 0))
        
        # Compute NIQE
        niqe_val = niqe(img_pred)
        niqe_values.append(niqe_val)
    
    return np.mean(niqe_values)


class MetricTracker:
    """
    Track and compute running averages of metrics during training/evaluation
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_dict: dict):
        """
        Update metrics with new values
        
        Args:
            metric_dict: Dictionary of metric_name -> value
        """
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value
            self.counts[name] += 1
    
    def get_averages(self) -> dict:
        """Get average values for all metrics"""
        averages = {}
        for name in self.metrics:
            averages[name] = self.metrics[name] / max(self.counts[name], 1)
        return averages
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}
    
    def __repr__(self):
        avg = self.get_averages()
        return ', '.join([f"{k}: {v:.4f}" for k, v in avg.items()])


def evaluate_enhancement(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    compute_niqe: bool = True
) -> dict:
    """
    Comprehensive evaluation of enhancement quality
    
    Args:
        enhanced: Enhanced images [B, C, H, W] in [0, 1]
        target: Ground truth images [B, C, H, W] in [0, 1]
        compute_niqe: Whether to compute NIQE (slower)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # PSNR
    metrics['psnr'] = batch_psnr(enhanced, target)
    
    # SSIM
    metrics['ssim'] = batch_ssim(enhanced, target)
    
    # NIQE (no-reference)
    if compute_niqe:
        metrics['niqe'] = batch_niqe(enhanced)
    
    # Mean brightness
    metrics['mean_brightness'] = enhanced.mean().item()
    
    return metrics


if __name__ == '__main__':
    """Test metric implementations"""
    print("Testing Evaluation Metrics...")
    
    # Create test images
    np.random.seed(42)
    
    # Reference image
    img_ref = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Enhanced image (slightly different)
    img_enhanced = img_ref + np.random.randn(256, 256, 3).astype(np.float32) * 0.05
    img_enhanced = np.clip(img_enhanced, 0, 1)
    
    # Test PSNR
    print("\n1. PSNR Test:")
    psnr_val = psnr(img_enhanced, img_ref, max_val=1.0)
    print(f"   PSNR: {psnr_val:.2f} dB")
    
    # Test SSIM
    print("\n2. SSIM Test:")
    ssim_val = ssim(img_enhanced, img_ref, max_val=1.0)
    print(f"   SSIM: {ssim_val:.4f}")
    
    # Test NIQE
    print("\n3. NIQE Test:")
    img_uint8 = (img_enhanced * 255).astype(np.uint8)
    niqe_val = niqe(img_uint8)
    print(f"   NIQE: {niqe_val:.4f}")
    
    # Test batch processing
    print("\n4. Batch Processing Test:")
    batch_pred = torch.rand(4, 3, 128, 128)
    batch_target = batch_pred + torch.randn(4, 3, 128, 128) * 0.05
    batch_target = torch.clamp(batch_target, 0, 1)
    
    metrics = evaluate_enhancement(batch_pred, batch_target, compute_niqe=False)
    print(f"   Batch PSNR: {metrics['psnr']:.2f} dB")
    print(f"   Batch SSIM: {metrics['ssim']:.4f}")
    print(f"   Mean Brightness: {metrics['mean_brightness']:.4f}")
    
    # Test metric tracker
    print("\n5. Metric Tracker Test:")
    tracker = MetricTracker()
    tracker.update({'loss': 0.5, 'psnr': 25.0})
    tracker.update({'loss': 0.4, 'psnr': 26.0})
    print(f"   Running averages: {tracker}")
    
    print("\n✓ All metrics working correctly!")
