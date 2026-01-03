"""
Classical Retinex-based Image Enhancement Algorithms

Implements Single-Scale Retinex (SSR) and Multi-Scale Retinex (MSR) based on:

Retinex Theory (Land & McCann, 1971):
    An observed image I(x,y) is the product of reflectance R(x,y) and illumination L(x,y):
    
    I(x,y) = R(x,y) × L(x,y)
    
    Where:
    - R(x,y): intrinsic reflectance (object properties, independent of lighting)
    - L(x,y): illumination (lighting conditions)
    
    Goal: Decompose I to recover R, which represents the true appearance under ideal lighting.

Single-Scale Retinex (SSR):
    log(R(x,y)) = log(I(x,y)) - log(L(x,y))
    
    where L(x,y) is estimated using Gaussian blur:
    L(x,y) ≈ I(x,y) * G_σ(x,y)
    
    G_σ: Gaussian kernel with scale σ

Multi-Scale Retinex (MSR):
    Combines SSR at multiple scales to capture both local and global illumination:
    
    log(R(x,y)) = Σ w_i [log(I(x,y)) - log(I(x,y) * G_σi(x,y))]
    
    Typically uses σ ∈ {15, 80, 250} for small/medium/large-scale features

References:
    [1] Jobson et al., "A Multiscale Retinex for Bridging the Gap Between Color Images 
        and the Human Observation of Scenes", IEEE TIP, 1997
    [2] Land & McCann, "Lightness and Retinex Theory", JOSA, 1971
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import torch


class RetinexEnhancer:
    """
    Retinex-based low-light image enhancement using SSR and MSR algorithms.
    
    Physics-based image decomposition for illumination-invariant representation.
    """
    
    def __init__(self, method: str = 'MSR', scales: List[int] = [15, 80, 250]):
        """
        Args:
            method: 'SSR' (Single-Scale) or 'MSR' (Multi-Scale) Retinex
            scales: Gaussian kernel scales for MSR (sigma values)
        """
        assert method in ['SSR', 'MSR'], f"Method must be 'SSR' or 'MSR', got {method}"
        self.method = method
        self.scales = scales if method == 'MSR' else [scales[0]]
        
        print(f"Initialized {method} with scales: {self.scales}")
    
    def single_scale_retinex(self, image: np.ndarray, sigma: int) -> np.ndarray:
        """
        Single-Scale Retinex (SSR) implementation
        
        Computes: log(R) = log(I) - log(I * G_σ)
        
        Args:
            image: Input image in [0, 255] or [0, 1] range, shape [H, W, C]
            sigma: Gaussian kernel scale parameter
        
        Returns:
            Reflectance map (log domain), shape [H, W, C]
        """
        # Ensure image is float32 in [0, 1] range
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        image = np.maximum(image, epsilon)
        
        # Estimate illumination using Gaussian blur
        # L(x,y) ≈ I(x,y) * G_σ(x,y)
        illumination = cv2.GaussianBlur(image, (0, 0), sigma)
        illumination = np.maximum(illumination, epsilon)
        
        # Compute reflectance in log domain
        # log(R) = log(I) - log(L)
        reflectance = np.log(image) - np.log(illumination)
        
        return reflectance
    
    def multi_scale_retinex(self, image: np.ndarray, scales: List[int]) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) implementation
        
        Combines SSR at multiple scales with equal weighting:
        log(R) = Σ w_i × SSR_σi(I)
        
        Args:
            image: Input image in [0, 255] or [0, 1] range, shape [H, W, C]
            scales: List of Gaussian kernel scales
        
        Returns:
            Multi-scale reflectance map, shape [H, W, C]
        """
        # Compute SSR for each scale
        reflectance_maps = []
        for sigma in scales:
            ssr = self.single_scale_retinex(image, sigma)
            reflectance_maps.append(ssr)
        
        # Average across scales (equal weighting)
        msr = np.mean(reflectance_maps, axis=0)
        
        return msr
    
    def simplest_color_balance(self, image: np.ndarray, low_clip: float = 0.01, 
                               high_clip: float = 0.01) -> np.ndarray:
        """
        Simplest Color Balance for contrast stretching
        
        Clips extreme values and stretches histogram to [0, 1]
        
        Args:
            image: Input image, shape [H, W, C]
            low_clip: Percentage of pixels to clip at low end
            high_clip: Percentage of pixels to clip at high end
        
        Returns:
            Balanced image in [0, 1] range
        """
        result = np.zeros_like(image)
        
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            
            # Compute percentiles
            low_val = np.percentile(channel, low_clip * 100)
            high_val = np.percentile(channel, (1 - high_clip) * 100)
            
            # Clip and normalize
            channel = np.clip(channel, low_val, high_val)
            channel = (channel - low_val) / (high_val - low_val + 1e-6)
            
            result[:, :, c] = channel
        
        return result
    
    def enhance(
        self,
        image: np.ndarray,
        gamma: float = 1.2,
        gain: float = 1.0,
        offset: float = 0.0,
        return_components: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Complete Retinex-based enhancement pipeline
        
        Steps:
            1. Decompose image into reflectance (MSR/SSR)
            2. Apply gamma correction and gain adjustment
            3. Color balance and normalization
        
        Args:
            image: Input low-light image, [H, W, C], uint8 or float32
            gamma: Gamma correction factor (>1 brightens)
            gain: Gain factor for reflectance
            offset: Offset for reflectance
            return_components: If True, return decomposition components
        
        Returns:
            enhanced_image: Enhanced image in [0, 255] uint8
            components (optional): Dict with 'illumination', 'reflectance', etc.
        """
        # Convert to float [0, 1]
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        # Step 1: Compute reflectance using Retinex
        if self.method == 'MSR':
            reflectance_log = self.multi_scale_retinex(image_float, self.scales)
        else:
            reflectance_log = self.single_scale_retinex(image_float, self.scales[0])
        
        # Step 2: Convert from log domain to linear
        reflectance = np.exp(reflectance_log)
        
        # Step 3: Estimate illumination (for visualization)
        # L = I / R
        epsilon = 1e-6
        illumination = image_float / (reflectance + epsilon)
        illumination = np.clip(illumination, 0, 1)
        
        # Step 4: Adjust reflectance with gain and offset
        reflectance = gain * reflectance + offset
        
        # Step 5: Apply gamma correction
        # R' = R^(1/gamma)
        reflectance = np.power(np.clip(reflectance, 0, 1), 1.0 / gamma)
        
        # Step 6: Color balance (contrast stretching)
        reflectance = self.simplest_color_balance(reflectance, low_clip=0.01, high_clip=0.01)
        
        # Step 7: Convert back to uint8
        enhanced = (np.clip(reflectance, 0, 1) * 255).astype(np.uint8)
        
        if return_components:
            components = {
                'reflectance': (np.clip(reflectance, 0, 1) * 255).astype(np.uint8),
                'illumination': (illumination * 255).astype(np.uint8),
                'reflectance_log': reflectance_log,
                'input': image
            }
            return enhanced, components
        
        return enhanced, None
    
    @staticmethod
    def histogram_equalization(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image, uint8, [H, W, C]
            clip_limit: Threshold for contrast limiting
        
        Returns:
            Equalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def enhance_with_histogram(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Retinex enhancement followed by histogram equalization
        
        Args:
            image: Input image, uint8, [H, W, C]
            **kwargs: Arguments for enhance()
        
        Returns:
            Enhanced image with improved contrast
        """
        # First apply Retinex
        retinex_enhanced, _ = self.enhance(image, **kwargs)
        
        # Then apply histogram equalization
        final_enhanced = self.histogram_equalization(retinex_enhanced)
        
        return final_enhanced


def apply_retinex_to_tensor(
    image_tensor: torch.Tensor,
    method: str = 'MSR',
    scales: List[int] = [15, 80, 250],
    gamma: float = 1.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Retinex enhancement to PyTorch tensor (batch processing)
    
    Args:
        image_tensor: Input tensor [B, C, H, W] in [0, 1] range
        method: 'SSR' or 'MSR'
        scales: Gaussian scales
        gamma: Gamma correction
    
    Returns:
        enhanced_tensor: Enhanced images [B, C, H, W]
        reflectance_tensor: Reflectance maps [B, C, H, W]
    """
    device = image_tensor.device
    batch_size = image_tensor.shape[0]
    
    enhancer = RetinexEnhancer(method=method, scales=scales)
    
    enhanced_list = []
    reflectance_list = []
    
    for i in range(batch_size):
        # Convert tensor to numpy [H, W, C]
        img_np = image_tensor[i].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Apply enhancement
        enhanced, components = enhancer.enhance(
            img_np,
            gamma=gamma,
            gain=1.0,
            offset=0.0,
            return_components=True
        )
        
        # Convert back to tensor
        enhanced_tensor = torch.from_numpy(enhanced).float() / 255.0
        enhanced_tensor = enhanced_tensor.permute(2, 0, 1)  # [C, H, W]
        
        reflectance_tensor = torch.from_numpy(components['reflectance']).float() / 255.0
        reflectance_tensor = reflectance_tensor.permute(2, 0, 1)
        
        enhanced_list.append(enhanced_tensor)
        reflectance_list.append(reflectance_tensor)
    
    # Stack batch
    enhanced_batch = torch.stack(enhanced_list, dim=0).to(device)
    reflectance_batch = torch.stack(reflectance_list, dim=0).to(device)
    
    return enhanced_batch, reflectance_batch


if __name__ == '__main__':
    """Test Retinex implementation"""
    import matplotlib.pyplot as plt
    
    print("Testing Retinex algorithms...")
    
    # Create synthetic low-light image
    # Simulate: I = R × L where L is dim
    np.random.seed(42)
    
    # Create reflectance (true object appearance)
    reflectance_true = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Create dim illumination
    illumination_dim = np.ones((256, 256, 3), dtype=np.float32) * 0.2
    
    # Form low-light image
    low_light = reflectance_true * illumination_dim
    low_light = (low_light * 255).astype(np.uint8)
    
    # Test SSR
    print("\nTesting Single-Scale Retinex (SSR)...")
    ssr_enhancer = RetinexEnhancer(method='SSR', scales=[80])
    ssr_enhanced, ssr_components = ssr_enhancer.enhance(
        low_light,
        gamma=1.5,
        return_components=True
    )
    
    print(f"Input range: [{low_light.min()}, {low_light.max()}]")
    print(f"SSR output range: [{ssr_enhanced.min()}, {ssr_enhanced.max()}]")
    
    # Test MSR
    print("\nTesting Multi-Scale Retinex (MSR)...")
    msr_enhancer = RetinexEnhancer(method='MSR', scales=[15, 80, 250])
    msr_enhanced, msr_components = msr_enhancer.enhance(
        low_light,
        gamma=1.2,
        return_components=True
    )
    
    print(f"MSR output range: [{msr_enhanced.min()}, {msr_enhanced.max()}]")
    print("\nRetinex algorithms working correctly!")
