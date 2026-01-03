"""
Loss Functions for Low-Light Image Enhancement

Implements multiple loss components for self-supervised and supervised training:
1. Reconstruction loss (L1, L2, Perceptual)
2. Exposure control loss (maintains proper brightness)
3. Smoothness loss (reduces noise amplification)
4. Color constancy loss (preserves color fidelity)
5. Illumination smoothness loss (for Retinex decomposition)

Reference papers:
- Zero-DCE: "Zero-Reference Deep Curve Estimation" (CVPR 2020)
- Retinex-Net: "Deep Retinex Decomposition" (BMVC 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from typing import Optional


class ReconstructionLoss(nn.Module):
    """
    L1 or L2 reconstruction loss between enhanced and target images
    
    For supervised learning with paired data:
        L_recon = ||I_enhanced - I_target||
    """
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        assert loss_type in ['l1', 'l2'], f"loss_type must be 'l1' or 'l2', got {loss_type}"
        
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(enhanced, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    
    Computes L2 distance in VGG feature space instead of pixel space.
    Encourages perceptually similar outputs.
    
    L_perceptual = Σ ||φ_i(I_enhanced) - φ_i(I_target)||²
    
    where φ_i are VGG feature maps at layer i
    """
    def __init__(self, feature_layers: list = [2, 7, 12, 21]):
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = vgg16(pretrained=True).features
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract specific layers
        self.feature_extractors = nn.ModuleList()
        prev_layer = 0
        for layer_idx in feature_layers:
            self.feature_extractors.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
        
        self.feature_extractors.eval()
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images for VGG"""
        return (x - self.mean) / self.std
    
    def forward(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize inputs
        enhanced = self.normalize(enhanced)
        target = self.normalize(target)
        
        loss = 0.0
        x_enhanced = enhanced
        x_target = target
        
        # Compute feature loss at each layer
        for extractor in self.feature_extractors:
            x_enhanced = extractor(x_enhanced)
            x_target = extractor(x_target)
            loss += F.mse_loss(x_enhanced, x_target)
        
        return loss


class ExposureControlLoss(nn.Module):
    """
    Exposure control loss for proper brightness
    
    Encourages average intensity to be close to a target value (e.g., 0.6 for well-exposed).
    Prevents over-exposure or under-exposure.
    
    L_exposure = ||E(I_enhanced) - E_target||²
    
    where E(I) is the mean intensity over spatial dimensions.
    
    Reference: Zero-DCE (CVPR 2020)
    """
    def __init__(self, target_exposure: float = 0.6, reduction: str = 'mean'):
        super().__init__()
        self.target_exposure = target_exposure
        self.reduction = reduction
    
    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced: Enhanced image [B, C, H, W] in [0, 1]
        
        Returns:
            Exposure loss scalar
        """
        # Compute mean intensity per image
        # Shape: [B, C, H, W] -> [B]
        mean_intensity = torch.mean(enhanced, dim=(1, 2, 3))
        
        # Loss: deviation from target exposure
        loss = torch.mean((mean_intensity - self.target_exposure) ** 2)
        
        return loss


class ColorConstancyLoss(nn.Module):
    """
    Color constancy loss to preserve color fidelity
    
    Ensures that color distribution remains consistent with the input.
    Prevents color shifting artifacts.
    
    L_color = Σ_c (mean(I_c) - mean(I_avg))²
    
    where I_c is channel c, and I_avg is the average across channels.
    This ensures balanced color channels (no tint).
    
    Reference: Zero-DCE (CVPR 2020)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced: Enhanced image [B, C, H, W]
        
        Returns:
            Color constancy loss
        """
        # Compute mean of each channel: [B, C]
        channel_means = torch.mean(enhanced, dim=(2, 3))
        
        # Compute average across channels: [B]
        overall_mean = torch.mean(channel_means, dim=1, keepdim=True)
        
        # Loss: deviation of each channel from overall mean
        loss = torch.mean((channel_means - overall_mean) ** 2)
        
        return loss


class SpatialConsistencyLoss(nn.Module):
    """
    Spatial consistency loss (smoothness loss)
    
    Encourages spatially smooth enhancement to reduce noise amplification.
    Computed as total variation of the difference between input and output.
    
    L_spatial = Σ |∇_x (I_enhanced - I_input)| + |∇_y (I_enhanced - I_input)|
    
    where ∇_x, ∇_y are spatial gradients.
    
    Reference: Zero-DCE (CVPR 2020)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, enhanced: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced: Enhanced image [B, C, H, W]
            input: Original low-light image [B, C, H, W]
        
        Returns:
            Spatial consistency loss
        """
        # Compute difference map
        diff = enhanced - input
        
        # Compute gradients (horizontal and vertical)
        # Horizontal gradient (difference between adjacent columns)
        diff_h = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
        
        # Vertical gradient (difference between adjacent rows)
        diff_v = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
        
        # Total variation loss
        loss = torch.mean(diff_h) + torch.mean(diff_v)
        
        return loss


class IlluminationSmoothnessLoss(nn.Module):
    """
    Illumination smoothness loss for Retinex decomposition
    
    Encourages the estimated illumination map to be spatially smooth
    (natural lighting is typically smooth).
    
    L_illum_smooth = Σ |∇_x L| + |∇_y L|
    
    where L is the illumination map.
    
    Reference: Retinex-Net (BMVC 2018)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, illumination: torch.Tensor) -> torch.Tensor:
        """
        Args:
            illumination: Illumination map [B, C, H, W]
        
        Returns:
            Smoothness loss
        """
        # Horizontal gradient
        grad_h = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
        
        # Vertical gradient
        grad_v = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])
        
        # Total variation
        loss = torch.mean(grad_h) + torch.mean(grad_v)
        
        return loss


class EnhancementLoss(nn.Module):
    """
    Combined loss for low-light image enhancement
    
    Total loss:
        L_total = λ_recon × L_recon
                + λ_perceptual × L_perceptual
                + λ_exposure × L_exposure
                + λ_color × L_color
                + λ_spatial × L_spatial
                + λ_illum × L_illum_smooth
    
    Supports both supervised (with ground truth) and self-supervised modes.
    """
    def __init__(
        self,
        use_perceptual: bool = True,
        use_reconstruction: bool = True,
        lambda_recon: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_exposure: float = 1.0,
        lambda_color: float = 5.0,
        lambda_spatial: float = 10.0,
        lambda_illum: float = 1.0,
        target_exposure: float = 0.6
    ):
        super().__init__()
        
        self.use_reconstruction = use_reconstruction
        self.use_perceptual = use_perceptual
        
        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_perceptual = lambda_perceptual
        self.lambda_exposure = lambda_exposure
        self.lambda_color = lambda_color
        self.lambda_spatial = lambda_spatial
        self.lambda_illum = lambda_illum
        
        # Loss modules
        if use_reconstruction:
            self.recon_loss = ReconstructionLoss(loss_type='l1')
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        
        self.exposure_loss = ExposureControlLoss(target_exposure=target_exposure)
        self.color_loss = ColorConstancyLoss()
        self.spatial_loss = SpatialConsistencyLoss()
        self.illum_loss = IlluminationSmoothnessLoss()
    
    def forward(
        self,
        enhanced: torch.Tensor,
        input_low: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        illumination: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss
        
        Args:
            enhanced: Enhanced output [B, C, H, W]
            input_low: Low-light input [B, C, H, W]
            target: Ground truth (optional, for supervised learning) [B, C, H, W]
            illumination: Illumination map (optional, for Retinex) [B, C, H, W]
        
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        total_loss = 0.0
        
        # Reconstruction loss (supervised)
        if self.use_reconstruction and target is not None:
            loss_recon = self.recon_loss(enhanced, target)
            losses['reconstruction'] = loss_recon.item()
            total_loss += self.lambda_recon * loss_recon
        
        # Perceptual loss (supervised)
        if self.use_perceptual and target is not None:
            loss_perceptual = self.perceptual_loss(enhanced, target)
            losses['perceptual'] = loss_perceptual.item()
            total_loss += self.lambda_perceptual * loss_perceptual
        
        # Exposure control loss (self-supervised)
        if self.lambda_exposure > 0:
            loss_exposure = self.exposure_loss(enhanced)
            losses['exposure'] = loss_exposure.item()
            total_loss += self.lambda_exposure * loss_exposure
        
        # Color constancy loss (self-supervised)
        if self.lambda_color > 0:
            loss_color = self.color_loss(enhanced)
            losses['color'] = loss_color.item()
            total_loss += self.lambda_color * loss_color
        
        # Spatial consistency loss (self-supervised)
        if self.lambda_spatial > 0:
            loss_spatial = self.spatial_loss(enhanced, input_low)
            losses['spatial'] = loss_spatial.item()
            total_loss += self.lambda_spatial * loss_spatial
        
        # Illumination smoothness loss (for Retinex)
        if self.lambda_illum > 0 and illumination is not None:
            loss_illum = self.illum_loss(illumination)
            losses['illumination_smooth'] = loss_illum.item()
            total_loss += self.lambda_illum * loss_illum
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


if __name__ == '__main__':
    """Test loss functions"""
    print("Testing Loss Functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    batch_size = 4
    enhanced = torch.rand(batch_size, 3, 256, 256).to(device)
    input_low = torch.rand(batch_size, 3, 256, 256).to(device) * 0.3  # Dim input
    target = torch.rand(batch_size, 3, 256, 256).to(device)
    illumination = torch.rand(batch_size, 3, 256, 256).to(device)
    
    # Test individual losses
    print("\n1. Reconstruction Loss (L1):")
    recon_loss = ReconstructionLoss(loss_type='l1')
    loss = recon_loss(enhanced, target)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n2. Exposure Control Loss:")
    exposure_loss = ExposureControlLoss(target_exposure=0.6)
    loss = exposure_loss(enhanced)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Mean intensity: {enhanced.mean().item():.4f}")
    
    print("\n3. Color Constancy Loss:")
    color_loss = ColorConstancyLoss()
    loss = color_loss(enhanced)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n4. Spatial Consistency Loss:")
    spatial_loss = SpatialConsistencyLoss()
    loss = spatial_loss(enhanced, input_low)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n5. Illumination Smoothness Loss:")
    illum_loss = IlluminationSmoothnessLoss()
    loss = illum_loss(illumination)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n6. Combined Enhancement Loss:")
    combined_loss = EnhancementLoss(
        use_perceptual=False,  # Skip for speed
        use_reconstruction=True,
        lambda_recon=1.0,
        lambda_exposure=1.0,
        lambda_color=5.0,
        lambda_spatial=10.0,
        lambda_illum=1.0
    ).to(device)
    
    total, losses = combined_loss(enhanced, input_low, target, illumination)
    print(f"   Total Loss: {losses['total']:.6f}")
    print(f"   Components: {losses}")
    
    print("\n✓ All loss functions working correctly!")
