"""
Common utility functions for the project
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get computing device (GPU if available, else CPU)
    
    Args:
        gpu_id: Specific GPU ID to use (optional)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: dict,
    save_dir: str,
    is_best: bool = False
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        metrics: Dictionary of metrics
        save_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest checkpoint
    checkpoint_path = save_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = save_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved (epoch {epoch})")
    
    # Save periodic checkpoint
    if epoch % 10 == 0:
        periodic_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, periodic_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✓ Checkpoint loaded from epoch {epoch}")
    
    return {'epoch': epoch, 'metrics': metrics}


def count_files(directory: str, extensions: list = ['.png', '.jpg', '.jpeg']) -> int:
    """
    Count files with specific extensions in a directory
    
    Args:
        directory: Directory path
        extensions: List of file extensions to count
    
    Returns:
        Number of files
    """
    count = 0
    for ext in extensions:
        count += len(list(Path(directory).glob(f'*{ext}')))
    return count


def create_experiment_dir(base_dir: str = 'experiments', exp_name: Optional[str] = None) -> Path:
    """
    Create a new experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        exp_name: Experiment name (optional)
    
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if exp_name:
        dir_name = f"{exp_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    """Test utility functions"""
    print("Testing common utilities...")
    
    # Test seed setting
    set_seed(42)
    
    # Test device
    device = get_device()
    
    # Test average meter
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    print(f"Average: {meter.avg:.2f} (should be 2.00)")
    
    print("✓ All utilities working!")
