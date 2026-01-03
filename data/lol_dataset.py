"""
LOL Dataset Loader for Low-Light Image Enhancement
Supports paired low-light and normal-light images with augmentations
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from typing import Tuple, Optional


class LOLDataset(Dataset):
    """
    Low-Light (LOL) Dataset for paired low/normal light images.
    
    Dataset Structure:
        lol_dataset/
            our485/  (training set)
                low/
                high/
            eval15/  (evaluation set)
                low/
                high/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 256,
        augment: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            root_dir: Root directory of LOL dataset
            split: 'train', 'val', or 'test'
            image_size: Target image size for resizing
            augment: Apply data augmentation
            normalize: Normalize images to [0, 1]
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        
        # Determine data directory based on split
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'our485')
        else:  # val or test
            self.data_dir = os.path.join(root_dir, 'eval15')
        
        self.low_dir = os.path.join(self.data_dir, 'low')
        self.high_dir = os.path.join(self.data_dir, 'high')
        
        # Get image filenames
        self.low_images = sorted([f for f in os.listdir(self.low_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # For training, use 90% for train, 10% for validation
        if split == 'train':
            split_idx = int(0.9 * len(self.low_images))
            self.low_images = self.low_images[:split_idx]
        elif split == 'val':
            # Use our485 for validation too (last 10%)
            self.data_dir = os.path.join(root_dir, 'our485')
            self.low_dir = os.path.join(self.data_dir, 'low')
            self.high_dir = os.path.join(self.data_dir, 'high')
            all_images = sorted([f for f in os.listdir(self.low_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            split_idx = int(0.9 * len(all_images))
            self.low_images = all_images[split_idx:]
        
        print(f"Loaded {len(self.low_images)} {split} image pairs from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.low_images)
    
    def _get_transforms(self):
        """Get basic transforms for resize and tensor conversion"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def _augment_pair(self, low_img: Image.Image, high_img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized augmentation to paired images
        
        Augmentations:
            - Random horizontal flip
            - Random vertical flip
            - Random rotation (90, 180, 270 degrees)
            - Random crop and resize
        """
        # Convert to tensor first
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)
        
        # Random horizontal flip
        if random.random() > 0.5:
            low_tensor = TF.hflip(low_tensor)
            high_tensor = TF.hflip(high_tensor)
        
        # Random vertical flip
        if random.random() > 0.5:
            low_tensor = TF.vflip(low_tensor)
            high_tensor = TF.vflip(high_tensor)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            low_tensor = TF.rotate(low_tensor, angle)
            high_tensor = TF.rotate(high_tensor, angle)
        
        # Random crop (80% to 100% of image)
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                TF.to_pil_image(low_tensor), 
                scale=(0.8, 1.0), 
                ratio=(0.95, 1.05)
            )
            low_tensor = TF.resized_crop(TF.to_pil_image(low_tensor), i, j, h, w, 
                                         (self.image_size, self.image_size))
            high_tensor = TF.resized_crop(TF.to_pil_image(high_tensor), i, j, h, w, 
                                          (self.image_size, self.image_size))
            low_tensor = TF.to_tensor(low_tensor)
            high_tensor = TF.to_tensor(high_tensor)
        
        return low_tensor, high_tensor
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict: {
                'low': low-light image tensor [C, H, W],
                'high': normal-light image tensor [C, H, W],
                'filename': image filename
            }
        """
        low_filename = self.low_images[idx]
        
        # Find corresponding high-light image
        # Handle different naming conventions
        high_filename = low_filename
        
        low_path = os.path.join(self.low_dir, low_filename)
        high_path = os.path.join(self.high_dir, high_filename)
        
        # Load images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Resize to target size
        low_img = low_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        high_img = high_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Apply augmentation if enabled
        if self.augment:
            low_tensor, high_tensor = self._augment_pair(low_img, high_img)
        else:
            low_tensor = TF.to_tensor(low_img)
            high_tensor = TF.to_tensor(high_img)
        
        return {
            'low': low_tensor,
            'high': high_tensor,
            'filename': low_filename
        }


class UnpairedLowLightDataset(Dataset):
    """
    Unpaired low-light dataset for self-supervised learning
    (e.g., Zero-DCE style training without ground truth)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 256,
        augment: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Load only low-light images
        if split == 'train':
            self.low_dir = os.path.join(root_dir, 'our485', 'low')
        else:
            self.low_dir = os.path.join(root_dir, 'eval15', 'low')
        
        self.low_images = sorted([f for f in os.listdir(self.low_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Split for train/val
        if split == 'train':
            split_idx = int(0.9 * len(self.low_images))
            self.low_images = self.low_images[:split_idx]
        elif split == 'val':
            # Use our485 for validation
            self.low_dir = os.path.join(root_dir, 'our485', 'low')
            all_images = sorted([f for f in os.listdir(self.low_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            split_idx = int(0.9 * len(all_images))
            self.low_images = all_images[split_idx:]
        
        print(f"Loaded {len(self.low_images)} unpaired {split} low-light images")
    
    def __len__(self) -> int:
        return len(self.low_images)
    
    def __getitem__(self, idx: int) -> dict:
        low_filename = self.low_images[idx]
        low_path = os.path.join(self.low_dir, low_filename)
        
        low_img = Image.open(low_path).convert('RGB')
        low_img = low_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        low_tensor = TF.to_tensor(low_img)
        
        # Simple augmentation for unpaired data
        if self.augment and random.random() > 0.5:
            low_tensor = TF.hflip(low_tensor)
        
        return {
            'low': low_tensor,
            'filename': low_filename
        }


def get_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 8,
    image_size: int = 256,
    augment: bool = True,
    num_workers: int = 4,
    paired: bool = True
) -> DataLoader:
    """
    Create DataLoader for LOL dataset
    
    Args:
        root_dir: Dataset root directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        image_size: Target image size
        augment: Enable augmentation
        num_workers: Number of worker threads
        paired: Use paired dataset (with ground truth) or unpaired
    
    Returns:
        DataLoader instance
    """
    if paired:
        dataset = LOLDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            augment=augment
        )
    else:
        dataset = UnpairedLowLightDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            augment=augment
        )
    
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset loading
    print("Testing LOL Dataset Loader...")
    
    dataset_root = 'lol_dataset'
    
    # Test paired dataset
    train_loader = get_dataloader(
        root_dir=dataset_root,
        split='train',
        batch_size=4,
        image_size=256,
        augment=True,
        num_workers=0
    )
    
    print("\nTesting paired training data...")
    batch = next(iter(train_loader))
    print(f"Low-light batch shape: {batch['low'].shape}")
    print(f"High-light batch shape: {batch['high'].shape}")
    print(f"Filenames: {batch['filename']}")
    
    # Test unpaired dataset
    unpaired_loader = get_dataloader(
        root_dir=dataset_root,
        split='train',
        batch_size=4,
        image_size=256,
        augment=True,
        num_workers=0,
        paired=False
    )
    
    print("\nTesting unpaired training data...")
    batch = next(iter(unpaired_loader))
    print(f"Low-light batch shape: {batch['low'].shape}")
    print(f"Filenames: {batch['filename']}")
