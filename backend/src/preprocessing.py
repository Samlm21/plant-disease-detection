import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, List

from src import config

def get_train_transforms() -> transforms.Compose:
    """Returns torchvision transforms for training with image augmentation."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # Standard normalization for ImageNet-trained models (e.g., ResNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transforms() -> transforms.Compose:
    """Returns torchvision transforms for validation and testing (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class PlantDataset(Dataset):
    """Custom PyTorch Dataset for loading plant leaf images."""
    
    def __init__(self, root_dir: str, transform: transforms.Compose = None):
        """
        Args:
            root_dir (str): Path to the split directory (e.g. data/processed/train)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # OpenCV loads as BGR; Pillow loads as RGB. Use Pillow to stay aligned with torchvision norms
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback/logging if image is corrupted
            raise IOError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(
    processed_dir: str = config.DATA_PROCESSED_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")
    
    # Initialize Datasets
    train_dataset = PlantDataset(root_dir=train_dir, transform=get_train_transforms())
    val_dataset = PlantDataset(root_dir=val_dir, transform=get_val_test_transforms())
    test_dataset = PlantDataset(root_dir=test_dir, transform=get_val_test_transforms())
    
    class_names = train_dataset.classes
    
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, class_names
