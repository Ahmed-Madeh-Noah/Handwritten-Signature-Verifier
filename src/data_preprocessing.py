# -*- coding: utf-8 -*-

"""
CEDAR Signature Dataset Preprocessing Pipeline
==============================================

This module handles preprocessing for the CEDAR signature verification dataset.

USE CASE 1: Generating Dataset Exports for Model Training
----------------------------------------------------------
Run the entire notebook/script to:
1. Load raw CEDAR signatures from genuine_dir and forged_dir
2. Apply preprocessing pipeline (blur → threshold → resize)
3. Create writer-independent train/val/test splits (70/15/15)
4. Save splits.pkl containing file paths and labels for each split
5. Create datasets and dataloaders ready for model training

Key outputs:
- splits.pkl: Contains train/val/test file lists and writer assignments
- train_loader, val_loader, test_loader: PyTorch dataloaders for training

USE CASE 2: Preprocessing Input for Deployment
-----------------------------------------------
For inference on new signatures:
1. Import the preprocess_signature() function
2. Call: preprocessed_img = preprocess_signature(img_path, target_size=(150, 220))
3. Apply same normalization transforms used during training
4. Feed to trained model for prediction

Example:
    from this_module import preprocess_signature
    import torchvision.transforms as transforms
    
    # Preprocess new signature
    img = preprocess_signature('new_signature.png', (150, 220))
    
    # Apply training transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Run inference
    prediction = model(tensor)

Note: For deployment, you only need preprocess_signature() and the normalization
transforms. The dataset splitting and dataloader creation are training-only steps.
"""

import os
import cv2
import pickle
import random
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms

# Constants
SEED = 42

# Configuration
CONFIG = {
    'data_dir': '../datasets/CEDAR',
    'genuine_dir': '../datasets/CEDAR/original',
    'forged_dir': '../datasets/CEDAR/forged',
    'img_size': (150, 220),  # (height, width)
    'batch_size': 32,
    'num_workers': 2,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}


# Dataset Verification

def verify_dataset(genuine_dir, forged_dir):
    """Verify dataset exists and count samples."""
    if not os.path.exists(genuine_dir):
        raise FileNotFoundError(f"Genuine directory not found: {genuine_dir}")
    if not os.path.exists(forged_dir):
        raise FileNotFoundError(f"Forged directory not found: {forged_dir}")

    genuine_files = [f for f in os.listdir(genuine_dir) if f.endswith('.png')]
    forged_files = [f for f in os.listdir(forged_dir) if f.endswith('.png')]

    print("=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)
    print(f"Genuine signatures: {len(genuine_files)}")
    print(f"Forged signatures:  {len(forged_files)}")
    print(f"Total samples:      {len(genuine_files) + len(forged_files)}")
    print("=" * 50)

    writers = set()
    for f in genuine_files:
        parts = f.replace('.png', '').split('_')
        if len(parts) >= 2:
            writers.add(int(parts[1]))

    print(f"Number of writers: {len(writers)}")
    print(f"Expected: 55 writers × 24 samples × 2 types = 2640 total")
    print("=" * 50)

    return genuine_files, forged_files, sorted(writers)


# Image Preprocessing Functions

def preprocess_signature(img_path, target_size=(150, 220)):
    """
    Preprocess a signature image using OpenCV

    Pipeline:
    1. Load grayscale
    2. Apply Gaussian blur
    3. Otsu's thresholding
    4. Resize to target size
    5. Convert to PIL Image

    Args:
        img_path: Path to image file
        target_size: (height, width) tuple

    Returns:
        PIL Image (mode 'L' for grayscale)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (target_size[1], target_size[0]),
                     interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(img)

    return pil_img


# Cedar Signature Dataset Class

class CedarSignatureDataset(data.Dataset):
    """
    Cedar Signature Dataset for verification task

    Following curriculum pattern:
    - Inherits from data.Dataset
    - Implements __init__, __len__, __getitem__
    """

    def __init__(self, file_list, img_size=(150, 220), transform=None):
        """
        Initialize the dataset

        Args:
            file_list: List of tuples (file_path, label)
            img_size: Target image size (height, width)
            transform: Optional torchvision transforms
        """
        super().__init__()
        self.file_list = file_list
        self.img_size = img_size
        self.transform = transform
        self.size = len(file_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Get a single sample

        Returns:
            data_point: Image tensor
            data_label: Label tensor (0=forged, 1=genuine)
        """
        try:
            img_path, label = self.file_list[idx]
            img = preprocess_signature(img_path, self.img_size)

            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
                
            if self.transform:
                data_point = self.transform(img)
            else:
                data_point = transforms.ToTensor()(img)

            data_label = torch.tensor(label, dtype=torch.long)

            filename = os.path.basename(img_path)
            parts = filename.replace('.png', '').split('_')
            writer_id = int(parts[1])

            return data_point, data_label, writer_id
        except Exception as e:
            print(f"Error loading data at index {idx}: {str(e)}")
            print(f"Skipping index {idx}")
            return self.__getitem__((idx + 1) % len(self.file_list))


# Transforms

def get_train_transform():
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=255
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_val_transform():
    """Validation/Test transforms without augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# Create Writer-Independent Splits

def create_splits(genuine_dir, forged_dir, writers, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits by writer (writer-independent)

    Args:
        genuine_dir: Path to genuine signatures
        forged_dir: Path to forged signatures
        writers: List of writer IDs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation

    Returns:
        Dictionary with train/val/test file lists
    """
    writers_shuffled = writers.copy()
    random.shuffle(writers_shuffled)

    n_writers = len(writers_shuffled)
    n_train = int(n_writers * train_ratio)
    n_val = int(n_writers * val_ratio)

    train_writers = set(writers_shuffled[:n_train])
    val_writers = set(writers_shuffled[n_train:n_train + n_val])
    test_writers = set(writers_shuffled[n_train + n_val:])

    print(f"Writers split: Train={len(train_writers)}, Val={len(val_writers)}, Test={len(test_writers)}")

    splits = {'train': [], 'val': [], 'test': []}

    for filename in os.listdir(genuine_dir):
        if not filename.endswith('.png'):
            continue
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 2:
            writer_id = int(parts[1])
            file_path = os.path.join(genuine_dir, filename)

            if writer_id in train_writers:
                splits['train'].append((file_path, 1))
            elif writer_id in val_writers:
                splits['val'].append((file_path, 1))
            else:
                splits['test'].append((file_path, 1))

    for filename in os.listdir(forged_dir):
        if not filename.endswith('.png'):
            continue
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 2:
            writer_id = int(parts[1])
            file_path = os.path.join(forged_dir, filename)

            if writer_id in train_writers:
                splits['train'].append((file_path, 0))
            elif writer_id in val_writers:
                splits['val'].append((file_path, 0))
            else:
                splits['test'].append((file_path, 0))

    for key in splits:
        random.shuffle(splits[key])

    splits['writer_assignments'] = {
        'train': sorted(train_writers),
        'val': sorted(val_writers),
        'test': sorted(test_writers)
    }

    return splits


# Dataset Statistics

def compute_statistics(splits):
    """Compute class distribution for each split."""
    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    for split_name in ['train', 'val', 'test']:
        samples = splits[split_name]
        genuine = sum(1 for _, label in samples if label == 1)
        forged = sum(1 for _, label in samples if label == 0)
        total = len(samples)

        print(f"\n{split_name.upper()}:")
        print(f"  Genuine: {genuine} ({100*genuine/total:.1f}%)")
        print(f"  Forged:  {forged} ({100*forged/total:.1f}%)")
        print(f"  Total:   {total}")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)

    print("Configuration loaded successfully")
    print(f"Image size: {CONFIG['img_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")

    # Verify dataset
    genuine_files, forged_files, writers = verify_dataset(
        CONFIG['genuine_dir'],
        CONFIG['forged_dir']
    )

    print("Preprocessing functions defined")
    print("Pipeline: Load → Blur → Threshold → Resize → PIL")

    print("CedarSignatureDataset class defined")
    print("Following curriculum pattern: __init__, __len__, __getitem__")

    # Define transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    print("Transforms defined:")
    print("- Training: Affine augmentation + Normalize")
    print("- Validation/Test: Normalize only")

    # Create splits
    splits = create_splits(
        CONFIG['genuine_dir'],
        CONFIG['forged_dir'],
        writers,
        CONFIG['train_ratio'],
        CONFIG['val_ratio']
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")

    # Create datasets
    train_dataset = CedarSignatureDataset(
        splits['train'],
        img_size=CONFIG['img_size'],
        transform=train_transform
    )

    val_dataset = CedarSignatureDataset(
        splits['val'],
        img_size=CONFIG['img_size'],
        transform=val_transform
    )

    test_dataset = CedarSignatureDataset(
        splits['test'],
        img_size=CONFIG['img_size'],
        transform=val_transform
    )

    print("Datasets created:")
    print(f"  train_dataset: {len(train_dataset)} samples")
    print(f"  val_dataset:   {len(val_dataset)} samples")
    print(f"  test_dataset:  {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("DataLoaders created:")
    print(f"  train_loader: {len(train_loader)} batches")
    print(f"  val_loader:   {len(val_loader)} batches")
    print(f"  test_loader:  {len(test_loader)} batches")
    print(f"\nBatch size: {CONFIG['batch_size']}")

    # Compute statistics
    compute_statistics(splits)
