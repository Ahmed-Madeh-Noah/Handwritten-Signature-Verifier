from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import random
import torchvision.transforms.v2 as transforms
import json
from torchvision.datasets import ImageFolder
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent


class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Dataset:
        PATH = BASE_DIR / Path('../datasets/CEDAR')
        CLASS_NAMES = {
            'original': 'full_org',
            'forged': 'full_forg'
        }
        NUM_WRITERS = 55

    class Transformation:
        SIZE = (350, 543)
        MEAN = (0.016121178860934875, )
        STD = (0.0796078863779478, )

        class OtsuThreshold(nn.Module):
            def forward(self, img: Image):
                img = np.asarray(img)
                _, img = cv2.threshold(
                    img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
                img = Image.fromarray(img)
                return img

        class Affination:
            DEGREES = 5
            TRANSLATION = (0.05, 0.05)
            SCALE = (0.95, 1.05)

    class DataLoader:
        SPLIT = {
            'num_train_writers': 40,
            'num_test_writers': 10,
            'num_val_writers': 5
        }
        COMMON_KWARGS = {
            'num_workers': 8,
            'pin_memory': torch.cuda.is_available()
        }
        TRAIN_KWARGS = {
            'batch_size': 128,
            'shuffle': True,
            'drop_last': True,
            **COMMON_KWARGS
        }
        NON_TRAIN_KWARGS = {
            'batch_size': TRAIN_KWARGS['batch_size'] * 2,
            **COMMON_KWARGS
        }

    @staticmethod
    def setup_reproducibility(seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f'[Config Info] Reproducibility set with SEED = {seed}.')
        else:
            print('[Config Info] SEED is None. Randomness enabled.')


class Transformations:
    INPUT_TRANSFORMATIONS = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomInvert(p=1),
        Config.Transformation.OtsuThreshold()
    ])
    OUTPUT_TRANSFORMATIONS = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])


def _validate_paths() -> bool:
    root = Config.Dataset.PATH
    org = root / Config.Dataset.CLASS_NAMES['original']
    forg = root / Config.Dataset.CLASS_NAMES['forged']
    if not root.exists():
        raise FileNotFoundError(
            f'[Config Error] Dataset root not found: {root}')
    if not org.exists():
        raise FileNotFoundError(
            f'[Config Error] Class directory missing: {org}')
    if not forg.exists():
        raise FileNotFoundError(
            f'[Config Error] Class directory missing: {forg}')
    return True


def _calculate_imgs_average_size(dataset: ImageFolder, total_images: int) -> tuple[int, int]:
    deterministic_transformation = transforms.Compose([
        Transformations.INPUT_TRANSFORMATIONS,
        Transformations.OUTPUT_TRANSFORMATIONS
    ])
    dataset.transform = deterministic_transformation
    total_h, total_w = 0, 0
    for img, _ in tqdm(dataset, desc='Calculating average image size', leave=False, total=total_images):
        _, h, w = img.size()
        total_h += h
        total_w += w
    final_h = total_h // total_images
    final_w = total_w // total_images
    return final_h, final_w


def _calculate_imgs_mean_and_std(dataset: ImageFolder, total_images: int) -> tuple[float, float]:
    deterministic_transformation = transforms.Compose([
        Transformations.INPUT_TRANSFORMATIONS,
        transforms.Resize(Config.Transformation.SIZE),
        Transformations.OUTPUT_TRANSFORMATIONS
    ])
    dataset.transform = deterministic_transformation
    mean_accumulator = 0.0
    std_accumulator = 0.0
    for img, _ in tqdm(dataset, desc='Calculating average image statistics', leave=False, total=total_images):
        mean = img.mean().item()
        std = img.std().item()
        mean_accumulator += mean
        std_accumulator += std
    final_mean = mean_accumulator / total_images
    final_std = std_accumulator / total_images
    return final_mean, final_std


def _cache_dataset_info() -> None:
    cache_path = BASE_DIR / Path('./dataset_information.json')
    if cache_path.exists():
        print(
            f'[Config Info] Loading dataset information cache from {cache_path}...')
        with open(cache_path, 'r') as f:
            data = json.load(f)
            print(f'[Config Info] Dataset information loaded: {data}')
            Config.Transformation.SIZE = tuple(data['size'])
            Config.Transformation.MEAN = data['mean']
            Config.Transformation.STD = data['std']
        return
    print('[Config Warn] Cache missing. Calculating dataset information (this happens once)...')
    dataset = ImageFolder(root=Config.Dataset.PATH)
    total_images = len(dataset)
    print(
        f'[Config Warn] Calculating the average height and average width for {total_images} images...')
    final_size = _calculate_imgs_average_size(dataset, total_images)
    Config.Transformation.SIZE = final_size
    print(
        f'[Config Warn] Calculating the average mean and average standard deviation for {total_images} images...')
    final_mean, final_std = _calculate_imgs_mean_and_std(dataset, total_images)
    Config.Transformation.MEAN = (final_mean, )
    Config.Transformation.STD = (final_std, )
    dataset_info = {
        'size': Config.Transformation.SIZE,
        'mean': Config.Transformation.MEAN,
        'std': Config.Transformation.STD
    }
    print(f'[Config Warn] Dataset information calculated: {dataset_info}.')
    with open(cache_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
        print(f'[Config Warn] Dataset information saved.')


if _validate_paths():
    _cache_dataset_info()
