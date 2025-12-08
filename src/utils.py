from config import CONFIG
from pathlib import Path
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from data_preprocessing import transform
from torch.utils.data import DataLoader
import torch
config = CONFIG()


def get_average_image_size(root: Path = config.dataset) -> tuple[int, int]:
    dataset = ImageFolder(root)
    height = 0
    width = 0
    num_images = 0
    for img, _ in dataset:
        w, h = img.size
        width += w
        height += h
        num_images += 1
    average_width = width // num_images
    average_height = height // num_images
    return average_height, average_width


def display_tensor(tensor: torch.Tensor, title: str = None) -> None:
    plt.figure()
    plt.imshow(tensor.squeeze(), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def get_dataset_mean_std(root: Path = config.dataset) -> tuple[float, float]:
    dataset = ImageFolder(root, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    total_sum = 0.
    total_sq_sum = 0.
    total_pixels = 0
    for img, _ in data_loader:
        total_sum += torch.sum(img)
        total_sq_sum += torch.sum(img ** 2)
        total_pixels += img.numel()
    mean = total_sum / total_pixels
    std = (total_sq_sum / total_pixels - mean ** 2) ** 0.5
    return mean.item(), std.item()
