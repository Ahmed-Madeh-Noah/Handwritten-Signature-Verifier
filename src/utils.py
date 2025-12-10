import torch
import matplotlib.pyplot as plt

def display_tensor(tensor: torch.Tensor, title: str = None) -> None:
    plt.figure()
    plt.imshow(tensor.squeeze(), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()
