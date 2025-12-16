from config import Config
from dataset import CedarLikeDataset, split_cedar_like_dataset
from transformations import TRANSFORMATIONS
from torch.utils.data import DataLoader
from SiameseNetwork import SiameseNetwork
from train import ContrastiveLoss, train
import torch.optim as optim
import torch.amp as amp
from test import test


def main() -> None:
    Config.setup_reproducibility(seed=42)

    full_dataset = CedarLikeDataset(Config.Dataset.PATH, TRANSFORMATIONS)
    train_dataset, val_dataset, test_dataset = split_cedar_like_dataset(
        full_dataset, **Config.Dataset.SPLIT
    )
    train_loader = DataLoader(train_dataset, **Config.DataLoader.TRAIN_KWARGS)
    val_loader = DataLoader(val_dataset, **Config.DataLoader.NON_TRAIN_KWARGS)
    test_loader = DataLoader(test_dataset, **Config.DataLoader.NON_TRAIN_KWARGS)

    model = SiameseNetwork().to(Config.Global.DEVICE)
    loss_function = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=Config.Model.LEARNING_RATE)
    scaler = (
        amp.grad_scaler.GradScaler("cuda") if Config.Global.DEVICE == "cuda" else None
    )
    model = train(model, train_loader, val_loader, loss_function, optimizer, scaler)

    test(model, test_loader)


if __name__ == "__main__":
    main()
