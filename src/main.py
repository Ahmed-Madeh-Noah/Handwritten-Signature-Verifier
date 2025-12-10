from config import Config
from dataset import split_cedar_dataset
from torch.utils.data import DataLoader


def main():
    Config.setup_reproducibility(seed=42)

    train, test, val = split_cedar_dataset(**Config.DataLoader.SPLIT)
    train_loader = DataLoader(train, **Config.DataLoader.TRAIN_KWARGS)
    val_loader = DataLoader(val, **Config.DataLoader.NON_TRAIN_KWARGS)
    test_loader = DataLoader(test, **Config.DataLoader.NON_TRAIN_KWARGS)


if __name__ == '__main__':
    main()
