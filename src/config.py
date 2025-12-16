import random
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from PIL import Image
import cv2


BASE_DIR = Path(__file__).resolve().parent


class Config:
    class Global:
        SEED: int | None = None
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class Dataset:
        PATH = BASE_DIR / Path("../datasets/CEDAR")
        CLASS_NAMES = {"original": "full_org", "forged": "full_forg"}
        SPLIT = {"num_train_writers": 40, "num_val_writers": 5, "num_test_writers": 10}
        NUM_OF_WRITERS = 55

    class Transformation:
        SIZE = (350, 543)
        MEAN = (0.016121178860934875,)
        STD = (0.0796078863779478,)

        class Affination:
            DEGREES = 5
            TRANSLATION = (0.05, 0.05)
            SCALE = (0.95, 1.05)

        class OtsuThreshold(nn.Module):
            def forward(self, img: Image.Image) -> Image.Image:
                img_np = np.asarray(img)
                _, img_np = cv2.threshold(
                    img_np, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU
                )
                img = Image.fromarray(img_np)
                return img

    class DataLoader:
        COMMON_KWARGS = {"num_workers": 4, "pin_memory": torch.cuda.is_available()}
        TRAIN_KWARGS = {
            "batch_size": 8,
            "shuffle": True,
            **COMMON_KWARGS,
        }
        NON_TRAIN_KWARGS = {
            "batch_size": TRAIN_KWARGS["batch_size"] * 2,
            **COMMON_KWARGS,
        }

    class Model:
        LEARNING_RATE = 0.0005
        EPOCHS = 20
        PATH = BASE_DIR / Path("./siamese_network_model.pt")
        THRESHOLD = 1.0

    @staticmethod
    def setup_reproducibility(seed: int | None) -> None:
        Config.Global.SEED = seed
        if Config.Global.SEED is not None:
            random.seed(Config.Global.SEED)
            np.random.seed(Config.Global.SEED)
            torch.manual_seed(Config.Global.SEED)
            torch.cuda.manual_seed_all(Config.Global.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(
                f"[Config INFO] Reproducibility set with SEED = {Config.Global.SEED}."
            )
        else:
            print("[Config INFO] SEED is None. Randomness enabled.")


class Transformations:
    INPUT_TRANSFORMATIONS = transforms.Compose(
        (
            transforms.Grayscale(),
            transforms.RandomInvert(p=1),
            Config.Transformation.OtsuThreshold(),
        )
    )
    OUTPUT_TRANSFORMATIONS = transforms.Compose(
        (
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        )
    )
