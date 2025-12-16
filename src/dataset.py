from torchvision.datasets import ImageFolder
from pathlib import Path
import torchvision.transforms.v2 as transforms
from config import Config
import pandas as pd
import torch
from torch.utils.data import Subset
import numpy as np


class CedarLikeDataset(ImageFolder):
    def __init__(
        self, in_root: Path, in_transform: transforms.Compose, *args, **kwargs
    ) -> None:
        super().__init__(root=in_root, transform=in_transform, *args, **kwargs)

        self._org_class_index = self.class_to_idx[
            Config.Dataset.CLASS_NAMES["original"]
        ]
        self._forg_class_index = self.class_to_idx[Config.Dataset.CLASS_NAMES["forged"]]

        self._signatures = self._list_signatures()
        self.pairs = self._init_pairs()

    def _list_signatures(self) -> pd.DataFrame:
        signatures = pd.DataFrame(columns=("path", "class", "writer"))
        for path, class_index in self.samples:
            writer = Path(path).stem.split("_")[1]
            signature = {"path": (path,), "class": (class_index,), "writer": (writer,)}
            signature = pd.DataFrame(signature)
            signatures = pd.concat((signatures, signature), ignore_index=True)
        return signatures

    def _init_pairs(self) -> pd.DataFrame:
        anchor_signatures = self._signatures[
            self._signatures["class"] == self._org_class_index
        ]
        df_pairs = pd.merge(
            anchor_signatures,
            self._signatures,
            on="writer",
            suffixes=("_anchor", "_sample"),
        )
        return pd.DataFrame(
            {
                "anchor": df_pairs["path_anchor"],
                "sample": df_pairs["path_sample"],
                "same": df_pairs["class_anchor"] == df_pairs["class_sample"],
                "writer": df_pairs["writer"],
            }
        )

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        anchor_path = self.pairs.iloc[index]["anchor"]
        sample_path = self.pairs.iloc[index]["sample"]
        same = self.pairs.iloc[index]["same"]

        anchor_img = self.loader(anchor_path)
        sample_img = self.loader(sample_path)

        if self.transform is None:
            return (anchor_img, sample_img), same
        transformed_anchor_img = self.transform(anchor_img)
        transformed_sample_img = self.transform(sample_img)

        return (transformed_anchor_img, transformed_sample_img), torch.tensor(
            same
        ).float()

    def __len__(self) -> int:
        return len(self.pairs)


def split_cedar_like_dataset(
    dataset: CedarLikeDataset,
    num_train_writers: int,
    num_val_writers: int,
    num_test_writers: int,
) -> tuple[Subset, Subset, Subset]:
    assert (
        num_train_writers + num_val_writers + num_test_writers
        <= Config.Dataset.NUM_OF_WRITERS
    ), (
        f"Total number of writers must be less than or equal {Config.Dataset.NUM_OF_WRITERS}."
    )

    assert all(w > 0 for w in (num_train_writers, num_val_writers, num_test_writers)), (
        "All number of writers have a minimum of 1."
    )

    unique_writers = dataset.pairs["writer"].unique()
    shuffled_writers = np.random.permutation(unique_writers)

    train_end = num_train_writers
    val_end = num_train_writers + num_val_writers
    test_end = val_end + num_test_writers

    train_writers = shuffled_writers[:train_end]
    val_writers = shuffled_writers[train_end:val_end]
    test_writers = shuffled_writers[val_end:test_end]

    all_writers = dataset.pairs["writer"]
    train_indices = np.where(np.isin(all_writers, train_writers))[0].tolist()
    val_indices = np.where(np.isin(all_writers, val_writers))[0].tolist()
    test_indices = np.where(np.isin(all_writers, test_writers))[0].tolist()

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, val_subset, test_subset
