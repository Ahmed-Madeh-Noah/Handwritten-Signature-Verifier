from torchvision.datasets import ImageFolder
from pathlib import Path
from config import Config
import torchvision.transforms.v2 as transforms
from transformations import TRANSFORMATIONS
import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import Subset


class CEDARDataset(ImageFolder):
    def __init__(self, root: Path = Config.Dataset.PATH, transform: transforms.Compose = TRANSFORMATIONS, *args, **kwargs) -> None:
        super().__init__(root, transform, *args, **kwargs)

        self._org_class_index = self.class_to_idx[Config.Dataset.CLASS_NAMES['original']]
        self._forg_class_index = self.class_to_idx[Config.Dataset.CLASS_NAMES['forged']]

        self._signature_samples = self._list_signature_samples()
        self.pairs = self._init_pairs()

    def _list_signature_samples(self) -> pd.DataFrame:
        signatures = []
        for path, class_idx in self.samples:
            writer = Path(path).stem.split('_')[1]
            signatures.append({
                'path': path,
                'class': class_idx,
                'writer': writer
            })
        return pd.DataFrame(signatures)

    def _init_pairs(self) -> pd.DataFrame:
        anchor_signatures = self._signature_samples[
            self._signature_samples['class'] == self._org_class_index
        ]
        df_pairs = pd.merge(anchor_signatures, self._signature_samples,
                            on='writer', suffixes=('_anchor', '_sample'))
        return pd.DataFrame({
            'anchor': df_pairs['path_anchor'],
            'sample': df_pairs['path_sample'],
            'match': df_pairs['class_anchor'] == df_pairs['class_sample'],
            'writer': df_pairs['writer']
        })

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], bool]:
        anchor_path = self.pairs.iloc[index]['anchor']
        sample_path = self.pairs.iloc[index]['sample']
        match = self.pairs.iloc[index]['match']
        anchor_img = self.loader(anchor_path)
        sample_img = self.loader(sample_path)
        transformed_anchor_img = self.transform(anchor_img)
        transformed_sample_img = self.transform(sample_img)
        return (transformed_anchor_img, transformed_sample_img), match

    def __len__(self) -> int:
        return len(self.pairs)


def split_cedar_dataset(num_train_writers: int, num_test_writers: int, num_val_writers: int, dataset: CEDARDataset = CEDARDataset()) -> tuple[Subset, Subset, Subset]:
    assert num_train_writers + num_val_writers + num_test_writers <= Config.Dataset.NUM_WRITERS, \
        f'Total number of writers must be less than or equal {Config.Dataset.NUM_WRITERS}.'
    assert all(w > 0 for w in (num_train_writers,
               num_test_writers, num_val_writers)), 'All number of writers have a minimum of 1.'
    unique_writers = dataset.pairs['writer'].unique()
    random.shuffle(unique_writers)
    train_end = num_train_writers
    test_end = num_train_writers + num_test_writers
    val_end = num_train_writers + num_test_writers + num_val_writers
    train_writers = unique_writers[:train_end]
    test_writers = unique_writers[train_end:test_end]
    val_writers = unique_writers[test_end:val_end]
    train_mask = dataset.pairs['writer'].isin(train_writers)
    test_mask = dataset.pairs['writer'].isin(test_writers)
    val_mask = dataset.pairs['writer'].isin(val_writers)
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    val_indices = np.where(val_mask)[0]
    return Subset(dataset, train_indices), Subset(dataset, test_indices), Subset(dataset, val_indices)
