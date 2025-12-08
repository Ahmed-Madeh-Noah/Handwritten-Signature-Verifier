from config import CONFIG
from torch.nn import Module
from PIL import Image
from numpy import asarray
from cv2 import threshold, THRESH_TOZERO, THRESH_OTSU
import torchvision.transforms.v2 as transforms
from torch import float32
from torchvision.datasets import ImageFolder
from pathlib import Path
from pandas import DataFrame, merge
from torch import Tensor
config = CONFIG()


class OtsuThreshold(Module):
    def forward(self, img: Image):
        img = asarray(img)
        img = threshold(img, 0, 255, THRESH_TOZERO + THRESH_OTSU)[1]
        img = Image.fromarray(img)
        return img


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(config.size),
    transforms.RandomInvert(p=1),
    OtsuThreshold(),
    transforms.RandomAffine(
        degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
    ),
    transforms.ToImage(),
    transforms.ToDtype(float32, scale=True),
    transforms.Normalize((config.mean, ), (config.std, ))
])


class CEDARDataset(ImageFolder):
    def __init__(self, root: Path = config.dataset, transform: transforms.Compose = transform, *args, **kwargs) -> None:
        super().__init__(root, transform, *args, **kwargs)
        self._org_class_index = self.class_to_idx[config.original_signatures_class]
        self._forg_class_index = self.class_to_idx[config.forged_signatures_class]
        self._signature_samples = self._list_signature_samples()
        self._pairs = self._init_pairs()

    def _list_signature_samples(self) -> DataFrame:
        signatures = []
        for path, class_idx in self.samples:
            writer = Path(path).stem.split('_')[1]
            signatures.append({
                'path': path,
                'class': class_idx,
                'writer': writer
            })
        return DataFrame(signatures)

    def _init_pairs(self) -> DataFrame:
        anchor_signatures = self._signature_samples[self._signature_samples['class']
                                                    == self._org_class_index]
        df_pairs = merge(anchor_signatures, self._signature_samples,
                         on='writer', suffixes=('_anchor', '_sample'))
        return DataFrame({
            'anchor': df_pairs['path_anchor'],
            'sample': df_pairs['path_sample'],
            'match': df_pairs['class_anchor'] == df_pairs['class_sample']
        })

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor], bool]:
        anchor_path = self._pairs.iloc[index]['anchor']
        sample_path = self._pairs.iloc[index]['sample']
        match = self._pairs.iloc[index]['match']
        anchor_img = self.loader(anchor_path)
        sample_img = self.loader(sample_path)
        transformed_anchor_img = self.transform(anchor_img)
        transformed_sample_img = self.transform(sample_img)
        return (transformed_anchor_img, transformed_sample_img), match

    def __len__(self) -> int:
        return len(self._pairs)
