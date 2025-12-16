import torch.nn as nn
import torch


class SiameseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn1 = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=11, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3, inplace=True),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3, inplace=True),
            # 128 * 4 * 4 = 2,048 features
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn1(x)
        y = y.view(y.size()[0], -1)
        y = self.fc1(y)
        return y

    def forward(
        self, anchor: torch.Tensor, sample: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_y = self.forward_once(anchor)
        sample_y = self.forward_once(sample)
        return anchor_y, sample_y
