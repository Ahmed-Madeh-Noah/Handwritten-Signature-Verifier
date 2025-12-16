import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from config import Config
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def test(model: nn.Module, test_loader: DataLoader) -> None:
    y = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for (anchor, sample), same in tqdm(
            test_loader, desc="Testing Batches", colour="green"
        ):
            anchor = anchor.to(Config.Global.DEVICE, non_blocking=True)
            sample = sample.to(Config.Global.DEVICE, non_blocking=True)
            same = same.to(Config.Global.DEVICE)
            anchor_y, sample_y = model(anchor, sample)
            distance = F.pairwise_distance(anchor_y, sample_y)
            pred = (distance < Config.Model.THRESHOLD).cpu().numpy()
            y.extend(same.cpu().numpy())
            y_pred.extend(pred)

    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)
