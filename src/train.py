import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.amp as amp
from config import Config
from tqdm import tqdm


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._margin = margin

    def forward(self, anchor, sample, label):
        euclidean_distance = F.pairwise_distance(anchor, sample)

        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self._margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.grad_scaler.GradScaler | None,
) -> nn.Module:
    with tqdm(range(Config.Model.EPOCHS), desc="Epochs") as epochs_bar:
        for _ in epochs_bar:
            model.train()
            with tqdm(
                train_loader, desc="Batches", colour="cyan", leave=False
            ) as batches_bar:
                for (anchor, sample), same in batches_bar:
                    anchor = anchor.to(Config.Global.DEVICE, non_blocking=True)
                    sample = sample.to(Config.Global.DEVICE, non_blocking=True)
                    same = same.to(Config.Global.DEVICE)
                    optimizer.zero_grad()
                    if scaler:
                        with amp.autocast_mode.autocast(device_type="cuda"):
                            anchor_y, sample_y = model(anchor, sample)
                            loss = loss_function(anchor_y, sample_y, same)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        anchor_y, sample_y = model(anchor, sample)
                        loss = loss_function(anchor_y, sample_y, same)
                        loss.backward()
                        optimizer.step()
                    batches_bar.set_postfix(loss=round(loss.item(), 2))
            epoch_loss = validate(model, val_loader, loss_function)
            epochs_bar.set_postfix(loss=epoch_loss)
    return model


def validate(
    model: nn.Module, val_loader: DataLoader, loss_function: nn.Module
) -> float:
    running_loss = 0.0
    model.eval()
    with tqdm(
        val_loader, desc="Validation Batches", colour="magenta", leave=False
    ) as validation_batches_bar:
        with torch.no_grad():
            for (anchor, sample), same in validation_batches_bar:
                anchor = anchor.to(Config.Global.DEVICE, non_blocking=True)
                sample = sample.to(Config.Global.DEVICE, non_blocking=True)
                same = same.to(Config.Global.DEVICE)
                anchor_y, sample_y = model(anchor, sample)
                loss = loss_function(anchor_y, sample_y, same)
                running_loss += loss.item()
        avg_loss = round(running_loss / len(val_loader), 2)
        torch.save(model.state_dict(), Config.Model.PATH)
        return avg_loss
