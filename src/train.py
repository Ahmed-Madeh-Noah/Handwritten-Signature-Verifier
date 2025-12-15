import torch
import torch.nn.functional as F
from torch.nn import Module
from tqdm import tqdm
from torch.cuda.amp import autocast 

class ContrastiveLoss(Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, epoch_index=0):

    model.train()
    running_loss = 0.0
    
    loop = tqdm(dataloader, desc=f"Train Epoch {epoch_index}", leave=False)

    for batch_idx, ((img1, img2), match) in enumerate(loop):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        match = match.to(device).float()

        optimizer.zero_grad()

        if scaler:
            with autocast():
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, match)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, match)
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val

        loop.set_postfix(loss=loss_val)

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def eval_epoch(model, dataloader, criterion, device, threshold=1.0):
 
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(dataloader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for (img1, img2), match in loop:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            match = match.to(device).float()

            out1, out2 = model(img1, img2)
            
            loss = criterion(out1, out2, match)
            running_loss += loss.item()

            euclidean_distance = F.pairwise_distance(out1, out2)
            predictions = (euclidean_distance < threshold).float()
            
            correct += (predictions == match).sum().item()
            total += match.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy