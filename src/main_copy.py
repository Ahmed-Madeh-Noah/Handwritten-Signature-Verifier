import sys
sys.stdout.reconfigure(line_buffering=True)

print(">>> CODE START: Loading modules...")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from pathlib import Path
import shutil
import gc

try:
    from config import Config
    from dataset import CEDARDataset 
    from SiameseNetwork import SiameseNetworkLite 
    
    from train import train_epoch, eval_epoch, ContrastiveLoss
    print(">>> Modules loaded successfully.")
except ImportError as e:
    print(f"\n!!! IMPORT ERROR: {e}")
    print("Double check your filenames matches the import exactly (case-sensitive).")
    sys.exit(1)

def main():
    print(">>> ENTERING MAIN FUNCTION <<<")

    # 1. Setup Reproducibility
    if hasattr(Config, 'setup_reproducibility'):
        Config.setup_reproducibility(42)
    
    # 2. Cleanup & Device
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device detected: {device}")

    # 3. Hyperparameters
    BATCH_SIZE = 7  
    EPOCHS = 20
    LEARNING_RATE = 0.0005
    VAL_SPLIT_RATIO = 0.2
    
    # 4. Dataset Path
    real_data_path = Config.Dataset.PATH
    print(f"Loading Dataset from: {real_data_path}")
    
    mac_junk = real_data_path / "__MACOSX"
    if mac_junk.exists():
        try: shutil.rmtree(mac_junk)
        except: pass

    if not real_data_path.exists():
        print(f"ERROR: Path not found: {real_data_path}")
        return

    # 5. Initialize Dataset
    print("Initializing CEDARDataset...")
    full_dataset = CEDARDataset(root=real_data_path)
    print(f"Dataset Loaded. Found {len(full_dataset)} pairs.")

    # 6. Split Data
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT_RATIO)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {train_size} Train | {val_size} Val")

    # 7. DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=0)

    # 8. Model & Optimizer
    print("Initializing Model...")
    model = SiameseNetworkLite().to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() if device.type == 'cuda' else None

    # 9. Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        t_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            scaler=scaler, epoch_index=epoch+1
        )
        
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device, threshold=1.0)
        
        print(f"Results - Train Loss: {t_loss:.4f} | Val Acc: {v_acc:.2f}%")

    # 10. Save
    torch.save(model.state_dict(), "siamese_model.pth")
    print("\n>>> SUCCESS: Training complete. Model saved.")

if __name__ == "__main__":
    main()