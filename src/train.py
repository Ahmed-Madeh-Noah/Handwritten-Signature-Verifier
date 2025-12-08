import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from src.SiameseNetwork import SiameseNetwork
from src.dataset import SiamesePairDataset, ContrastiveLoss
from src.data_preprocessing import (
    verify_dataset, 
    create_splits, 
    get_train_transform, 
    get_val_transform, 
    CONFIG
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = CONFIG['batch_size'] 
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
PATIENCE = 7            
IMG_SIZE = (105, 105)   

def main():

    print("Initializing Data Splits...")
    genuine_files, forged_files, writers = verify_dataset(CONFIG['genuine_dir'], CONFIG['forged_dir'])
    
    splits = create_splits(CONFIG['genuine_dir'], CONFIG['forged_dir'], writers)
    
    train_dataset = SiamesePairDataset(
        file_list=splits['train'], 
        img_size=IMG_SIZE, 
        transform=get_train_transform()
    )
    
    val_dataset = SiamesePairDataset(
        file_list=splits['val'], 
        img_size=IMG_SIZE, 
        transform=get_val_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Training Pairs: {len(train_dataset)}")
    print(f"Validation Pairs: {len(val_dataset)}")

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting Training...")
    print("-" * 65)

    for epoch in range(NUM_EPOCHS):
        
        model.train()
        running_train_loss = 0.0
        
        for i, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad(): 
            for i, (img1, img2, label) in enumerate(val_loader):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 
            
            torch.save(model.state_dict(), "best_siamese_model.pth")
            print("  --> Validation Loss Improved. Model Saved.")
            
        else:
            patience_counter += 1
            print(f"  --> No improvement. Early Stopping Counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("\n" + "="*40)
                print("EARLY STOPPING TRIGGERED")
                print("Model stopped to prevent overfitting.")
                print(f"Best Validation Loss: {best_val_loss:.5f}")
                print("="*40)
                break

    print("Training process finished.")

if __name__ == "__main__":
    main()