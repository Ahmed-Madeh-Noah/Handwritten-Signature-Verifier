import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import Config
from src.dataset import CEDARDataset
from src.SiameseNetwork import SiameseNetwork
from src.utils import display_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 1.0  

def test(test_loader):

    model = SiameseNetwork()
    model.load_state_dict(torch.load("src/siamese_model_best.pth", map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    print("\nStarting Testing...")

    with torch.no_grad():
        for (img1, img2), match in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            match = match.to(device)

            output1, output2 = model(img1, img2)
            distance = torch.nn.functional.pairwise_distance(output1, output2)

            pred = (distance < THRESHOLD).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(match.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-score:  {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)

  
    print("\nDisplaying 5 test samples...")

    img1_batch, img2_batch, match_batch = next(iter(test_loader))

    for i in range(min(5, len(match_batch))):
        display_tensor(img1_batch[i], title=f"Anchor - Match: {match_batch[i].item()}")
        display_tensor(img2_batch[i], title=f"Sample - Match: {match_batch[i].item()}")
