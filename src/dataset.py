import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import os
from collections import defaultdict

from src.data_preprocessing import preprocess_signature

def get_empty_writer_entry():
    return {'genuine': [], 'forged': []}

class SiamesePairDataset(Dataset):
    def __init__(self, file_list, img_size, transform=None):
        self.file_list = file_list
        self.img_size = img_size
        self.transform = transform
        
        self.data_by_writer = defaultdict(get_empty_writer_entry)
        
        for file_path, label in file_list:
            try:
                filename = os.path.basename(file_path)
                parts = filename.replace('.png', '').split('_')
                writer_id = int(parts[1])
                
                if label == 1: 
                    self.data_by_writer[writer_id]['genuine'].append(file_path)
                else: 
                    self.data_by_writer[writer_id]['forged'].append(file_path)
            except:
                continue

        self.writers = list(self.data_by_writer.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        writer_id = random.choice(self.writers)
        
        while len(self.data_by_writer[writer_id]['genuine']) < 1:
             writer_id = random.choice(self.writers)
             
        anchor_path = random.choice(self.data_by_writer[writer_id]['genuine'])
        
        should_be_same = random.randint(0, 1) 
        
        if should_be_same == 0:
            img2_path = random.choice(self.data_by_writer[writer_id]['genuine'])
            target = torch.tensor([0], dtype=torch.float32)
        else:
            if self.data_by_writer[writer_id]['forged']:
                img2_path = random.choice(self.data_by_writer[writer_id]['forged'])
            else:
                diff_writer = random.choice(self.writers)
                while diff_writer == writer_id:
                    diff_writer = random.choice(self.writers)
                img2_path = random.choice(self.data_by_writer[diff_writer]['genuine'])
            
            target = torch.tensor([1], dtype=torch.float32)

        try:
            img1 = preprocess_signature(anchor_path, self.img_size)
            img2 = preprocess_signature(img2_path, self.img_size)
        except ValueError:
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, target

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive