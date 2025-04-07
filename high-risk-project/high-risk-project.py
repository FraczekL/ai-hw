import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import kagglehub

# Set my device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set path to the full dataset and validate
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Path to dataset files:", path)

# Set baseline parameters
img_size = 630
batch_size = 32
lr = 1e-4
wd = 1e-5
epochs = 25
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Image transformations
normalize_mean = [0.5, 0.5, 0.5]
normalize_std = [0.5, 0.5, 0.5]

train_transform = transform.Compose([
    transform.Resize((img_size, img_size)),
    transform.RandomHorizontalFlip(),
    transform.RandomRotation(15),
    transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transform.ToTensor(),
    transform.Normalize(mean=normalize_mean, std=normalize_std)
])

val_test_transform = transform.Compose([
    transform.Resize((img_size, img_size)),
    transform.ToTensor(),
    transform.Normalize(mean=normalize_mean, std=normalize_std)
])

# Load the entire dataset and verify
dataset = datasets.ImageFolder(path, transform=None)
print(f"Dataset has {len(dataset)} images.")
print(f"Dataset has {dataset.classes} classes.")

# Divide the dataset into training, validation, and test subsets
total_size = len(dataset)
train_size = int(train_split * total_size)
val_size = int(val_split * total_size)
test_size = total_size - train_size - val_size
print(f"train={train_size}, val={val_size}, test={test_size}")

# Index the data subsets
train_idx, val_idx, test_idx = random_split(range(total_size), [train_size, val_size, test_size])

# Transform our data subsets
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
# Apply transforms
train_dataset = TransformedSubset(torch.utils.data.Subset(dataset, train_idx), train_transform)
val_dataset = TransformedSubset(torch.utils.data.Subset(dataset, val_idx), val_test_transform)
test_dataset = TransformedSubset(torch.utils.data.Subset(dataset, test_idx), val_test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)