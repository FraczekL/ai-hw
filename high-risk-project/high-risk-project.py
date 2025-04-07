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

# Set path to the dataset and validate
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

