import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.utils.tensorboard as tb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import kagglehub

class ResNetCNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-2])
        self.projection = nn.Conv2d(512, d_model, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.projection(x)
        return x
        

# Set my device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set path to the full dataset and validate
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Path to dataset files:", path)

# Set baseline parameters
img_size = 224
batch_size = 32
lr = 1e-4
wd = 1e-5
epochs = 25
train_split = 0.7
val_split = 0.15
# test_split = 0.15

# Image transformations
normalize_mean = [0.5, 0.5, 0.5]
normalize_std = [0.5, 0.5, 0.5]

# Set image transforms in order to get variety
# and send to tensor
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
# dataset = datasets.ImageFolder(path, transform=None)
dataset = datasets.ImageFolder(os.path.join(path, "brain_tumor_dataset"), transform=None)
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
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

# Define CNN
cnn_output_channels = 512

class OurBrainReadingCNN(nn.Module):
    def __init__(self, cnn_output_channels=cnn_output_channels):
        super().__init__()
        self.layers = nn.Sequential(
        # Downconvolution block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Downconvolution block 2
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Downconvolution block 3
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Downconvolution block 4
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Downconvolution block 5
        nn.Conv2d(512, cnn_output_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(cnn_output_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )        

        # Calculate spatial dimensions, five layers * stride of 2 = 2^5
        self.output_spatial_dim = img_size // 32

    def forward(self, x):
        return self.layers(x)

# Define Tranformer Encoder Head
d_model = 512
nhead = 8
num_encoder_layers = 3
dim_feedforward = 2048
dropout = 0.1

class OurBrainReadingTransformerEncoderHead(nn.Module):
    def __init__(self, sequence_length, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Create position encoding and CLS token (hence the +1)
        self.position_encoding = nn.Parameter(torch.zeros(1, self.sequence_length + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Create encoder layer and then using it to create encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Normalize
        self.normalize = nn.LayerNorm(d_model)
        
        # Classify
        self.classify = nn.Linear(d_model, num_classes)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_encoding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.classify.weight)
        nn.init.constant_(self.classify.bias, 0)

    def forward(self, x):
        # Get batch size
        B = x.shape[0]
        
        # Create CLS tokens for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Prepend CLS tokens
        enc_input = torch.cat((cls_tokens, x), dim=1)
        enc_input = enc_input + self.position_encoding
        
        # Pass through encoder
        enc_output = self.encoder(enc_input)

        # Get tokens, normalize, classify
        enc_output = enc_output[:, 0]
        normalized_output = self.normalize(enc_output)
        logits = self.classify(normalized_output)
        
        return logits

# Define Hybrid

class OurBrainReadingCNNTransformerHybrid(nn.Module):
    def __init__(self, cnn_output_channels=cnn_output_channels, d_model=d_model,
                 nhead=nhead, num_encoder_layers=num_encoder_layers,
                 dim_feedforward=dim_feedforward, num_classes=1, dropout=dropout):
        super().__init__()
        
        # Extract important features using our custom CNN
        self.cnn = OurBrainReadingCNN(cnn_output_channels=cnn_output_channels)

        # Calculate the sequence length using our CNNs output spatial dims
        cnn_out_dim = self.cnn.output_spatial_dim
        self.sequence_length = cnn_out_dim * cnn_out_dim

        # Connect our CNN to transformer
        self.projection_input = nn.Conv2d(cnn_output_channels, d_model, kernel_size=1)
        
        # Transformer
        self.our_transformer_head = OurBrainReadingTransformerEncoderHead(
            sequence_length=self.sequence_length,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes
        )

        # Init this layer weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection_input.weight)
    
    def forward(self, x):
        # Extract features
        cnn_features = self.cnn(x)
        
        # Project to Transformer
        transformer_input = self.projection_input(cnn_features)
        
        # Flatten from spatial to sequence
        transformer_input = transformer_input.flatten(2)
        transformer_input = transformer_input.transpose(1, 2)

        # Feed through our Tranformer head to classify
        logits = self.our_transformer_head(transformer_input)
        
        return logits

# Define Hybrid with ResNet

class OurBrainReadingCNNTransformerHybridWithResNet(nn.Module):
    def __init__(self, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                 dim_feedforward=dim_feedforward, num_classes=1, dropout=dropout):
        super().__init__()
        
        # Extract important features using ResNet, then project
        self.cnn = ResNetCNN(d_model=d_model)
        
        # Set spatial dims per ResNet documentation
        self.sequence_length = 7 * 7
    
        # Transformer
        self.our_transformer_head = OurBrainReadingTransformerEncoderHead(
            sequence_length=self.sequence_length,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection_input.weight)
    
    def forward(self, x):
        # Extract features
        cnn_features = self.cnn(x)
   
        # Flatten from spatial to sequence
        transformer_input = cnn_features.flatten(2)
        transformer_input = transformer_input.transpose(1, 2)

        # Feed through our Tranformer head to classify
        logits = self.our_transformer_head(transformer_input)
        
        return logits

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Set the model and drop it on GPU
model = OurBrainReadingCNNTransformerHybridWithResNet().to(device)

# Set loss function, optimizer, logger
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

# Protect against multiprocessing issues
if __name__ == "__main__":

    # Set logging parameters
    exp_dir = "logs"
    model_name = "OurBrainReadingCNNTransformerHybrid"

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Set other variables
    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # Training and eval loop
    for epoch in range(epochs):

        # Clear metrics dictionary
        for key in metrics:
            metrics[key].clear()

        # Enable training mode
        model.train()
        
        # Load batches of file using DataLoader
        for img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            img, label = img.to(device), label.to(device)
            
            # Retain original label for accuracy calculation
            orig_label = label.clone()
            

            # BCEWithLogitsLoss requires a float
            label = label.float().unsqueeze(1)

            prediction = model(img)
            loss_value = loss_fn(prediction, label)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # Log per batch
            metrics["train_acc"].append(torch.sigmoid(prediction).round().squeeze(1).eq(orig_label).float().mean().item())
            
            global_step += 1

        # Disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                img, label = img.to(device), label.to(device)

                orig_label = label.clone()
                
                label = label.float().unsqueeze(1)

                prediction = model(img)
                
                metrics["val_acc"].append(torch.sigmoid(prediction).round().squeeze(1).eq(orig_label).float().mean().item())

        # Log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean().item()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean().item()
        
        logger.add_scalar("train_acc", epoch_train_acc, global_step=global_step)
        logger.add_scalar("val_acc", epoch_val_acc, global_step=global_step)

        # Print on first, last, every 10th epoch
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {epochs:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Final testing of the model
    with torch.inference_mode():
        model.eval()
        test_acc = []
        
        for img, label in tqdm(test_loader, desc="Testing"):
            img, label = img.to(device), label.to(device)

            orig_label = label.clone()
            
            label = label.float().unsqueeze(1)

            prediction = model(img)
            
            test_acc.append(torch.sigmoid(prediction).round().squeeze(1).eq(orig_label).float().mean().item())

        mean_test_acc = torch.as_tensor(test_acc).mean().item()
        print(f"Test accuracy: {mean_test_acc:.4f}")
        

    # Save the model
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Final model saved to {model_name}.pth")