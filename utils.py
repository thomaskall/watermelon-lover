import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
import json
import pandas as pd
import numpy as np
from glob import glob as glob # glob
import os


def get_device():
    """Get the appropriate device for training"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Define a custom dataset
class SpectrogramDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, data_dir: str = "data", data_type: str = "sweep", include_dirs=None, transform=None):
        self.image_paths = glob(os.path.join(data_dir, '**', 'wav', 'spectrograms', f'{data_type}*.png'), recursive=True)
        self.metadata_df = metadata_df
        self.transform = transform

        # Filter image paths based on specified directories
        if include_dirs is not None:
            self.image_paths = [p for p in self.image_paths if any(d in p for d in include_dirs)]

        # Extract watermelon IDs from image paths and match with metadata
        self.image_metadata = []
        for img_path in self.image_paths:
            # Extract the watermelon ID from the directory name
            watermelon_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path)))).split('_')[-1]
            # Find the corresponding metadata row
            metadata_row = self.metadata_df[self.metadata_df['watermelon_id'] == watermelon_id]
            if not metadata_row.empty:
                row = metadata_row.iloc[0]
                self.image_metadata.append((img_path, row['weight'], row['brix_score']))

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, idx):
        img_path, weight, brix_score = self.image_metadata[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        metadata = np.array([weight], dtype=np.float32)
        target = np.array([brix_score], dtype=np.float32)

        return image, metadata, target

# Define a custom model
#### Previously CombinedModel
class SpectrogramModel(nn.Module):
    def __init__(self, metadata_dim=1, output_dim=1):
        super(SpectrogramModel, self).__init__()
        self.cnn = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Freeze convolutional layers
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Remove the final classification layer
        cnn_output_dim = self.cnn.classifier[0].in_features
        self.cnn.classifier = nn.Identity()

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + metadata_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, image, metadata):
        cnn_features = self.cnn(image)
        combined_features = torch.cat((cnn_features, metadata), dim=1)
        output = self.fc(combined_features)
        return output