import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import sys
from glob import glob
import random
from utils import get_device
import matplotlib.pyplot as plt
from pprint import pprint
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
class CombinedModel(nn.Module):
    def __init__(self, metadata_dim, output_dim=1):
        super(CombinedModel, self).__init__()
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


if __name__ == "__main__":
    # Set memory management parameters
    torch.backends.cudnn.benchmark = True
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load metadata
    metadata_df: pd.DataFrame = pd.read_csv('data/metadata.csv')
    data_type = sys.argv[1]

    # Specify directories for training and validation
    watermelon_dirs: list[str] = glob('data/session_w*')
    training_dirs: list[str] = watermelon_dirs[:(len(watermelon_dirs) - 2)]
    training_dirs.sort()
    validation_dirs: list[str] = watermelon_dirs[(len(watermelon_dirs) - 2):]
    validation_dirs.sort()

    # Create training and validation datasets and dataloaders
    train_dataset = SpectrogramDataset(metadata_df=metadata_df, data_dir='data', data_type=data_type, include_dirs=training_dirs, transform=transform)
    val_dataset = SpectrogramDataset(metadata_df=metadata_df, data_dir='data', data_type=data_type, include_dirs=validation_dirs, transform=transform)

    # Reduce batch size and disable multiprocessing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Reduced batch size
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,  # Reduced batch size
        shuffle=False,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False
    )

    # Initialize model
    device = get_device()
    metadata_dim = 1
    model = CombinedModel(metadata_dim).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=20, mode='triangular')

    # Training loop with early stopping and memory management
    epochs = int(sys.argv[2])
    patience = 50
    min_delta = 0.0001
    train_losses = []
    val_losses = []
    best_model_loss = float('inf')
    best_model = None
    patience_counter = 0

    try:
        for epoch in tqdm(range(epochs)):
            # Clear memory before each epoch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.train()
            running_loss = 0.0
            for images, metadata, targets in train_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images, metadata)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)

                # Clear memory after each batch
                del images, metadata, targets, outputs, loss
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, metadata, targets in val_loader:
                    images = images.to(device)
                    metadata = metadata.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(images, metadata)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * images.size(0)

                    # Clear memory after each batch
                    del images, metadata, targets, outputs, loss
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            if val_loss < best_model_loss - min_delta:
                best_model_loss = val_loss
                best_model = model.state_dict().copy()  # Create a copy of the state dict
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            scheduler.step()

    except Exception as e:
        print(f"Error during training: {str(e)}")
        if best_model is not None:
            print("Loading last best model...")
            model.load_state_dict(best_model)

    # Model evaluation

    if best_model is not None:
        model = model.to('cpu')
        model.load_state_dict(best_model)
        model = model.to(device)

    # Save the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'models/cnn/{data_type}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

    # Plot and save the loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_plot.png')
    plt.close()

    # Evaluate the model on the validation set
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images, metadata, targets = images.to(device), metadata.to(device), targets.to(device)
            outputs = model(images, metadata)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate evaluation metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # Save evaluation results
    evaluation_results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Print the best model loss
    print(f'Best model loss:')
    pprint(evaluation_results)
    print('Training complete')