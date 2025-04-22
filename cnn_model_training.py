import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from pprint import pprint
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from utils import get_device, NumpyEncoder, SpectrogramDataset, SpectrogramModel


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
    n_splits = int(sys.argv[2])  # Number of folds
    epochs = int(sys.argv[3])    # Epochs per fold

    # Get all watermelon directories
    watermelon_dirs: list[str] = glob('data/session_w*')
    watermelon_dirs.sort()

    # Create dataset with all data
    full_dataset = SpectrogramDataset(metadata_df=metadata_df, data_dir='data', data_type=data_type, include_dirs=watermelon_dirs, transform=transform)

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store metrics for each fold
    fold_metrics = []

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'models/cnn/{data_type}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # K-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f'\nStarting fold {fold + 1}/{n_splits}')
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=2048,
            sampler=train_subsampler,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            full_dataset,
            batch_size=1024,
            sampler=val_subsampler,
            num_workers=0,
            pin_memory=False
        )

        # Initialize model for this fold
        device = get_device()
        metadata_dim = 1
        output_dim = 1
        model = SpectrogramModel(metadata_dim, output_dim).to(device)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=50, mode='triangular')

        # Training loop with early stopping and memory management
        patience = 50
        min_delta = 0.0001
        train_losses = []
        val_losses = []
        fold_best_loss = float('inf')
        fold_best_model = None
        patience_counter = 0
        best_epoch = 0

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
                
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                # Update best model for this fold
                if val_loss < fold_best_loss - min_delta:
                    fold_best_loss = val_loss
                    fold_best_model = model
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

                scheduler.step()

        except Exception as e:
            print(f"Error during training fold {fold + 1}: {str(e)}")
            continue

        # Create fold output directory
        fold_output_dir = f'{output_dir}/fold_{fold + 1}'
        os.makedirs(fold_output_dir, exist_ok=True)

        # Save the best model for this fold
        if fold_best_model is not None:
            torch.save(fold_best_model.state_dict(), f'{fold_output_dir}/best_fold_model.pth')

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

        # Calculate evaluation metrics for this fold
        mse = float(mean_squared_error(all_targets, all_predictions))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(all_targets, all_predictions))
        r2 = float(r2_score(all_targets, all_predictions))

        fold_metrics.append({
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_epoch': best_epoch + 1,
            'best_val_loss': float(fold_best_loss)
        })

        # Save evaluation results for this fold
        with open(f'{fold_output_dir}/evaluation_results.json', 'w') as f:
            json.dump(fold_metrics[-1], f, indent=4, cls=NumpyEncoder)

        # Save loss plot for this fold
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Fold {fold + 1}')
        plt.legend()
        plt.savefig(f'{fold_output_dir}/loss_plot.png')
        plt.close()


    # Calculate statistics for each metric
    metrics_stats = {}
    for metric in ['mse', 'rmse', 'mae', 'r2', 'best_epoch', 'best_val_loss']:
        values = [m[metric] for m in fold_metrics]
        metrics_stats[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    # Save all metrics and statistics
    metrics_data = {
        'fold_metrics': fold_metrics,
        'metrics_statistics': metrics_stats
    }

    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(metrics_data, f, indent=4, cls=NumpyEncoder)

    # Print the results
    print('\nCross-validation results:')
    print('Fold metrics:')
    pprint(fold_metrics)
    print('\nAverage metrics:')
    pprint(metrics_stats)
    print('Training complete')