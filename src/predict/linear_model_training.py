import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
from plotly.io import show
import sys
from typing import Dict, List
from tqdm import tqdm
import joblib
import os
import json
from datetime import datetime
from utils import get_device

torch.set_num_threads(1)

class WatermelonDataset(Dataset):
    """PyTorch Dataset for watermelon data"""
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Keep data on CPU and only move to device when getting batch
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BrixPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rates: List[float]):
        super(BrixPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, (hdim, drop_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.BatchNorm1d(hdim),
                nn.Dropout(drop_rate)
            ])
            prev_dim = hdim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    

class ModelTrainer:
    def __init__(self, data: pd.DataFrame, data_type: str, random_state: int = 42, save_dir: str = "models"):
        self.random_state = random_state
        self.data_type = data_type
        self.scaler = StandardScaler()
        self.models: Dict[str, nn.Module|RandomForestRegressor|XGBRegressor] = {}
        self.histories = {}
        self.best_params = {}
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = "cpu" #get_device()
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.prepare_data(data)
        
    def prepare_data(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for modeling"""

        print("Preparing data for modeling...")

        # Create splits based on watermelon_id
        train_ids: list[str] = []
        for i in range(1, 14):
            train_ids.append(f'w{i}')
        val_ids: list[str] = []
        for i in range(14, 16):
            val_ids.append(f'w{i}')
        test_ids: list[str] = ["w16"]
        
        # Split data
        train_data = data[data['watermelon_id'].isin(train_ids)]
        val_data = data[data['watermelon_id'].isin(val_ids)]
        test_data = data[data['watermelon_id'].isin(test_ids)]
        
        # Separate features and target
        feature_cols = [col for col in data.columns 
                    if col not in ['density', 'watermelon_id', 'brix_score', 'file_name']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['brix_score']

        # Convert data to numpy right away
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Convert to float32 and ensure correct shapes
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        X_val = val_data[feature_cols]
        y_val = val_data['brix_score']

        # Convert data to numpy right away
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Convert to float32 and ensure correct shapes
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        X_test = test_data[feature_cols]
        y_test = test_data['brix_score']

        # Convert data to numpy right away
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Convert to float32 and ensure correct shapes
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        print(f"Train set: {len(train_ids)} watermelons, {len(X_train)} samples")
        print(f"Val set: {len(val_ids)} watermelons, {len(X_val)} samples")
        print(f"Test set: {len(test_ids)} watermelons, {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def optimize_rf(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize Random Forest hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
        
        model = RandomForestRegressor(**params, random_state=self.random_state)
        scores = cross_val_score(model, X, y.ravel(), cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def optimize_xgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize XGBoost hyperparameters"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        model = XGBRegressor(**params, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    def optimize_nn(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> float:
        """Optimize Neural Network hyperparameters"""
        n_layers = trial.suggest_int('n_layers', 2, 10)
        hidden_dims = []
        dropout_rates = []

        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 32, 256))
            dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.0, 0.5))
        
        model = BrixPredictor(X.shape[1], hidden_dims, dropout_rates).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial.suggest_float('learning_rate', 1e-8, 1e-2, log=True)
        )
        
        # Quick CV evaluation
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            batch_size = min(1024, len(X_train_fold))
            train_dataset = WatermelonDataset(X_train_fold, y_train_fold)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            # Training loop
            model = model.train()
            for _ in tqdm(range(epochs), desc=f"NN Optimization Fold {fold+1}/5"):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    output = model(batch_X)
                    if output.shape != batch_y.shape:
                        output = output.view(batch_y.shape)
                    loss = nn.MSELoss()(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model = model.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(X_val_fold).to(self.device)
                val_output = model(val_tensor)
                # Move to CPU for numpy conversion
                val_output = val_output.cpu()
                val_score = mean_squared_error(y_val_fold, val_output.numpy())
                scores.append(val_score)

        del train_loader, train_dataset
        torch.mps.empty_cache()

        return np.mean(scores)
    
    def save_models(self):
        """Save all trained models and their parameters"""
        model_dir = os.path.join(self.save_dir, self.timestamp)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        # Save tree-based models
        joblib.dump(self.models['rf'], os.path.join(model_dir, 'random_forest.joblib'))
        print("Random Forest model saved")
        joblib.dump(self.models['xgb'], os.path.join(model_dir, 'xgboost.joblib'))
        print("XGBoost model saved")
        
        # Save neural network weights and metadata separately
        self.models['nn'] = self.models['nn'].to('cpu')
        torch.save(self.models['nn'].state_dict(), os.path.join(model_dir, 'neural_network.pt'))
        print("Neural Network model saved")
        nn_metadata = {
            'model_architecture': {
                'input_dim': self.X_train.shape[1],
                'hidden_dims': [self.best_params['nn'][f'hidden_dim_{i}'] 
                              for i in range(self.best_params['nn']['n_layers'])],
                'dropout_rates': [self.best_params['nn'][f'dropout_{i}'] 
                                for i in range(self.best_params['nn']['n_layers'])]
            },
            'training_history': self.histories['nn']
        }
        with open(os.path.join(model_dir, 'neural_network_metadata.json'), 'w') as f:
            json.dump(nn_metadata, f, indent=4)
        
        print(f"Models saved in {model_dir}")
        return model_dir

    def train_models(self, n_trials: int = 100, epochs: int = 200, early_stopping_patience: int = 20):
        """Train and optimize all models"""

        # Optimize Neural Network
        # study_nn = optuna.create_study(direction='minimize', study_name=f"Neural Network {self.data_type}")
        # study_nn.optimize(lambda trial: self.optimize_nn(trial, self.X_train, self.y_train, epochs=epochs), n_trials=n_trials)
        self.best_params['nn'] = {
            "n_layers": 4,
            "hidden_dim_0": 256,
            "dropout_0": 0.0,
            "hidden_dim_1": 128,
            "dropout_1": 0.0,
            "hidden_dim_2": 64,
            "dropout_2": 0.0,
            "hidden_dim_3": 32,
            "dropout_3": 0.0,
            "learning_rate": 1e-4,
            "weight_decay": 1e-7
        }
        
        print("Training Neural Network...")
        hidden_dims = [self.best_params['nn'][f'hidden_dim_{i}'] 
                      for i in range(self.best_params['nn']['n_layers'])]
        dropout_rates = [self.best_params['nn'][f'dropout_{i}']
                        for i in range(self.best_params['nn']['n_layers'])]
        
        self.models['nn'] = BrixPredictor(self.X_train.shape[1], hidden_dims, dropout_rates).to(self.device)
        optimizer = torch.optim.NAdam(self.models['nn'].parameters(), 
                                   lr=self.best_params['nn']['learning_rate'],
                                   weight_decay=self.best_params['nn']['weight_decay'])

        train_dataset = WatermelonDataset(self.X_train, self.y_train)
        val_dataset = WatermelonDataset(self.X_val, self.y_val)
        
        # Reduce number of workers and disable pin_memory for MPS
        train_loader = DataLoader(
            train_dataset, 
            batch_size=1024, 
            shuffle=True,
            num_workers=4,  # Set to 0 for MPS
            pin_memory=False  # Disable pin_memory for MPS
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1024,
            num_workers=2,
            pin_memory=False
        )
        # Training with validation
        self.histories['nn'] = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training Neural Network"):
            # Training phase
            self.models['nn'].train()
            train_epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.models['nn'](batch_X)
                if output.shape != batch_y.shape:
                    output = output.view(batch_y.shape)
                loss = nn.MSELoss()(output, batch_y)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            train_epoch_loss /= len(train_loader)
            
            # Validation phase
            self.models['nn'].eval()
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                    val_output = self.models['nn'](val_X)
                    if val_output.shape != val_y.shape:
                        val_output = val_output.view(val_y.shape)
                    val_loss = mean_squared_error(val_y.cpu().numpy(), val_output.cpu().numpy())
            
            # Save losses
            self.histories['nn']['train_loss'].append(train_epoch_loss)
            self.histories['nn']['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.models['nn'].state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        del train_loader, val_loader, train_dataset, val_dataset
        torch.mps.empty_cache()

        self.models['nn'].load_state_dict(best_model_state)

        # Optimize Random Forest
        study_rf = optuna.create_study(direction='minimize', study_name=f"Random Forest {self.data_type}")
        study_rf.optimize(lambda trial: self.optimize_rf(trial, self.X_train, self.y_train), n_trials=n_trials)
        self.best_params['rf'] = study_rf.best_params

        # fig = optuna.visualization.plot_optimization_history(study_rf)
        # show(fig)
        # fig = optuna.visualization.plot_param_importances(study_rf)
        # show(fig)
        
        # Optimize XGBoost
        study_xgb = optuna.create_study(direction='minimize', study_name=f"XGBoost {self.data_type}")
        study_xgb.optimize(lambda trial: self.optimize_xgb(trial, self.X_train, self.y_train), n_trials=n_trials)
        self.best_params['xgb'] = study_xgb.best_params

        # fig = optuna.visualization.plot_optimization_history(study_xgb)
        # show(fig)
        # fig = optuna.visualization.plot_param_importances(study_xgb)
        # show(fig)
        
        # Train final models with best parameters
        print("Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(**self.best_params['rf'], random_state=self.random_state)
        self.models['rf'].fit(self.X_train, self.y_train.ravel())
        
        print("Training XGBoost...")
        self.models['xgb'] = XGBRegressor(**self.best_params['xgb'], random_state=self.random_state)
        self.models['xgb'].fit(
            self.X_train, 
            self.y_train, 
            eval_set=[(self.X_val, self.y_val)], 
        )
        
        # After training is complete, save all models
        model_dir = self.save_models()
        return model_dir
    
    def evaluate_models(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all models on test set"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'nn':
                model = model.to(self.device)
                model = model.eval()
                with torch.no_grad():
                    train_tensor = torch.FloatTensor(self.X_train).to(self.device)
                    val_tensor = torch.FloatTensor(self.X_val).to(self.device)
                    test_tensor = torch.FloatTensor(self.X_test).to(self.device)
                    
                    train_pred = model(train_tensor).cpu().numpy()
                    val_pred = model(val_tensor).cpu().numpy()
                    test_pred = model(test_tensor).cpu().numpy()
            else:
                train_pred = model.predict(self.X_train)
                val_pred = model.predict(self.X_val)
                test_pred = model.predict(self.X_test)
            
            results[name] = {
                'train': {
                    'mse': mean_squared_error(self.y_train, train_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
                    'mae': mean_absolute_error(self.y_train, train_pred),
                    'r2': r2_score(self.y_train, train_pred)
                },
                'val': {
                    'mse': mean_squared_error(self.y_val, val_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_val, val_pred)),
                    'mae': mean_absolute_error(self.y_val, val_pred),
                    'r2': r2_score(self.y_val, val_pred)
                },
                'test': {
                    'mse': mean_squared_error(self.y_test, test_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
                    'mae': mean_absolute_error(self.y_test, test_pred),
                    'r2': r2_score(self.y_test, test_pred)
                }
            }
        return results

def main():
    data_type: str = sys.argv[1]
    metadata = pd.read_csv(f"data/metadata.csv")
    data = pd.read_csv(f"data/{data_type}_audio_features.csv")
    data = data.merge(metadata, on="watermelon_id")
    data = data.dropna()

    # Set memory management parameters
    torch.backends.cudnn.benchmark = True
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize and train models
    trainer = ModelTrainer(data, data_type, save_dir=f"models/{data_type}", random_state=42)
    model_dir = trainer.train_models(n_trials=10, epochs=200, early_stopping_patience=50)  # Reduced trials for testing
    results = trainer.evaluate_models()
    
    # Save evaluation results
    results_file = os.path.join(model_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Results:")
        for split, split_metrics in metrics.items():
            print(f"\t{split.upper()} Results:")
            for metric_name, value in split_metrics.items():
                print(f"\t\t{metric_name}: {value:.4f}")
    
    print(f"\nAll results and models saved in: {model_dir}")

if __name__ == "__main__":
    main() 