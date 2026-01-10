"""
Training module for LSTM model
"""
import json
import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.lstm import LSTMModel
from src.financial.preprocessing import DataPreprocessor
from src.utils import get_model_path, get_config_path, ensure_dirs


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class ModelTrainer:
    """Class responsible for training the LSTM model"""

    def __init__(
        self,
        ticker: str = "KLBN3.SA",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None
    ):
        self.ticker = ticker
        self.model_path = get_model_path(ticker)
        self.config_path = get_config_path(ticker)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # Set device (GPU if available, otherwise CPU)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.train_losses = []
        self.val_losses = []

    def prepare_data(self, data: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Convert numpy arrays to PyTorch DataLoaders"""
        # Convert to tensors
        X_train = torch.FloatTensor(data['X_train'])
        y_train = torch.FloatTensor(data['y_train'])
        X_val = torch.FloatTensor(data['X_val'])
        y_val = torch.FloatTensor(data['y_val'])
        X_test = torch.FloatTensor(data['X_test'])
        y_test = torch.FloatTensor(data['y_test'])

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def create_model(self, input_size: int, output_size: int) -> LSTMModel:
        """Create and initialize the LSTM model"""
        self.model = LSTMModel(
            input_size=input_size,
            output_size=output_size
        ).to(self.device)
        return self.model

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, data: dict) -> dict:
        """
        Full training pipeline

        Args:
            data: Dictionary from DataPreprocessor.preprocess()

        Returns:
            Dictionary with training results
        """
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(data)

        # Get dimensions
        input_size = data['X_train'].shape[2]  # Number of features
        output_size = data['y_train'].shape[1]  # Prediction horizon

        # Create model
        self.create_model(input_size, output_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Early stopping
        early_stopping = EarlyStopping(patience=self.patience)

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        print(f"Training on {self.device}")
        print(f"Train samples: {len(data['X_train'])}, Val samples: {len(data['X_val'])}")
        print("-" * 50)

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}] "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early stopping check
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Save model
        self.save_model()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.train_losses),
            'device': str(self.device)
        }

    def save_model(self):
        """Save the trained model and configuration"""
        ensure_dirs(self.ticker)

        # Save model weights
        torch.save(self.model.state_dict(), self.model_path)

        # Save configuration
        config = self.model.get_config()
        config['ticker'] = self.ticker
        config['training'] = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs_trained': len(self.train_losses),
            'best_val_loss': min(self.val_losses) if self.val_losses else None
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {self.model_path}")
        print(f"Config saved to {self.config_path}")

    def load_model(self) -> LSTMModel:
        """Load a trained model"""
        # Load config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Create model with saved config
        self.model = LSTMModel(
            input_size=config['input_size'],
            hidden_size_1=config['hidden_size_1'],
            hidden_size_2=config['hidden_size_2'],
            output_size=config['output_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        return self.model

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Make prediction for a single sequence

        Args:
            sequence: Input array of shape (sequence_length, input_size)

        Returns:
            Predictions array of shape (prediction_horizon,)
        """
        if self.model is None:
            self.load_model()

        self.model.eval()

        with torch.no_grad():
            # Add batch dimension
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            prediction = self.model(x)

        return prediction.cpu().numpy().squeeze()


def train_model(ticker: str = "KLBN3.SA"):
    """Convenience function to train the model from scratch"""
    preprocessor = DataPreprocessor(ticker=ticker)
    data = preprocessor.preprocess()

    trainer = ModelTrainer(ticker=ticker)
    results = trainer.train(data)

    return results


if __name__ == '__main__':
    train_model()
