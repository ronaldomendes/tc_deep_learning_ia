"""
Evaluation module for LSTM model
Calculates metrics: MAE, RMSE, MAPE
"""
import os
from typing import Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.lstm import LSTMModel
from src.model.train import ModelTrainer
from src.financial.preprocessing import DataPreprocessor
from src.utils import get_plot_path, ensure_dirs


class ModelEvaluator:
    """Class responsible for evaluating the LSTM model"""

    def __init__(self, ticker: str = "KLBN3.SA"):
        self.ticker = ticker
        self.plot_path = get_plot_path(ticker)
        self.trainer = ModelTrainer(ticker=ticker)
        self.preprocessor = DataPreprocessor(ticker=ticker)

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def get_predictions(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for test set"""
        model = self.trainer.load_model()
        device = self.trainer.device

        X_test = torch.FloatTensor(data['X_test']).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test)

        return predictions.cpu().numpy(), data['y_test']

    def evaluate(self, data: dict = None) -> dict:
        """
        Evaluate model on test set

        Args:
            data: Optional preprocessed data dictionary. If None, will preprocess.

        Returns:
            Dictionary with evaluation metrics
        """
        if data is None:
            data = self.preprocessor.preprocess()

        # Get predictions (normalized)
        y_pred_normalized, y_true_normalized = self.get_predictions(data)

        # Calculate metrics on normalized data
        metrics_normalized = {
            'mae_normalized': self.calculate_mae(y_true_normalized, y_pred_normalized),
            'rmse_normalized': self.calculate_rmse(y_true_normalized, y_pred_normalized),
            'mape_normalized': self.calculate_mape(y_true_normalized, y_pred_normalized)
        }

        # Inverse transform to original scale for more interpretable metrics
        y_pred_original = self._inverse_transform_batch(y_pred_normalized)
        y_true_original = self._inverse_transform_batch(y_true_normalized)

        metrics_original = {
            'mae': self.calculate_mae(y_true_original, y_pred_original),
            'rmse': self.calculate_rmse(y_true_original, y_pred_original),
            'mape': self.calculate_mape(y_true_original, y_pred_original)
        }

        # Generate evaluation plot
        plot_path = self.plot_predictions(y_true_original, y_pred_original)

        return {
            **metrics_original,
            **metrics_normalized,
            'test_samples': len(y_true_original),
            'plot_path': plot_path
        }

    def _inverse_transform_batch(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform a batch of predictions to original scale"""
        self.preprocessor.load_scaler()

        # Flatten predictions for inverse transform
        flat_pred = predictions.flatten()
        original = self.preprocessor.inverse_transform_predictions(flat_pred)

        return original.reshape(predictions.shape)

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_samples: int = 50
    ) -> str:
        """
        Create visualization comparing predictions vs actual values

        Args:
            y_true: Actual values
            y_pred: Predicted values
            num_samples: Number of samples to plot

        Returns:
            Path to saved plot
        """
        ensure_dirs(self.ticker)

        # Take first prediction day for comparison
        y_true_first = y_true[:num_samples, 0]
        y_pred_first = y_pred[:num_samples, 0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Predictions vs Actual (line plot)
        axes[0, 0].plot(y_true_first, label='Actual', color='blue', alpha=0.7)
        axes[0, 0].plot(y_pred_first, label='Predicted', color='red', alpha=0.7)
        axes[0, 0].set_title('Predictions vs Actual (Day 1)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Close Price (R$)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        axes[0, 1].scatter(y_true_first, y_pred_first, alpha=0.5, color='purple')
        axes[0, 1].plot(
            [y_true_first.min(), y_true_first.max()],
            [y_true_first.min(), y_true_first.max()],
            'r--', label='Perfect Prediction'
        )
        axes[0, 1].set_title('Scatter: Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual Price (R$)')
        axes[0, 1].set_ylabel('Predicted Price (R$)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Error distribution
        errors = y_pred_first - y_true_first
        axes[1, 0].hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', label='Zero Error')
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].set_xlabel('Error (R$)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Percentage error over time
        pct_errors = np.abs(errors / y_true_first) * 100
        axes[1, 1].bar(range(len(pct_errors)), pct_errors, color='orange', alpha=0.7)
        axes[1, 1].axhline(y=np.mean(pct_errors), color='red', linestyle='--',
                          label=f'Mean MAPE: {np.mean(pct_errors):.2f}%')
        axes[1, 1].set_title('Percentage Error by Sample')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Percentage Error (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Evaluation plot saved to {self.plot_path}")
        return self.plot_path

    def get_metrics_summary(self) -> dict:
        """Get a summary of model metrics"""
        metrics = self.evaluate()

        summary = {
            'mae': round(metrics['mae'], 4),
            'rmse': round(metrics['rmse'], 4),
            'mape': round(metrics['mape'], 2),
            'test_samples': metrics['test_samples'],
            'interpretation': {
                'mae': f"On average, predictions differ from actual by R$ {metrics['mae']:.2f}",
                'rmse': f"Root mean square error is R$ {metrics['rmse']:.2f}",
                'mape': f"Average percentage error is {metrics['mape']:.2f}%"
            }
        }

        return summary


def evaluate_model(ticker: str = "KLBN3.SA"):
    """Convenience function to evaluate the model"""
    evaluator = ModelEvaluator(ticker=ticker)
    metrics = evaluator.evaluate()

    print("\n" + "=" * 50)
    print(f"MODEL EVALUATION RESULTS - {ticker}")
    print("=" * 50)
    print(f"MAE:  R$ {metrics['mae']:.4f}")
    print(f"RMSE: R$ {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Plot saved: {metrics['plot_path']}")
    print("=" * 50)

    return metrics


if __name__ == '__main__':
    evaluate_model()
