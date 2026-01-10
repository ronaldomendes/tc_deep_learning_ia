"""
Preprocessing module for financial data
Handles data cleaning, feature engineering, normalization and sequence creation for LSTM
"""
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import get_data_path, get_scaler_path, ensure_dirs

# Hyperparameters
SEQUENCE_LENGTH = 60  # Days of historical data to use as input
PREDICTION_HORIZON = 5  # Days to predict ahead
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


class DataPreprocessor:
    """Class responsible for preprocessing financial data for LSTM model"""

    def __init__(self, ticker: str = "KLBN3.SA"):
        self.ticker = ticker
        self.data_path = get_data_path(ticker)
        self.scaler_path = get_scaler_path(ticker)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = ['Close', 'Volume', 'SMA_7', 'SMA_21', 'Returns', 'Volatility']

    def load_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        df = pd.read_csv(self.data_path, skiprows=2)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Convert numeric columns
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values with forward fill
        df = df.ffill().bfill()

        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features"""
        # Simple Moving Averages
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()

        # Daily returns (percentage change)
        df['Returns'] = df['Close'].pct_change()

        # Volatility (rolling standard deviation of returns)
        df['Volatility'] = df['Returns'].rolling(window=21).std()

        # Drop rows with NaN values created by rolling calculations
        df = df.dropna().reset_index(drop=True)

        return df

    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Normalize features using MinMaxScaler"""
        data = df[self.feature_columns].values

        if fit:
            normalized = self.scaler.fit_transform(data)
            self.save_scaler()
        else:
            normalized = self.scaler.transform(data)

        return normalized

    def save_scaler(self):
        """Save the fitted scaler for later use in inference"""
        ensure_dirs(self.ticker)
        joblib.dump(self.scaler, self.scaler_path)

    def load_scaler(self):
        """Load a previously saved scaler"""
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

    def create_sequences(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        X: sequences of SEQUENCE_LENGTH days
        y: next PREDICTION_HORIZON days of Close price (first column)
        """
        X, y = [], []

        for i in range(len(data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            # Input sequence: all features for SEQUENCE_LENGTH days
            X.append(data[i:(i + SEQUENCE_LENGTH)])
            # Target: Close price (column 0) for next PREDICTION_HORIZON days
            y.append(data[(i + SEQUENCE_LENGTH):(i + SEQUENCE_LENGTH + PREDICTION_HORIZON), 0])

        return np.array(X), np.array(y)

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation and test sets"""
        total_samples = len(X)
        train_size = int(total_samples * TRAIN_RATIO)
        val_size = int(total_samples * VAL_RATIO)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self) -> dict:
        """
        Full preprocessing pipeline
        Returns dictionary with all processed data
        """
        # Load and clean data
        df = self.load_data()

        # Add technical indicators
        df = self.add_features(df)

        # Normalize
        normalized_data = self.normalize_data(df, fit=True)

        # Create sequences
        X, y = self.create_sequences(normalized_data)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_horizon': PREDICTION_HORIZON,
            'scaler': self.scaler,
            'raw_df': df
        }

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale"""
        # Create dummy array with same shape as original features
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, 0] = predictions.flatten() if predictions.ndim > 1 else predictions
        # Inverse transform and get only Close column
        return self.scaler.inverse_transform(dummy)[:, 0]

    def get_latest_sequence(self) -> np.ndarray:
        """Get the most recent SEQUENCE_LENGTH days for prediction"""
        df = self.load_data()
        df = self.add_features(df)
        self.load_scaler()
        normalized = self.normalize_data(df, fit=False)
        return normalized[-SEQUENCE_LENGTH:]
