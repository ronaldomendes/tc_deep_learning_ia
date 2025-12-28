"""
LSTM Model Architecture for Stock Price Prediction
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for time series prediction
    Architecture: Input -> LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> Linear(output)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        output_size: int = 5,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize the LSTM model

        Args:
            input_size: Number of input features
                (default: 6 for Close, Volume, SMA_7, SMA_21, Returns, Volatility)
            hidden_size_1: Number of hidden units in first LSTM layer
            hidden_size_2: Number of hidden units in second LSTM layer
            output_size: Number of days to predict (default: 5)
            num_layers: Number of LSTM layers per block
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout after first LSTM
        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout after second LSTM
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        # Take only the last time step output
        last_output = lstm2_out[:, -1, :]

        # Pass through fully connected layer
        output = self.fc(last_output)

        return output

    def get_config(self) -> dict:
        """Return model configuration as dictionary"""
        return {
            'input_size': self.lstm1.input_size,
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2,
            'output_size': self.fc.out_features,
            'num_layers': self.num_layers,
            'dropout': self.dropout1.p
        }
