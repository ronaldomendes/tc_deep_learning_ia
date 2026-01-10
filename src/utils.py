"""
Utility functions for dynamic path generation based on ticker symbol
"""
import os
import re
from typing import List


def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker format.
    Accepts:
    - Brazilian tickers: XXXX3.SA, XXXX4.SA, XXXX11.SA
    - International tickers: AAPL, GOOGL, MSFT
    """
    # Brazilian pattern: letters + number(s) + .SA
    brazilian_pattern = r'^[A-Z]{4}\d{1,2}\.SA$'
    # International pattern: just letters
    international_pattern = r'^[A-Z]{1,5}$'

    return bool(re.match(brazilian_pattern, ticker.upper()) or
                re.match(international_pattern, ticker.upper()))


def get_data_dir(ticker: str) -> str:
    """Get data directory for a ticker"""
    return f"data/{ticker}"


def get_data_path(ticker: str) -> str:
    """Get CSV data path for a ticker"""
    return f"data/{ticker}/data.csv"


def get_scaler_path(ticker: str) -> str:
    """Get scaler path for a ticker"""
    return f"data/{ticker}/scaler.pkl"


def get_model_dir(ticker: str) -> str:
    """Get model directory for a ticker"""
    return f"models/{ticker}"


def get_model_path(ticker: str) -> str:
    """Get model weights path for a ticker"""
    return f"models/{ticker}/lstm_model.pt"


def get_config_path(ticker: str) -> str:
    """Get model config path for a ticker"""
    return f"models/{ticker}/model_config.json"


def get_plot_path(ticker: str) -> str:
    """Get evaluation plot path for a ticker"""
    return f"models/{ticker}/evaluation_plot.png"


def ensure_dirs(ticker: str):
    """Create necessary directories for a ticker"""
    os.makedirs(get_data_dir(ticker), exist_ok=True)
    os.makedirs(get_model_dir(ticker), exist_ok=True)


def list_available_tickers() -> List[str]:
    """List all tickers that have trained models"""
    tickers = []
    models_dir = "models"

    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item, "lstm_model.pt")
            if os.path.isdir(os.path.join(models_dir, item)) and os.path.exists(model_path):
                tickers.append(item)

    return sorted(tickers)
