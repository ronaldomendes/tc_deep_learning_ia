"""
Financial Controller
"""
from fastapi import APIRouter

from src.financial.service import FinancialService

financial_router = APIRouter()
service = FinancialService()


@financial_router.post('/{ticker}/save-data')
async def save_financial_data(ticker: str):
    """
    Download and save financial data for a specific ticker from Yahoo Finance.

    - **ticker**: Stock ticker symbol (e.g., KLBN3.SA, PETR4.SA, AAPL)
    """
    return await service.save_financial_data(ticker)


@financial_router.post('/{ticker}/preprocessing-data')
async def process_financial_data(ticker: str):
    """
    Preprocess financial data for LSTM model training.

    - **ticker**: Stock ticker symbol
    """
    return await service.preprocessing_data(ticker)


@financial_router.post('/{ticker}/train')
async def train_model(ticker: str):
    """
    Train the LSTM model with preprocessed data for a specific ticker.

    - **ticker**: Stock ticker symbol
    """
    return await service.train_model(ticker)


@financial_router.post('/{ticker}/predict')
async def predict(ticker: str):
    """
    Predict the next 5 days of closing prices using the trained LSTM model.

    - **ticker**: Stock ticker symbol

    Returns predictions with dates and model performance metrics.
    """
    return await service.predict(ticker)


@financial_router.get('/{ticker}/metrics')
async def get_metrics(ticker: str):
    """
    Get evaluation metrics (MAE, RMSE, MAPE) for the trained model.

    - **ticker**: Stock ticker symbol
    """
    return await service.get_metrics(ticker)


@financial_router.get('/tickers')
async def list_tickers():
    """
    List all tickers that have trained models available.
    """
    return await service.list_tickers()
