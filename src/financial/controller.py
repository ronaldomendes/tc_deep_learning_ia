"""
Financial Controller
"""
from fastapi import APIRouter

from src.financial.service import FinancialService

financial_router = APIRouter()
service = FinancialService()


@financial_router.post('/save-data')
async def save_financial_data():
    """Method responsible for getting data and save a csv file"""
    return await service.save_financial_data()


@financial_router.post('/preprocessing-data')
async def process_financial_data():
    """Method responsible for retrieve a previously saved csv file and start preprocessing"""
    return await service.preprocessing_data()


@financial_router.post('/train')
async def train_model():
    """Train the LSTM model with preprocessed data"""
    return await service.train_model()


@financial_router.post('/predict')
async def predict():
    """
    Predict the next 5 days of closing prices using the trained LSTM model.
    Returns predictions with dates and model performance metrics.
    """
    return await service.predict()


@financial_router.get('/metrics')
async def get_metrics():
    """Get evaluation metrics (MAE, RMSE, MAPE) for the trained model"""
    return await service.get_metrics()
