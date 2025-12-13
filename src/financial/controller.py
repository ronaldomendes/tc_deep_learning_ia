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
