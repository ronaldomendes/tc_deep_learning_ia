"""
Finance Service
"""
import yfinance as yf
from fastapi import status
from fastapi.exceptions import HTTPException


class FinanceService:
    """Finance Service for managing all data from yfinance"""

    async def get_market_data(self):
        """
        Initially this service is only getting data from Klabin (KLBN3.SA).
        In the future this service can get all data dynamically.
        :return:
        """
        symbol = 'KLBN3.SA'
        start_date = '2020-01-01'
        end_date = '2025-10-31'
        path = 'data/KLBN_data.csv'

        try:
            # Use a função download para obter os dados
            df = yf.download(symbol, start=start_date, end=end_date)
            df.to_csv(path)
            return {'message': 'File saved successfully.'}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
