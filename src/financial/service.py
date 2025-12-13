"""
Financial Service
"""
import os

import pandas as pd
import yfinance as yf
from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

PATH = 'data/KLBN_data.csv'


class FinancialService:
    """Financial Service for managing all data from yfinance"""

    async def save_financial_data(self):
        """
        Initially this service is only getting data from Klabin (KLBN3.SA).
        In the future this service can get all data dynamically.
        :return:
        """
        symbol = 'KLBN3.SA'
        start_date = '2020-01-01'
        end_date = '2025-10-31'

        try:
            if os.path.exists(PATH):
                return JSONResponse(
                    content={'message': 'File already saved.'},
                    status_code=status.HTTP_208_ALREADY_REPORTED
                )

            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                return JSONResponse(
                    content={'message': 'Empty dataset! Check all parameters before try again.'},
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
                )

            df.to_csv(PATH)
            return JSONResponse(
                content={'message': 'File saved successfully.'},
                status_code=status.HTTP_201_CREATED
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def preprocessing_data(self):
        """
        TODO -> need to finish this method
        :return:
        """
        df = pd.read_csv(PATH, skiprows=3)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        return ''
