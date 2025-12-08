"""
Financial Service
"""
import os

import yfinance as yf
from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse


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
        path = 'data/KLBN_data.csv'

        try:
            if os.path.exists(path):
                return JSONResponse(
                    content={'message': 'File already saved.'},
                    status_code=status.HTTP_208_ALREADY_REPORTED
                )

            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
            df.to_csv(path)
            return JSONResponse(
                content={'message': 'File saved successfully.'},
                status_code=status.HTTP_201_CREATED
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
