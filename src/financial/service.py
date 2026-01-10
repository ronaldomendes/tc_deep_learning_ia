"""
Financial Service
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from src.financial.preprocessing import DataPreprocessor, PREDICTION_HORIZON
from src.model.train import ModelTrainer
from src.model.evaluate import ModelEvaluator
from src.utils import (
    get_data_path, get_model_path, ensure_dirs,
    list_available_tickers, validate_ticker
)


class FinancialService:
    """Financial Service for managing all data from yfinance"""

    async def save_financial_data(self, ticker: str):
        """
        Download and save financial data for any ticker from Yahoo Finance.
        """
        # Validate ticker format
        if not validate_ticker(ticker):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid ticker format: {ticker}. Use format like KLBN3.SA or AAPL'
            )

        data_path = get_data_path(ticker)
        start_date = '2020-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            if os.path.exists(data_path):
                return JSONResponse(
                    content={
                        'message': f'Data file for {ticker} already exists.',
                        'ticker': ticker,
                        'path': data_path
                    },
                    status_code=status.HTTP_208_ALREADY_REPORTED
                )

            ensure_dirs(ticker)
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                return JSONResponse(
                    content={
                        'message': f'No data found for {ticker}. Check if ticker is valid.',
                        'ticker': ticker
                    },
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
                )

            df.to_csv(data_path)
            return JSONResponse(
                content={
                    'message': f'Data for {ticker} saved successfully.',
                    'ticker': ticker,
                    'path': data_path,
                    'records': len(df)
                },
                status_code=status.HTTP_201_CREATED
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def preprocessing_data(self, ticker: str):
        """
        Preprocess financial data for LSTM model training
        Creates features, normalizes data and splits into train/val/test sets
        """
        try:
            preprocessor = DataPreprocessor(ticker=ticker)
            result = preprocessor.preprocess()
            return JSONResponse(
                content={
                    'message': f'Data for {ticker} preprocessed successfully.',
                    'ticker': ticker,
                    'train_samples': len(result['X_train']),
                    'val_samples': len(result['X_val']),
                    'test_samples': len(result['X_test']),
                    'features': result['feature_columns'],
                    'sequence_length': result['sequence_length'],
                    'prediction_horizon': result['prediction_horizon']
                },
                status_code=status.HTTP_200_OK
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Data file for {ticker} not found. Please call /{ticker}/save-data first.'
            ) from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def train_model(self, ticker: str):
        """Train the LSTM model for a specific ticker"""
        try:
            preprocessor = DataPreprocessor(ticker=ticker)
            trainer = ModelTrainer(ticker=ticker)

            # Preprocess data
            data = preprocessor.preprocess()

            # Train model
            results = trainer.train(data)

            return JSONResponse(
                content={
                    'message': f'Model for {ticker} trained successfully.',
                    'ticker': ticker,
                    'epochs_trained': results['epochs_trained'],
                    'best_val_loss': round(results['best_val_loss'], 6),
                    'device': results['device'],
                    'model_path': get_model_path(ticker)
                },
                status_code=status.HTTP_200_OK
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Data file for {ticker} not found. Please call /{ticker}/save-data first.'
            ) from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def predict(self, ticker: str):
        """
        Make predictions for the next 5 days using the trained model
        """
        try:
            model_path = get_model_path(ticker)
            preprocessor = DataPreprocessor(ticker=ticker)
            trainer = ModelTrainer(ticker=ticker)
            evaluator = ModelEvaluator(ticker=ticker)

            # Check if model exists
            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f'Model for {ticker} not found. '
                        f'Please train the model first using /{ticker}/train.'
                    )
                )

            # Get latest sequence for prediction
            sequence = preprocessor.get_latest_sequence()

            # Make prediction (normalized)
            predictions_normalized = trainer.predict(sequence)

            # Convert back to original scale
            predictions_original = preprocessor.inverse_transform_predictions(
                predictions_normalized
            )

            # Generate dates for predictions (next business days)
            df = preprocessor.load_data()
            last_date = df['Date'].max()
            prediction_dates = pd.bdate_range(
                start=last_date + timedelta(days=1),
                periods=PREDICTION_HORIZON
            )

            # Format predictions
            predictions_list = [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_close': round(float(price), 2)
                }
                for date, price in zip(prediction_dates, predictions_original)
            ]

            # Get model metrics
            metrics = evaluator.get_metrics_summary()

            return JSONResponse(
                content={
                    'ticker': ticker,
                    'predictions': predictions_list,
                    'model_metrics': {
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape']
                    },
                    'last_known_date': last_date.strftime('%Y-%m-%d'),
                    'currency': 'BRL'
                },
                status_code=status.HTTP_200_OK
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            ) from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def get_metrics(self, ticker: str):
        """Get evaluation metrics for the trained model"""
        try:
            model_path = get_model_path(ticker)
            evaluator = ModelEvaluator(ticker=ticker)

            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f'Model for {ticker} not found. '
                        f'Please train the model first using /{ticker}/train.'
                    )
                )

            metrics = evaluator.get_metrics_summary()

            return JSONResponse(
                content={
                    'ticker': ticker,
                    'metrics': {
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape']
                    },
                    'test_samples': metrics['test_samples'],
                    'interpretation': metrics['interpretation']
                },
                status_code=status.HTTP_200_OK
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def list_tickers(self):
        """List all tickers with trained models"""
        try:
            tickers = list_available_tickers()
            return JSONResponse(
                content={
                    'tickers': tickers,
                    'count': len(tickers)
                },
                status_code=status.HTTP_200_OK
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
