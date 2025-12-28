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
from src.model.train import ModelTrainer, MODEL_PATH
from src.model.evaluate import ModelEvaluator

PATH = 'data/KLBN_data.csv'


class FinancialService:
    """Financial Service for managing all data from yfinance"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

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
        Preprocess financial data for LSTM model training
        Creates features, normalizes data and splits into train/val/test sets
        """
        try:
            result = self.preprocessor.preprocess()
            return JSONResponse(
                content={
                    'message': 'Data preprocessed successfully.',
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
                detail='Data file not found. Please call /save-data first.'
            ) from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def train_model(self):
        """Train the LSTM model"""
        try:
            # Preprocess data
            data = self.preprocessor.preprocess()

            # Train model
            results = self.trainer.train(data)

            return JSONResponse(
                content={
                    'message': 'Model trained successfully.',
                    'epochs_trained': results['epochs_trained'],
                    'best_val_loss': round(results['best_val_loss'], 6),
                    'device': results['device'],
                    'model_path': MODEL_PATH
                },
                status_code=status.HTTP_200_OK
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Data file not found. Please call /save-data first.'
            ) from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    async def predict(self):
        """
        Make predictions for the next 5 days using the trained model
        """
        try:
            # Check if model exists
            if not os.path.exists(MODEL_PATH):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail='Model not found. Please train the model first using /train.'
                )

            # Get latest sequence for prediction
            sequence = self.preprocessor.get_latest_sequence()

            # Make prediction (normalized)
            predictions_normalized = self.trainer.predict(sequence)

            # Convert back to original scale
            predictions_original = self.preprocessor.inverse_transform_predictions(
                predictions_normalized
            )

            # Generate dates for predictions (next business days)
            df = self.preprocessor.load_data()
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
            metrics = self.evaluator.get_metrics_summary()

            return JSONResponse(
                content={
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

    async def get_metrics(self):
        """Get evaluation metrics for the trained model"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail='Model not found. Please train the model first using /train.'
                )

            metrics = self.evaluator.get_metrics_summary()

            return JSONResponse(
                content={
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
