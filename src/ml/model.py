"""
ML Model module
Responsible for:
- Building LSTM model
- Training
- Saving
- Loading
- Predicting stock prices
"""

import os
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MODEL_PATH = "models/lstm_stock_model.h5"

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------

def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Build and compile an LSTM model.

    :param input_shape: (timesteps, features)
    :return: Compiled Keras model
    """
    model = Sequential()

    model.add(LSTM(
        units=50,
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(Dropout(0.2))

    model.add(LSTM(
        units=50,
        return_sequences=False
    ))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32
):
    """
    Train LSTM model and save it to disk.
    """

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    return model

# ------------------------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------------------------

def load_trained_model():
    """
    Load trained LSTM model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Train the model before inference."
        )

    return load_model(MODEL_PATH)

# ------------------------------------------------------------------------------
# Prediction (Inference)
# ------------------------------------------------------------------------------

def predict_next_price(
    model,
    input_sequence: np.ndarray
) -> float:
    """
    Predict next stock price based on a window of historical prices.

    :param model: Loaded LSTM model
    :param input_sequence: shape (window_size, 1)
    :return: Predicted value
    """
    if input_sequence.ndim != 2:
        raise ValueError(
            "Input sequence must have shape (window_size, 1)"
        )

    input_sequence = input_sequence.reshape(1, input_sequence.shape[0], 1)

    prediction = model.predict(input_sequence)

    return float(prediction[0][0])
