import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model


def train_arima(returns: pd.Series):
    """
    Train ARIMA model on log returns.
    """
    model = ARIMA(returns, order=(5, 1, 0))
    fitted_model = model.fit()
    return fitted_model

def predict_arima(model, steps: int = 1):
    """
    Predict future returns using ARIMA.
    """
    forecast = model.forecast(steps=steps)
    return forecast.iloc[-1]

def train_random_forest(df: pd.DataFrame):
    """
    Train Random Forest on engineered features.
    """
    features = [
        'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_10', 'EMA_20',
        'Volatility_10', 'Volatility_20',
        'Volume_Z'
    ]

    X = df[features]
    y = df['Return']

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X, y)
    return rf

def predict_random_forest(model, latest_row: pd.DataFrame):
    """
    Predict next return using Random Forest.
    """
    return model.predict(latest_row)[0]


def prepare_lstm_data(series, window=30):
    """
    Prepare sliding window data for LSTM.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_lstm(model, last_window, scaler):
    pred = model.predict(last_window)
    return scaler.inverse_transform(pred)[0][0]

def get_all_model_predictions(
    arima_model,
    rf_model,
    lstm_model,
    lstm_scaler,
    feature_df
):
    """
    Return predictions from all models together.
    """
    arima_pred = predict_arima(arima_model)

    rf_features = feature_df[
        ['SMA_10','SMA_20','SMA_50','EMA_10','EMA_20',
         'Volatility_10','Volatility_20','Volume_Z']
    ].iloc[[-1]]

    rf_pred = predict_random_forest(rf_model, rf_features)

    lstm_window = feature_df['Return'].values[-30:].reshape(1, 30, 1)
    lstm_pred = predict_lstm(lstm_model, lstm_window, lstm_scaler)

    return {
        "ARIMA": arima_pred,
        "Random Forest": rf_pred,
        "LSTM": lstm_pred
    }


BASE_MODEL_PATH = "models"

ARIMA_PATH = os.path.join(BASE_MODEL_PATH, "arima")
RF_PATH = os.path.join(BASE_MODEL_PATH, "random_forest")
LSTM_PATH = os.path.join(BASE_MODEL_PATH, "lstm")


def load_arima_model(ticker: str):
    path = os.path.join(ARIMA_PATH, f"{ticker}_arima.pkl")
    return joblib.load(path)


def load_rf_model(ticker: str):
    path = os.path.join(RF_PATH, f"{ticker}_rf.pkl")
    return joblib.load(path)


def load_lstm_model(ticker: str):
    model_path = os.path.join(LSTM_PATH, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(LSTM_PATH, f"{ticker}_scaler.pkl")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    return model, scaler


@st.cache_resource
def load_all_models(ticker: str):
    arima = load_arima_model(ticker)
    rf = load_rf_model(ticker)
    lstm, scaler = load_lstm_model(ticker)
    return arima, rf, lstm, scaler