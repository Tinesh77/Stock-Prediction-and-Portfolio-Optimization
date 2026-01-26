import os
import joblib
import pandas as pd
from config import ALL_STOCKS
from data_utils import fetch_stock_data
from feature_utils import create_features
from models_utils import train_arima,train_random_forest,prepare_lstm_data,build_lstm


BASE_MODEL_PATH = "models"

ARIMA_PATH = os.path.join(BASE_MODEL_PATH, "arima")
RF_PATH = os.path.join(BASE_MODEL_PATH, "random_forest")
LSTM_PATH = os.path.join(BASE_MODEL_PATH, "lstm")

os.makedirs(ARIMA_PATH, exist_ok=True)
os.makedirs(RF_PATH, exist_ok=True)
os.makedirs(LSTM_PATH, exist_ok=True)

for ticker in ALL_STOCKS:
    print(f"\nTraining models for {ticker}...")

    df = fetch_stock_data(
        ticker=ticker,
        start_date="2019-01-01",
        end_date=pd.to_datetime("today")
    )

    if df.empty or len(df) < 300:
        print(f"Skipping {ticker}: insufficient data")
        continue

    feature_df = create_features(df)

    try:
        arima_model = train_arima(feature_df['Return'])
        arima_file = os.path.join(ARIMA_PATH, f"{ticker}_arima.pkl")
        joblib.dump(arima_model, arima_file)
        print("ARIMA saved")
    except Exception as e:
        print(f"ARIMA failed for {ticker}")
    try:
        rf_model = train_random_forest(feature_df)
        rf_file = os.path.join(RF_PATH, f"{ticker}_rf.pkl")
        joblib.dump(rf_model, rf_file)
        print("Random Forest saved")
    except Exception:
        print(f"RF failed for {ticker}")
        
    try:
        X_lstm, y_lstm, scaler = prepare_lstm_data(
            feature_df['Return'],
            window=30
        )

        lstm_model = build_lstm(
            input_shape=(X_lstm.shape[1], X_lstm.shape[2])
        )

        lstm_model.fit(
            X_lstm,
            y_lstm,
            epochs=10,
            batch_size=32,
            verbose=0
        )

        lstm_model.save(
            os.path.join(LSTM_PATH, f"{ticker}_lstm.h5")
        )

        joblib.dump(
            scaler,
            os.path.join(LSTM_PATH, f"{ticker}_scaler.pkl")
        )

        print("LSTM + scaler saved")

    except Exception:
        print(f"LSTM failed for {ticker}")
