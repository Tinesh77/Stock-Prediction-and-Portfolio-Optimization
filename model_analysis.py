import streamlit as st
import pandas as pd

from config import ALL_STOCKS
from data_utils import fetch_stock_data
from feature_utils import create_features
from models_utils import load_all_models, get_all_model_predictions
from ensemble_utils import ensemble_prediction


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Model Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# Sidebar Navigation (Explicit & Stable)
# -------------------------------------------------
st.sidebar.markdown(
    "<h2 style='color:#0F4C81;'>📌 Navigation</h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

if st.sidebar.button("🏠 Home Dashboard", key="nav_home_ma"):
    st.switch_page("app.py")

if st.sidebar.button("📊 Model Analysis", key="nav_ma_current"):
    st.experimental_rerun()

if st.sidebar.button("💼 Portfolio Optimization", key="nav_po_ma"):
    st.switch_page("pages/portfolio_optimization.py")

st.sidebar.markdown("---")

st.sidebar.info(
    "This page focuses on model behaviour, trend interpretation, "
    "and risk analysis using multiple ML models."
)


# -------------------------------------------------
# Page Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style="color:#0F4C81;">📊 Model Analysis Dashboard</h1>
    <p style="color:gray;font-size:16px;">
    Comparative analysis of ARIMA, Random Forest, and LSTM predictions
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# -------------------------------------------------
# Stock & Date Selection
# -------------------------------------------------
st.markdown("## 🔍 Stock & Date Selection")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker = st.selectbox("Select Stock", ALL_STOCKS)

with col2:
    start_date = st.date_input(
        "Start Date", pd.to_datetime("2021-01-01")
    )

with col3:
    end_date = st.date_input(
        "End Date", pd.to_datetime("today")
    )


# -------------------------------------------------
# Data Pipeline
# -------------------------------------------------
df = fetch_stock_data(ticker, start_date, end_date)

if df.empty:
    st.warning("No data available for the selected stock.")
    st.stop()

feature_df = create_features(df)


# -------------------------------------------------
# Load Models & Predictions
# -------------------------------------------------
arima_model, rf_model, lstm_model, lstm_scaler = load_all_models(ticker)

predictions = get_all_model_predictions(
    arima_model,
    rf_model,
    lstm_model,
    lstm_scaler,
    feature_df
)

ensemble_pred = ensemble_prediction(predictions)


# -------------------------------------------------
# Model Prediction Summary
# -------------------------------------------------
st.markdown("## 📊 Model Predicted Returns")

col1, col2, col3, col4 = st.columns(4, gap="large")

col1.metric("ARIMA", f"{predictions['ARIMA']:.2%}")
col2.metric("Random Forest", f"{predictions['Random Forest']:.2%}")
col3.metric("LSTM", f"{predictions['LSTM']:.2%}")
col4.metric(
    "Ensemble",
    f"{ensemble_pred:.2%}",
    help="Weighted combination of all models"
)

st.caption(
    "Predicted values represent next-period expected log returns."
)


# -------------------------------------------------
# Price Trend & Moving Averages
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Price Trend & Moving Averages")

st.line_chart(
    feature_df[['Close', 'SMA_10', 'SMA_20', 'SMA_50']]
)

st.caption(
    "Moving averages highlight short-, medium-, and long-term price trends."
)


# -------------------------------------------------
# Volatility & Risk Analysis
# -------------------------------------------------
st.markdown("---")
st.markdown("## ⚠️ Volatility & Risk Analysis")

st.line_chart(
    feature_df[['Volatility_10', 'Volatility_20']]
)

st.caption(
    "Rolling volatility reflects changing market risk over time."
)


# -------------------------------------------------
# Return Distribution
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📉 Recent Return Distribution")

st.bar_chart(feature_df["Return"].tail(60))

st.caption(
    "Return distribution highlights market noise and variability, "
    "justifying ensemble-based predictions."
)


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.info(
    "For asset allocation and investment decisions, "
    "navigate to the **Portfolio Optimization** section."
)
