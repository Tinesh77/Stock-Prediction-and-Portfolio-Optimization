import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import ALL_STOCKS
from data_utils import fetch_stock_data
from feature_utils import create_features
from models_utils import load_all_models, get_all_model_predictions
from ensemble_utils import ensemble_prediction
from portfolio_utils import (
    optimize_portfolio,
    portfolio_return,
    portfolio_risk,
    sharpe_ratio,
    efficient_frontier,
    min_variance_portfolio,
    max_sharpe_portfolio
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
st.sidebar.markdown(
    "<h2 style='color:#0F4C81;'>📌 Navigation</h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

if st.sidebar.button("🏠 Home Dashboard", key="nav_home_po"):
    st.switch_page("app.py")

if st.sidebar.button("📊 Model Analysis", key="nav_ma_po"):
    st.switch_page("pages/model_analysis.py")

if st.sidebar.button("💼 Portfolio Optimization", key="nav_po_current"):
    st.experimental_rerun()

st.sidebar.markdown("---")

st.sidebar.info(
    "Portfolio optimization using Modern Portfolio Theory "
    "with ML-based expected returns."
)

# -------------------------------------------------
# Page Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style="color:#0F4C81;">💼 Portfolio Optimization Dashboard</h1>
    <p style="color:gray;font-size:16px;">
    Risk–return optimized asset allocation using Modern Portfolio Theory
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# Portfolio Configuration
# -------------------------------------------------
st.markdown("## 🧩 Portfolio Configuration")

selected_stocks = st.multiselect(
    "Select Stocks for Portfolio",
    ALL_STOCKS,
    default=ALL_STOCKS[:4]
)

if len(selected_stocks) < 2:
    st.warning("Please select at least two stocks.")
    st.stop()

# -------------------------------------------------
# Date Range
# -------------------------------------------------
st.markdown("## 📅 Date Range")

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date", pd.to_datetime("2021-01-01")
    )

with col2:
    end_date = st.date_input(
        "End Date", pd.to_datetime("today")
    )

# -------------------------------------------------
# Expected Returns & Covariance Matrix
# -------------------------------------------------
expected_returns = []
return_series = []

for stock in selected_stocks:
    df = fetch_stock_data(stock, start_date, end_date)

    if df.empty:
        st.warning(f"No data available for {stock}")
        st.stop()

    feat_df = create_features(df)

    arima, rf, lstm, scaler = load_all_models(stock)

    preds = get_all_model_predictions(
        arima, rf, lstm, scaler, feat_df
    )

    expected_returns.append(ensemble_prediction(preds))
    return_series.append(feat_df["Return"])

returns_df = pd.concat(return_series, axis=1)
returns_df.columns = selected_stocks

cov_matrix = returns_df.cov().values
cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # numerical stability

expected_returns = np.array(expected_returns)

# -------------------------------------------------
# Optimized Portfolio (Primary)
# -------------------------------------------------
weights = optimize_portfolio(expected_returns, cov_matrix)

port_ret = portfolio_return(weights, expected_returns)
port_risk = portfolio_risk(weights, cov_matrix)
port_sharpe = sharpe_ratio(port_ret, port_risk)

# -------------------------------------------------
# Benchmark Portfolios (Already Defined in Utils)
# -------------------------------------------------
min_var_weights = min_variance_portfolio(cov_matrix)
max_sharpe_weights = max_sharpe_portfolio(expected_returns, cov_matrix)

min_var_return = portfolio_return(min_var_weights, expected_returns)
min_var_risk = portfolio_risk(min_var_weights, cov_matrix)
min_var_sharpe = sharpe_ratio(min_var_return, min_var_risk)

max_sharpe_return = portfolio_return(max_sharpe_weights, expected_returns)
max_sharpe_risk = portfolio_risk(max_sharpe_weights, cov_matrix)
max_sharpe_sharpe = sharpe_ratio(max_sharpe_return, max_sharpe_risk)

# -------------------------------------------------
# Executive Metrics
# -------------------------------------------------
st.markdown("## 📊 Optimized Portfolio Metrics")

c1, c2, c3 = st.columns(3, gap="large")

c1.metric("Expected Return", f"{port_ret:.2%}")
c2.metric("Risk (Volatility)", f"{port_risk:.2%}")
c3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")

# -------------------------------------------------
# Portfolio Comparison Metrics (RESTORED)
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Portfolio Comparison")

comparison_df = pd.DataFrame({
    "Portfolio": [
        "Optimized (ML + MPT)",
        "Minimum Variance",
        "Maximum Sharpe"
    ],
    "Expected Return": [
        f"{port_ret:.2%}",
        f"{min_var_return:.2%}",
        f"{max_sharpe_return:.2%}"
    ],
    "Risk (Volatility)": [
        f"{port_risk:.2%}",
        f"{min_var_risk:.2%}",
        f"{max_sharpe_risk:.2%}"
    ],
    "Sharpe Ratio": [
        f"{port_sharpe:.2f}",
        f"{min_var_sharpe:.2f}",
        f"{max_sharpe_sharpe:.2f}"
    ]
})

st.dataframe(comparison_df, use_container_width=True)

# -------------------------------------------------
# Asset Allocation
# -------------------------------------------------
st.markdown("---")
st.markdown("## ⚖️ Optimal Asset Allocation")

weights_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Weight": weights
})

weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
st.dataframe(weights_df, use_container_width=True)

# -------------------------------------------------
# Efficient Frontier (Interactive)
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📉 Efficient Frontier")

ef_risks, ef_returns = efficient_frontier(
    expected_returns, cov_matrix
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ef_risks,
        y=ef_returns,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="#0F4C81", width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=[port_risk],
        y=[port_ret],
        mode="markers",
        name="Optimized Portfolio",
        marker=dict(size=12, color="red")
    )
)

fig.update_layout(
    xaxis_title="Risk (Volatility)",
    yaxis_title="Expected Return",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# PORTFOLIO WEIGHTS PIE CHART
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🥧 Portfolio Weight Distribution")

pie_fig = go.Figure(
    data=[
        go.Pie(
            labels=selected_stocks,
            values=weights,
            hole=0.4
        )
    ]
)

pie_fig.update_layout(
    template="plotly_white"
)

st.plotly_chart(pie_fig, use_container_width=True)

# -------------------------------------------------
# DOWNLOADABLE PORTFOLIO REPORT
# -------------------------------------------------
st.markdown("---")
st.markdown("## 📥 Download Portfolio Report")

report_df = weights_df.copy()
report_df["Expected Portfolio Return"] = port_ret
report_df["Portfolio Risk"] = port_risk
report_df["Sharpe Ratio"] = port_sharpe

csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Portfolio as CSV",
    data=csv,
    file_name="optimized_portfolio.csv",
    mime="text/csv"
)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.info(
    "This dashboard compares ML-optimized portfolios with "
    "classical Modern Portfolio Theory benchmarks."
)
