# 📈 Stock Market Prediction & Portfolio Optimization Dashboard

An end-to-end **machine learning + quantitative finance system** that predicts stock returns and constructs optimal portfolios using **Modern Portfolio Theory (MPT)**. The project integrates statistical models, classical ML, and deep learning to generate **data-driven investment insights** through an interactive dashboard.

---

## 🚀 Overview

This project solves two core financial problems:

- **Return Prediction** → Forecast future stock returns using ML models  
- **Portfolio Optimization** → Allocate capital efficiently to balance risk and return  

Instead of just predicting stock prices, this system translates predictions into **actionable portfolio decisions**, similar to real-world quantitative investment systems.

---

## 🧠 Key Features

- 📊 Multi-model return prediction:
  - ARIMA (time-series)
  - Random Forest (non-linear ML)
  - LSTM (deep learning)

- 🔗 Ensemble modeling for stable expected returns  
- 📉 Risk modeling using covariance matrix  
- ⚖️ Portfolio optimization using Modern Portfolio Theory  
- 📈 Efficient Frontier visualization  
- ⭐ Portfolio comparison:
  - Optimized Portfolio
  - Minimum Variance Portfolio
  - Maximum Sharpe Portfolio

- 📊 Model analysis dashboard with:
  - SMA, EMA (trend analysis)
  - Volatility indicators
  - Return distribution
  - Accuracy metrics (MAE, RMSE, Directional Accuracy)

- 💼 Portfolio dashboard with:
  - Optimal weights
  - Risk-return metrics
  - Efficient frontier (interactive)
  - Downloadable CSV report

---

## 🏗️ System Architecture
Market Data (Yahoo Finance)
↓
Feature Engineering
↓
ML Models (ARIMA, RF, LSTM)
↓
Ensemble Expected Returns
↓
Covariance Matrix (Risk)
↓
Portfolio Optimization (MPT)
↓
Efficient Frontier + Allocation
↓
Streamlit Dashboard


---

## 📊 Finance Concepts Used

- Log Returns  
- Moving Averages (SMA, EMA)  
- Volatility (Rolling Std Dev)  
- Drawdown  
- Covariance Matrix  
- Diversification  
- Portfolio Return & Risk  
- Sharpe Ratio  
- Efficient Frontier  
- Modern Portfolio Theory (MPT)

---

## 🤖 Machine Learning Models

| Model | Purpose |
|------|--------|
| ARIMA | Captures linear time-series patterns |
| Random Forest | Captures non-linear relationships |
| LSTM | Captures temporal dependencies |
| Ensemble | Combines all models for stability |

---

## 📐 Model Evaluation

- **MAE (Mean Absolute Error)** → Average prediction error  
- **RMSE (Root Mean Squared Error)** → Penalizes large errors  
- **Directional Accuracy** → % of correct market direction  

> Financial data is noisy; the focus is on **robust, risk-aware predictions** rather than perfect accuracy.

---

## 💼 Portfolio Optimization

Using **Modern Portfolio Theory**, the system:

- Maximizes expected return  
- Minimizes risk using covariance  
- Generates optimal asset allocation  

### Outputs:
- Optimized Portfolio  
- Minimum Variance Portfolio  
- Maximum Sharpe Portfolio  
- Efficient Frontier  

---

## 🖥️ Dashboard Pages

### 🏠 Home
- Stock selection  
- Model predictions  
- Quick overview  

### 📊 Model Analysis
- Price trends  
- Volatility analysis  
- Return distribution  
- Accuracy metrics  

### 💼 Portfolio Optimization
- Asset selection  
- Optimal weights  
- Risk-return metrics  
- Efficient frontier  
- Portfolio comparison  
- CSV download  

---

## ⚙️ Tech Stack

- Python  
- pandas, numpy  
- scikit-learn  
- statsmodels  
- TensorFlow / Keras  
- cvxpy  
- yfinance  
- plotly  
- streamlit  

---

## 📁 Project Structure
├── app.py
├── config.py
├── data_utils.py
├── feature_utils.py
├── model_utils.py
├── ensemble_utils.py
├── portfolio_utils.py
├── pages/
│ ├── model_analysis.py
│ └── portfolio_optimization.py
├── models/
│ ├── arima/
│ ├── random_forest/
│ └── lstm/



---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
pip install -r requirements.txt
streamlit run app.py
