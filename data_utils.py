import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=3600 , show_spinner=False)
def fetch_stock_data(ticker: str,start_date,end_date,interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval ,progress=False,auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[["Close","Volume"]]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    

@st.cache_data(ttl=60 , show_spinner=False)
def fetch_intraday_data(ticker : str) -> pd.DataFrame:
    try:
        df = yf.download(ticker , period="1d" , interval="1m" , progress=False)
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close","Volume"]]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()
    
        

        
    
    
    
