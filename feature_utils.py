import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial features from price data.
    Input: DataFrame with columns ['Close', 'Volume']
    Output: Feature-enhanced DataFrame
    """

    df = df.copy()

    # -------------------------
    # 1. Log Returns
    # -------------------------
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # -------------------------
    # 2. Simple Moving Averages
    # -------------------------
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # -------------------------
    # 3. Exponential Moving Averages
    # -------------------------
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # -------------------------
    # 4. Volatility (Rolling Std)
    # -------------------------
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Return'].rolling(window=20).std()

    # -------------------------
    # 5. Drawdown
    # -------------------------
    rolling_max = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - rolling_max) / rolling_max

    # -------------------------
    # 6. Volume Z-score (Liquidity Shock)
    # -------------------------
    df['Volume_Z'] = (
        (df['Volume'] - df['Volume'].rolling(20).mean()) /
        df['Volume'].rolling(20).std()
    )

    # -------------------------
    # Cleanup
    # -------------------------
    df.dropna(inplace=True)

    return df
