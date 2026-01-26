# config.py

# Core NIFTY 50 stocks (high liquidity)
NIFTY50_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS"
]

# Additional NSE Large & Mid Cap stocks
NSE_ADDITIONAL_STOCKS = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "TATAMOTORS.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "HINDUNILVR.NS",
    "ASIANPAINT.NS",
    "DMART.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS"
]

# Combined stock universe
ALL_STOCKS = sorted(list(set(NIFTY50_STOCKS + NSE_ADDITIONAL_STOCKS)))
