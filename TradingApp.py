import streamlit as st
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

ALL_STOCKS = {
    # DJIA
    "DJIA:Apple": "AAPL", "DJIA:Microsoft": "MSFT", "DJIA:Visa": "V", "DJIA:Walmart": "WMT",
    "DJIA:Procter & Gamble": "PG", "DJIA:Nike": "NKE", "DJIA:Coca-Cola": "KO", "DJIA:Intel": "INTC",
    "DJIA:Cisco": "CSCO", "DJIA:Salesforce": "CRM", "DJIA:Boeing": "BA", "DJIA:Goldman Sachs": "GS",
    "DJIA:3M": "MMM", "DJIA:Chevron": "CVX", "DJIA:McDonald's": "MCD", "DJIA:IBM": "IBM",
    "DJIA:Caterpillar": "CAT", "DJIA:Merck": "MRK", "DJIA:Johnson & Johnson": "JNJ",
    "DJIA:American Express": "AXP", "DJIA:Walt Disney": "DIS", "DJIA:Dow Inc.": "DOW",
    "DJIA:Travelers": "TRV", "DJIA:Home Depot": "HD", "DJIA:Verizon": "VZ",
    "DJIA:UnitedHealth": "UNH", "DJIA:Amgen": "AMGN", "DJIA:Honeywell": "HON",

    # NSEI
    "NSEI:Reliance": "RELIANCE.NS", "NSEI:TCS": "TCS.NS", "NSEI:Infosys": "INFY.NS",
    "NSEI:HDFC Bank": "HDFCBANK.NS", "NSEI:ICICI Bank": "ICICIBANK.NS", "NSEI:Kotak Bank": "KOTAKBANK.NS",
    "NSEI:Axis Bank": "AXISBANK.NS", "NSEI:HUL": "HINDUNILVR.NS", "NSEI:ITC": "ITC.NS",
    "NSEI:Bajaj Finance": "BAJFINANCE.NS"
}



def stocks_data(ticker: str, start_date: str, end_date: str):
    stock = yf.Ticker(ticker)
    data_raw = stock.history(start=start_date, end=end_date)
    data_raw = data_raw.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
    data_raw['TP'] = (data_raw['High'] + data_raw['Low'] + data_raw['Close']) / 3

    data_raw['EMA_12'] = data_raw['Close'].ewm(span=12, adjust=False).mean()
    data_raw['EMA_26'] = data_raw['Close'].ewm(span=26, adjust=False).mean()
    data_raw['MACD'] = data_raw['EMA_12'] - data_raw['EMA_26']
    data_raw['Signal'] = data_raw['MACD'].ewm(span=9, adjust=False).mean()

    delta = data_raw['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    data_raw['RSI'] = 100 - (100 / (1 + rs))

    n = 14
    data_raw['SMA_TP'] = data_raw['TP'].rolling(n).mean()
    data_raw['MAD_TP'] = data_raw['TP'].rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    data_raw['CCI'] = (data_raw['TP'] - data_raw['SMA_TP']) / (0.015 * data_raw['MAD_TP'])

    data_raw['+DM'] = data_raw['High'].diff()
    data_raw['-DM'] = data_raw['Low'].diff()
    data_raw['+DM'] = np.where((data_raw['+DM'] > data_raw['-DM']) & (data_raw['+DM'] > 0), data_raw['+DM'], 0)
    data_raw['-DM'] = np.where((data_raw['-DM'] > data_raw['+DM']) & (data_raw['-DM'] > 0), data_raw['-DM'], 0)
    data_raw['TR'] = np.maximum(data_raw['High'] - data_raw['Low'],
                                np.maximum(abs(data_raw['High'] - data_raw['Close'].shift(1)),
                                           abs(data_raw['Low'] - data_raw['Close'].shift(1))))
    data_raw['ATR'] = data_raw['TR'].rolling(n).mean()
    data_raw['+DI'] = (data_raw['+DM'].rolling(n).mean() / data_raw['ATR']) * 100
    data_raw['-DI'] = (data_raw['-DM'].rolling(n).mean() / data_raw['ATR']) * 100
    data_raw['DX'] = (abs(data_raw['+DI'] - data_raw['-DI']) / (data_raw['+DI'] + data_raw['-DI'])) * 100
    data_raw['ADX'] = data_raw['DX'].rolling(n).mean()

    data_req = data_raw[['Open', 'High', 'Low', 'Close', 'Volume',
                         'EMA_12', 'EMA_26', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX']].bfill()

    return data_req.reset_index()



st.title("📊 Enhanced Stock Data Downloader (with Indicators)")

index = st.selectbox("Select Index", ["DJIA", "NSEI"])
filtered_stocks = {k: v for k, v in ALL_STOCKS.items() if k.startswith(f"{index}:")}
company = st.selectbox("Select Company", list(filtered_stocks.keys()))
ticker = filtered_stocks[company]
company_name = company.split(":", 1)[1]

data_path = os.path.join("C:/Users/RUTHVIK REDDY/StockSentinel/app/DB", index)
os.makedirs(data_path, exist_ok=True)  # Creates directory if it doesn't exist
csv_file = os.path.join(data_path, f"{ticker.replace('.', '_')}_data.csv")

if st.button("Load / Update Data"):
    with st.spinner("Calculating metrics and fetching data..."):
        today = datetime.now().date()
        start_date = (today - timedelta(days=365)).isoformat()
        end_date = today.isoformat()

        df = stocks_data(ticker, start_date, end_date)
        df.to_csv(csv_file, index=False)

        st.success(f"{company_name} data downloaded with indicators!")
        st.dataframe(df.tail())