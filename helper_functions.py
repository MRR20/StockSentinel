"""All the additional functions are written in this Python file, which are used through out this project"""

## This function is used for getting stocks data from yfinance
import pandas as pd
import numpy as np
import yfinance as yf
import sys

def stocks_data(ticker: str, period: str):
    """
    Args:

    ticker (str) - Stock symbol of company
    period (str) - To return history period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

    """

    # to get all about "ticker"
    stock = yf.Ticker(ticker)

    # get history data of the stock (already in pandas DataFrame)
    data_raw = stock.history(period=period)

    # remove unncessary columns ['Dividends', 'Stock Splits']
    data_raw = data_raw.drop(['Dividends', 'Stock Splits'], axis=1)

    # calculating and defining other stock metrics [EMA_12, EMA_26, MACD, Signal, RSI, CCI, ADX]

    """
    Definations:

    EMA_12 - 12 days Exponential Moving Average
    EMA_26 - 26 days Exponential Moving Average
    MACD - Moving Average Convergence and Divergence (difference of EMA_12 & EMA_26)
    Signal - 9 days EMA of MACD
    RSI - Relative Strength Index (14 days calulation)
    CCI- Used to identify overbought and oversold conditions
    ADX - Measures trend strength based on DMI (Directional Movement Index)

    """

    # calculating TP for CCI
    data_raw['TP'] = (data_raw['High'] + data_raw['Low'] + data_raw['Close']) / 3

    # 12 day & 26 day EMA for MACD
    data_raw['EMA_12'] = data_raw['Close'].ewm(span=12, adjust=False).mean()
    data_raw['EMA_26'] = data_raw['Close'].ewm(span=26, adjust=False).mean()
    data_raw['MACD'] = data_raw['EMA_12'] - data_raw['EMA_26']
    data_raw['Signal'] = data_raw['MACD'].ewm(span=9, adjust=False).mean()

    # RSI calculation
    delta = data_raw['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain/loss
    data_raw['RSI'] = 100 - (100/(1 + rs))

    # CCI calculation
    n = 14
    data_raw['SMA_TP'] = data_raw['TP'].rolling(n).mean()
    data_raw['MAD_TP'] = data_raw['TP'].rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    data_raw['CCI'] = (data_raw['TP'] - data_raw['SMA_TP'])/(0.015 * data_raw['MAD_TP'])

    # ADX calculation
    data_raw['+DM'] = data_raw['High'].diff()
    data_raw['-DM'] = data_raw['Low'].diff()
    data_raw['+DM'] = np.where((data_raw['+DM'] > data_raw['-DM']) & (data_raw['+DM'] > 0), data_raw['+DM'], 0)
    data_raw['-DM'] = np.where((data_raw['-DM'] > data_raw['+DM']) & (data_raw['-DM'] > 0), data_raw['-DM'], 0)
    data_raw['TR'] = np.maximum(data_raw['High'] - data_raw['Low'],
                                np.maximum(abs(data_raw['High'] - data_raw['Close'].shift(1)),
                                           abs(data_raw['Low'] - data_raw['Close'].shift(1))))
    data_raw['ATR'] = data_raw['TR'].rolling(n).mean()
    data_raw['+DI'] = (data_raw['+DM'].rolling(n).mean()/data_raw['ATR']) * 100
    data_raw['-DI'] = (data_raw['-DM'].rolling(n).mean()/data_raw['ATR']) * 100
    data_raw['DX'] = (abs(data_raw['+DI'] - data_raw['-DI'])/(data_raw['+DI'] + data_raw['-DI'])) * 100
    data_raw['ADX'] = data_raw['DX'].rolling(n).mean()


    # getting required data [Open, High, Low, Close, Volume, EMA_12, EMA_26, MACD, Signal, RSI, CCI, ADX]
    data_req = data_raw[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_12', 'EMA_26', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX']]

    #print(f"Initial Length {len(data_req)}")

    # drop NaN rows as it could be better solution
    # data_req = data_req.dropna()

    # use bfill to save data
    data_req = data_req.bfill()

    #print(f"Present Length {len(data_req)}")

    return data_req


# for testing purpose

if __name__ == "__main__":
    tickers = dow_jones_tickers = [
            "MMM",  # 3M Company
            "AXP",  # American Express
            "AMGN", # Amgen
            "AAPL", # Apple
            "BA",   # Boeing
            "CAT",  # Caterpillar
            "CVX",  # Chevron
            "CSCO", # Cisco Systems
            "KO",   # Coca-Cola
            "DOW",  # Dow Inc.
            "GS",   # Goldman Sachs
            "HD",   # Home Depot
            "HON",  # Honeywell International
            "IBM",  # IBM
            "INTC", # Intel
            "JNJ",  # Johnson & Johnson
            "JPM",  # JPMorgan Chase
            "MCD",  # McDonald's
            "MRK",  # Merck & Co.
            "MSFT", # Microsoft
            "NKE",  # Nike
            "PG",   # Procter & Gamble
            "CRM", #salesforce
            "TRV", #Travelers Companies Inc.
            "UNH", #UnitedHealth Group
            "V",   # Visa
            "VZ",   # Verizon
            "WBA",  # Walgreens Boots Alliance
            "WMT",  # Walmart
            "DIS" #Disney
        ]
    period = "10y"

    for ticker in tickers:
        data = stocks_data(ticker, period)
        data.to_csv(f"./stocks_data/{ticker}_data.csv")




## This function is used for getting news data from Guardian API
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

def get_news_data(API_KEY: str, from_date: str, to_date:str):
    try:
      start = datetime.strptime(from_date, "%Y-%m-%d").date()
      end = datetime.strptime(to_date, "%Y-%m-%d").date()
      dates_generated = [start + timedelta(days=x) for x in range(0, (end-start).days+1)]
      # print("dates done")

      articles = []
      for days in tqdm(dates_generated):
          days = datetime.strftime(days, "%Y-%m-%d")

          BASE_URL = "https://content.guardianapis.com/search"
          params = {
              "api-key": API_KEY,
              "from-date": days,
              "to-date": days,
              "section": "business",
              # "sectionName": "business news",
              # "q": f"{dow_jones_companies[ticker]}",
              "show-fields": "headline, body",
              "order-by": "newest",
              "page-size": 100
          }

          response = requests.get(BASE_URL, params)
          if response is not None:
              data = response.json()

              for article in data["response"]["results"]:
                  articles.append({
                      "Title": article["webTitle"],
                      "URL": article["webUrl"],
                      "Publication Date": article["webPublicationDate"],
                  })
          else:
              pass

      df = pd.DataFrame(articles)
      df = df.set_index("Publication Date")

      return df

    except:
      df = pd.DataFrame(articles)
      df = df.set_index("Publication Date")
      print("Limit completed...")
      return df

news = get_news_data("API-KEY", "2023-04-07", "2025-02-27") # (api-key, from_date, to_date)



## This function is used to get sentiment scores of news data using FinBERT model
# Suggestion: Try to run with GPU and with higher RAM (use Colab)
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def calculate_compound_scores(probabilities):
    return probabilities['positive'] - probabilities['negative']

def get_news_sentiment(df):

    titles = list(df["Title"].values)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    inputs = tokenizer(titles, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits
    print("[Logits generated...]")
    print()

    probabilities = scipy.special.softmax(logits.numpy(), axis=1)
    labels = list(model.config.id2label.values())
    print("[Labels are listed...]")
    print()

    sentiment_scores = []
    for i, title in enumerate(titles):
        scores_dict = {labels[j]: probabilities[i][j] for j in range(len(labels))}
        sentiment_scores.append(scores_dict)
    print("[Sentiment scores are given...]")
    print()

    compound_scores = [calculate_compound_scores(scores) for scores in sentiment_scores]
    compound_scores = [float(x) for x in compound_scores]

    return compound_scores



## This function is used to combine different dataframes (stocks data & sentiment data)
import os
import pandas as pd

def merge_dataframes(directory: str):

    directory = os.fsencode(directory)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            # import Stock data and convert date format
            df1 = pd.read_csv("./stocks_data/" + filename)
            df1["Date"] = pd.to_datetime(df1["Date"], utc=True).dt.tz_convert(None).dt.date

            # import sentiment data and convert dateformat
            df2 = pd.read_csv("sentiment_avg.csv") # change "sentiment_avg.csv" with your sentimnet scores csv file
            df2["Publish Date"] = pd.to_datetime(df2["Publish Date"]).dt.date

            # merge dataframes
            df = pd.merge(df1, df2, left_on="Date", right_on="Publish Date")
            df = df.drop("Publish Date", axis=1)
            df.rename(columns={"Average": "Sentiment Average"}, inplace=True)

            # rewrite existing csv file with additional changes
            df.to_csv("./stocks_data/" + filename, index=False)

directory = "./stocks_data" # change it with your directory name
merge_dataframes(directory)



## Title