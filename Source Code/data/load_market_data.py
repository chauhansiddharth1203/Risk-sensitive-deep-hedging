import yfinance as yf
import torch

def load_stock_data(ticker="AAPL", period="1y"):
    data = yf.download(ticker, period=period)

    prices = data["Close"].values

    # normalize (important)
    prices = prices / prices[0]

    return torch.tensor(prices, dtype=torch.float32).squeeze()