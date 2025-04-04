# data_utils.py
import yfinance as yf
import numpy as np
import pandas as pd
from config import DATA_SYMBOL, DATA_START, DATA_END, DATA_INTERVAL, MARKET_OPEN, MARKET_CLOSE

def download_data():
    """Download and filter minute-level stock data for the given symbol."""
    dataset = yf.download(DATA_SYMBOL, start=DATA_START, end=DATA_END, interval=DATA_INTERVAL)
    # Ensure dataset index is a DatetimeIndex, then filter by market hours
    dataset = dataset.between_time(MARKET_OPEN, MARKET_CLOSE)
    return dataset

def split_sequence(sequence, n_steps):
    """Split sequence into samples for time series prediction."""
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
