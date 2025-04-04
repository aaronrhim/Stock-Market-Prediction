# config.py
import datetime

# Data configuration
DATA_SYMBOL = 'NVDA'
DATA_START = '2025-03-24'
DATA_END = '2025-03-29'
DATA_INTERVAL = '1m'
MARKET_OPEN = '09:30'
MARKET_CLOSE = '16:00'

# Model configuration
RNN_UNITS = 125
LSTM_UNITS = [125, 50, 50, 50]  # list for each LSTM layer
DROPOUT_RATE = 0.2
EPOCHS = 10
BATCH_SIZE = 32
N_STEPS = 1
FEATURES = 1

# File paths
MODEL_SAVE_PATH = 'model_lstm.keras'
SCALER_SAVE_PATH = 'scaler.pkl'

# Date parameters for training/testing
def get_weekly_dates():
    """Return last week and current week start/end datetime objects."""
    today = datetime.date.today()
    current_monday = today - datetime.timedelta(days=today.weekday())
    last_monday = current_monday - datetime.timedelta(days=7)
    last_friday = last_monday + datetime.timedelta(days=4)
    # Combine with market times
    train_start = datetime.datetime.combine(last_monday, datetime.time(9, 30))
    train_end   = datetime.datetime.combine(last_friday, datetime.time(16, 0))
    test_start  = datetime.datetime.combine(current_monday, datetime.time(9, 30))
    test_end    = datetime.datetime.combine(today, datetime.time(16, 0))
    return train_start, train_end, test_start, test_end
