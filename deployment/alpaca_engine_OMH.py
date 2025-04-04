from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import datetime
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pytz

# Load environment variables and Alpaca API credentials
load_dotenv()
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET")
BASE_URL = 'https://paper-api.alpaca.markets/v2'

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('server.html')

def load_historical_data(symbol, start_time=None, end_time=None):
    """
    Load historical data for a symbol.
    If start_time and end_time are provided, use them;
    otherwise, use yesterday's market hours.
    """
    eastern = pytz.timezone('US/Eastern')
    if start_time is None or end_time is None:
        today = datetime.datetime.now(eastern).date()
        yesterday = today - datetime.timedelta(days=1)
        market_open = eastern.localize(datetime.datetime.combine(yesterday, datetime.time(9, 30)))
        market_close = eastern.localize(datetime.datetime.combine(yesterday, datetime.time(16, 0)))
    else:
        market_open = start_time
        market_close = end_time

    start = market_open.isoformat()
    end = market_close.isoformat()
    
    print(f"Loading historical data for {symbol} from {start} to {end}")
    
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, start, end, feed='iex').df
        return bars
    except Exception as e:
        print("Error fetching historical data:", e)
        return None

n_steps = 10
features = 1

window_data = []

# Load the pre-trained LSTM model and scaler
model_lstm = load_model('../prototyping/model_lstm.keras')
with open('../prototyping/scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run_model_prediction():
    if len(window_data) < n_steps:
        return None
    
    window = np.array(window_data[-n_steps:]).reshape(1, n_steps, features)
    window_scaled = sc.transform(window.reshape(-1, features)).reshape(1, n_steps, features)
    prediction_scaled = model_lstm.predict(window_scaled)
    print("Window data:", window_data[-n_steps:])
    print("Window scaled:", window_scaled)
    print("Scaled prediction:", prediction_scaled)

    predicted = sc.inverse_transform(prediction_scaled)[0][0]
    print("Inverse transformed prediction:", predicted)

    # DEBUGGING
    test_val = np.array([[104.0]])
    scaled_val = sc.transform(test_val)
    inv_val = sc.inverse_transform(scaled_val)
    print("DEBUGGING: Original:", test_val, "Scaled:", scaled_val, "Inverse Transformed:", inv_val)


    return float(predicted)

def update_model():
    """
    Fine-tune the current LSTM model using recent data.
    This function loads recent historical data, preprocesses it,
    and trains the model for a few epochs.
    """
    # For example, update with data from yesterday (you can modify this range as needed)
    eastern = pytz.timezone('US/Eastern')
    today = datetime.datetime.now(eastern).date()
    yesterday = today - datetime.timedelta(days=1)
    update_open = eastern.localize(datetime.datetime.combine(yesterday, datetime.time(9, 30)))
    update_close = eastern.localize(datetime.datetime.combine(yesterday, datetime.time(16, 0)))
    
    new_data = load_historical_data("NVDA", start_time=update_open, end_time=update_close)
    if new_data is None or new_data.empty:
        print("No new data available for updating.")
        return

    # Assume we use the 'close' prices for updating (or change to the relevant column)
    new_values = new_data['close'].values.reshape(-1, 1)
    new_values_scaled = sc.transform(new_values)
    
    # Create sequences from the new data
    X_new, y_new = split_sequence(new_values_scaled, n_steps)
    if X_new.size == 0:
        print("Not enough new data to create sequences.")
        return
    X_new = X_new.reshape(X_new.shape[0], n_steps, features)
    
    # Fine-tune the model with the new data
    print("Fine-tuning model on new data...")
    model_lstm.fit(X_new, y_new, epochs=5, batch_size=32, verbose=1)
    print("Model update complete.")

def background_thread():
    """
    Background thread to simulate streaming predictions.
    It loads historical data and iterates over it, predicting values and sending them to the frontend.
    It also periodically updates the model with recent data.
    """
    symbol = "NVDA"
    bars = load_historical_data(symbol)
    if bars is None or bars.empty:
        print("No historical data available.")
        return

    update_interval = 60 * 10  # update model every 30 minutes, for example
    last_update = time.time()

    for index, row in bars.iterrows():
        actual_value = row['close']
        window_data.append(actual_value)
        if len(window_data) > n_steps:
            window_data.pop(0)

        predicted_value = run_model_prediction() if len(window_data) >= n_steps else None

        data = {
            'time': index.strftime('%H:%M:%S'),
            'actual': actual_value,
            'predicted': predicted_value
        }
        print("Sending data:", data)

        socketio.emit('update', data)
        time.sleep(1)  # simulate 5-second interval between updates

        # Check if it's time to update the model with new data
        if time.time() - last_update > update_interval:
            update_model()
            last_update = time.time()

# Start background thread for streaming predictions and model updating
socketio.start_background_task(background_thread)

if __name__ == '__main__':
    socketio.run(app, debug=True)
