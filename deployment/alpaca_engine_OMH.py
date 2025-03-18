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

def load_historical_data(symbol):
    eastern = pytz.timezone('US/Eastern')
    today = datetime.datetime.now(eastern).date()
    market_open = eastern.localize(datetime.datetime.combine(today, datetime.time(9, 30)))
    market_close = eastern.localize(datetime.datetime.combine(today, datetime.time(16, 0)))
    
    market_open_utc = market_open.astimezone(datetime.timezone.utc)
    market_close_utc = market_close.astimezone(datetime.timezone.utc)
    
    start = market_open_utc.isoformat()
    end = market_close_utc.isoformat()
    
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

def run_model_prediction():
    if len(window_data) < n_steps:
        return None
    
    window = np.array(window_data[-n_steps:]).reshape(1, n_steps, features)
    window_scaled = sc.transform(window.reshape(-1, features)).reshape(1, n_steps, features)
    prediction_scaled = model_lstm.predict(window_scaled)
    predicted = sc.inverse_transform(prediction_scaled)[0][0]
    return float(predicted)

def background_thread():
    # Instead of fetching current price, load historical market data
    symbol = "IBIT"
    bars = load_historical_data(symbol)
    if bars is None or bars.empty:
        print("No historical data available.")
        return

    # Iterate over the historical data row by row
    for index, row in bars.iterrows():
        actual_value = row['close']
        # Update the global window_data list
        window_data.append(actual_value)
        if len(window_data) > n_steps:
            window_data.pop(0)

        # Only predict once we have enough data points
        predicted_value = run_model_prediction() if len(window_data) >= n_steps else None

        data = {
            'time': index.strftime('%H:%M:%S'),
            'actual': actual_value,
            'predicted': predicted_value
        }
        socketio.emit('update', data)
        time.sleep(5)  # Adjust the update interval as needed

# Start background thread for streaming data
#threading.Thread(target=background_thread, daemon=True).start()
socketio.start_background_task(background_thread)

if __name__ == '__main__':
    socketio.run(app, debug=True)
