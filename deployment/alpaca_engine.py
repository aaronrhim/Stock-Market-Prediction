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

# Load environment variables and Alpaca API credentials
load_dotenv()
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET")
BASE_URL = 'https://paper-api.alpaca.markets/v2'

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')

# Load your pre-trained LSTM model and scaler
model_lstm = load_model('../prototyping/model_lstm.keras')
with open('../prototyping/scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

# Define parameters for the LSTM model input
n_steps = 10
features = 1

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('server.html')

def current_price(symbol, dt):
    # Round current time to the nearest minute for Alpaca API
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    #specific_minute = now_dt.replace(second=0, microsecond=0)
    market_open = now_dt.replace(hour=6, minute=30, second=0, microsecond=0) # testing off market-hours
    start = market_open.isoformat()
    end = (market_open + datetime.timedelta(minutes=1)).isoformat()

    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, start, end, feed='iex').df
        if bars.empty:
            return None
        else:
            return bars.iloc[0]['close']
    except Exception as e:
        print("Error fetching stock price:", e)
        return None

# Global list to store recent price data for predictions
window_data = []

def run_model_prediction():
    # Ensure we have enough data points to form a valid input sequence
    if len(window_data) < n_steps:
        return None
    
    window = np.array(window_data[-n_steps:]).reshape(1, n_steps, features)
    # Scale input and predict
    window_scaled = sc.transform(window.reshape(-1, features)).reshape(1, n_steps, features)
    prediction_scaled = model_lstm.predict(window_scaled)
    predicted = sc.inverse_transform(prediction_scaled)[0][0]
    return predicted

def background_thread():
    while True:
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime('%H:%M:%S')
        actual_value = current_price("SMCI", now_dt)
        
        if actual_value is not None:
            window_data.append(actual_value)
            if len(window_data) > n_steps:
                window_data.pop(0)
        
        predicted_value = run_model_prediction()
        
        data = {
            'time': now_str,
            'actual': actual_value,
            'predicted': predicted_value
        }
        socketio.emit('update', data)
        time.sleep(5)  # Adjust the sleep interval as needed

# Start background thread for streaming data
threading.Thread(target=background_thread, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
