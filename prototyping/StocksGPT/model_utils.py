# model_utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
import datetime

def build_rnn_model(n_steps, features, rnn_units):
    """Build and compile a simple RNN model."""
    model = Sequential()
    model.add(SimpleRNN(units=rnn_units, input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model

def build_lstm_model(n_steps, features, lstm_units, dropout_rate):
    """Build and compile an LSTM model with multiple layers."""
    model = Sequential()
    # First LSTM layer with return_sequences=True for stacking
    model.add(LSTM(units=lstm_units[0], return_sequences=True, input_shape=(n_steps, features)))
    model.add(Dropout(dropout_rate))
    # Additional LSTM layers
    for units in lstm_units[1:-1]:
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    # Final LSTM layer
    model.add(LSTM(units=lstm_units[-1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model

def plot_predictions(test, predicted, title):
    """Plot real vs. predicted values for a time series."""
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(title)
    plt.legend()
    plt.show()

def train_test_plot(dataset, train_start, train_end, test_start, test_end):
    """Plot training and testing data based on provided date ranges."""
    dataset.loc[f"{train_start}":f"{train_end}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{test_start}":f"{test_end}", "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train: {train_start.date()} to {train_end.date()}",
                f"Test: {test_start.date()} to {test_end.date()}"])
    plt.title("NVDA Stock Price")
    plt.show()

def return_rmse(test, predicted):
    """Calculate and print the RMSE of the predictions."""
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print(f"The root mean squared error is {rmse:.2f}.")
