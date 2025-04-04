import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime

def train_test_plot(dataset, tstart, tend):
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    next_day = tend + datetime.timedelta(days=4)
    dataset.loc[f"{next_day}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {next_day})", f"Test ({next_day} and beyond)"])
    plt.title("NVDA stock price")
    plt.show()

def train_test_split(dataset, tstart, tend, columns=['High']):
    next_day = tend + datetime.timedelta(days=4)
    train = dataset.loc[f"{tstart}":f"{tend}", columns].values
    test = dataset.loc[f"{next_day}":, columns].values
    
    return train, test

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

def plot_predictions(test, predicted, title):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title(f'{title}')
    plt.xlabel("Time")
    plt.ylabel(f'{title}')
    
    plt.legend()
    plt.show()

def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))

    print("The root mean squared error is {:.2f}.".format(rmse))