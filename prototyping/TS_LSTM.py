import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Define a function to plot training and test data for a given time period
def train_test_plot(dataset, tstart, tend):
    # Plot the high stock prices for the specified time period (training data)
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    
    # Plot the high stock prices for the time period beyond the specified end year (test data)
    dataset.loc[f"{tend+1}":, "High"].plot(figsize=(16, 4), legend=True)
    
    # Add legends and title to the plot
    plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1} and beyond)"])
    plt.title("IBIT stock price")
    
    # Display the plot
    plt.show()

def train_test_split(dataset, tstart, tend, columns=['High']):
    train = dataset.loc[f"{tstart}":f"{tend}", columns].values
    test = dataset.loc[f"{tend+1}":, columns].values
    
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

# Define a function to plot real and predicted values for time series forecasting
def plot_predictions(test, predicted, title):
    # Plot the real values in gray
    plt.plot(test, color="gray", label="Real")
    
    # Plot the predicted values in red
    plt.plot(predicted, color="red", label="Predicted")
    
    # Set the title and labels for the plot
    plt.title(f'{title}')
    plt.xlabel("Time")
    plt.ylabel(f'{title}')
    
    # Add a legend to differentiate real and predicted values
    plt.legend()
    
    # Show the plot
    plt.show()

# Define a function to calculate and print the root mean squared error (RMSE)
def return_rmse(test, predicted):
    # Calculate the RMSE using the mean_squared_error function from scikit-learn
    rmse = np.sqrt(mean_squared_error(test, predicted))
    
    # Print the RMSE
    print("The root mean squared error is {:.2f}.".format(rmse))