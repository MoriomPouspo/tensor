import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import numpy as np

tf.random.set_seed(7)

# Load and preprocess the dataset
dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split the dataset into train and test sets
train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Objective function for Optuna to optimize
def objective(trial):
    # Create and compile the LSTM network with hyperparameters as suggested by Optuna
    model = Sequential()
    model.add(LSTM(trial.suggest_int('lstm_units', 1, 10), input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Fit the model
    model.fit(trainX, trainY, epochs=trial.suggest_int('epochs', 5, 50), batch_size=trial.suggest_int('batch_size', 1, 32), verbose=0)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY_inv = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY_inv = scaler.inverse_transform([testY])

    # Calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:, 0]))
    testScore = np.sqrt(mean_squared_error(testY_inv[0], testPredict[:, 0]))

    return testScore  # Objective is to minimize test RMSE

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
best_lstm_units = best_params['lstm_units']
best_epochs = best_params['epochs']
best_batch_size = best_params['batch_size']

# Train the model with the best hyperparameters
model = Sequential()
model.add(LSTM(best_lstm_units, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=best_epochs, batch_size=best_batch_size, verbose=2)

# Make predictions and plot results
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY])

"""
# Plotting
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredict, label='Train Predictions')
plt.plot(testPredict, label='Test Predictions')
plt.legend()
plt.show()
"""
