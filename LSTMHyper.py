
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pandas
import matplotlib.pyplot as plt


def create_model(num_neurons=4, look_back=1):
    model = Sequential()
    model.add(LSTM(num_neurons, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

#DataSet Graph
dataset = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
plt.plot(dataset)
plt.show()
tf.random.set_seed(7)
dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# print(dataset)

# convert an array of values into a dataset matrix
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# print(trainX)
# print("kas")
print(trainY)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(trainPredict.shape)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
print(trainPredict.shape)
trainY = scaler.inverse_transform([trainY])
trainY = trainY.transpose()
print(trainY.shape)
print(trainX.shape)
# print(trainX)
# print('BBjkashfk')
# print(trainY)

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# Reshape testY to match the shape of testPredict
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


model = KerasRegressor(build_fn=create_model, look_back=look_back, num_neurons = 2)

# Define hyperparameter grid
param_grid = {
    'epochs': [5, 5, 5],
    'batch_size': [1, 4, 8],
    'num_neurons': [2, 4, 8],
    # 'look_back': [1, 2, 3]
}

# Perform grid search cross-validation
trainX = trainX.reshape(trainX.shape[:-1])
# trainX = trainX.transpose()
print(trainX.shape)
# trainY.transpose()
# print(trainY.shape)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
# Perform grid search cross-validation
if len(trainX) < 5:  # Check if there are enough samples for 5-fold cross-validation
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=len(trainX))
else:
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

grid_result = grid.fit(trainX, trainY)

# grid_result = grid.fit(trainX, trainY)

# Retrieve best hyperparameters and model
best_params = grid_result.best_params_
best_model = grid_result.best_estimator_

# Evaluate best model on test set

testX = testX.reshape(testX.shape[:-1])
testPredict = best_model.predict(testX)
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score with best params: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
