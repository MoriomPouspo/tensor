from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_train = np.linspace(-10, 10, 1000)
y_train = np.sin(X_train) + np.random.normal(0, 0.2, size=X_train.shape)

X_test =  np.linspace(-10, 10, 500)
y_test =  np.sin(X_test) + np.random.normal(0, 0.2, size=X_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(62, activation='relu', input_dim=1))
model.add(tf.keras.layers.Dense(62, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100)

pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

pred = model.predict(X_test)

print("mean squared error: ",mean_squared_error(pred, y_test))