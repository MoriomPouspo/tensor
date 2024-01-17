from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from kerastuner.tuners import RandomSearch

# Generate data
X_train = np.linspace(-10, 10, 1000)
y_train = np.sin(X_train) + np.random.normal(0, 0.2, size=X_train.shape)

X_test = np.linspace(-10, 10, 500)
y_test = np.sin(X_test) + np.random.normal(0, 0.2, size=X_test.shape)

# Reshape data for Keras model
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Define the Keras model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=32),
                           activation=hp.Choice('activation_1', values=['relu', 'tanh', 'sigmoid']),
                           input_dim=1))
    model.add(layers.Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32),
                           activation=hp.Choice('activation_2', values=['relu', 'tanh', 'sigmoid'])))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# Split the data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Instantiate the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Adjust as needed
    executions_per_trial=1,
    directory='my_tuner_dir'  # Set a directory for logs and checkpoints
)

# Perform the search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)

# Train the final model
final_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Make predictions using the final model
pred = final_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(pred, y_test)
print("Mean Squared Error:", mse)
