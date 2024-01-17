import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import BaseEstimator
import csv

# Load dataset
df = pd.read_csv('monthly_milk_production.csv', index_col='Date', parse_dates=True)
df.index.freq = 'MS'

# Split the data into training and testing
train = df.iloc[:156]
test = df.iloc[156:]

# Scaling our data
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Convert data to PyTorch tensors
X_train = torch.tensor(scaled_train.reshape(-1, 1, 1), dtype=torch.float32)
y_train = torch.tensor(scaled_train.reshape(-1, 1), dtype=torch.float32)

# Function to create the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Wrapper class for PyTorch model
class PyTorchEstimator(BaseEstimator):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1, lr=0.001, epochs=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.model = SimpleLSTM(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        return self.model(X).detach().numpy()

    def get_params(self, deep=True):
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'lr': self.lr,
            'epochs': self.epochs
        }

# Use TimeSeriesSplit for cross-validation on time series data
tscv = TimeSeriesSplit(n_splits=3)

# Set the parameters to search
param_grid = {
    'input_size': [1],
    'hidden_size': [50, 100, 150],
    'num_layers': [1, 2],
    'lr': [0.001, 0.01, 0.1],
    'epochs': [5, 10, 15]
}

# Create an instance of the PyTorchEstimator
pytorch_estimator = PyTorchEstimator()

# Use GridSearchCV for hyperparameter tuning
grid = GridSearchCV(estimator=pytorch_estimator, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, refit=False)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and corresponding score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


training_accuracy = grid_result.best_score_*(-1)

"""
# Open the CSV file in append mode
with open("LSTMDataset2CSV.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([training_accuracy])
"""