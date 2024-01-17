from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dataset (replace with your own dataset)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Define the neural network model
mlp = MLPClassifier()

# Define the hyperparameters and their possible values
param_grid = {
    'hidden_layer_sizes': [(50,),(100,)],
    'activation': ['relu', 'tanh'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant', 'adaptive'],
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters
best_model = grid_search.best_estimator_
accuracyTuning = best_model.score(X_test, y_test)
print("Accuracy with Best Hyperparameters:", accuracyTuning)

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dataset (replace with your own dataset)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Define the neural network model with default parameters
mlp = MLPClassifier()

# Fit the model on the training data
mlp.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = mlp.score(X_test, y_test)
print("Accuracy on test data:", accuracy)
import csv
training_accuracy_tuning = accuracyTuning
training_accuracy = accuracy

"""
with open("MLPDataset1CSV.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([training_accuracy_tuning, training_accuracy])
"""