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