# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import StratifiedKFold
import csv

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the neural network model
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_layer1', min_value=32, max_value=512, step=32),
                    input_dim=4, activation='sigmoid'))
    model.add(Dense(units=hp.Int('units_layer2', min_value=16, max_value=256, step=16), activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize Keras Tuner RandomSearch
tuner = RandomSearch(build_model,
                     objective='val_accuracy',
                     max_trials=10,  # Adjust the number of trials as needed
                     directory='keras_tuner_dir',
                     project_name='iris_classification')

# Perform hyperparameter search
tuner.search(x_train, y_train,
             epochs=10,
             validation_split=0.2)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps) #extra

# Build the model with the best hyperparameters
model = build_model(best_hps)

# Train the model
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)

# Make predictions on the test set
y_pred_prob = model.predict(x_test)
y_pred = tf.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

test_accuracy = results
"""
# Open the CSV file in append mode
with open("MLPDataset2CSV.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['With Tuning:'])
    csv_writer.writerow([results])
"""