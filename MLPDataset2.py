# Import necessary libraries
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(4,)),  # Input shape for Iris dataset
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons for 3 classes
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# Evaluate the model on the test set
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)

#test_loss = x_test
test_accuracy = results

"""
# Open the CSV file in append mode
with open("MLPDataset2CSV.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Without Tuning:'])
    csv_writer.writerow([results])
"""