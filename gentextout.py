import tensorflow as tf
from keras import datasets, layers, models
from keras_tuner import RandomSearch
#from sklearn.model_selection import RandomizedSearchCV

import tensorflow as tf
import csv
import matplotlib.pyplot as plt

def write_to_txt(history, filename):
    #Printing to external txt file
    summary = history.history
    summary = str(summary)
    with open(filename, 'a') as f:
        f.writelines(summary)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1, 
                    validation_data=(test_images, test_labels))

#Write to txt
write_to_txt(history=history, filename='textout.txt')

# ... (Data loading and preprocessing as in your original code)

def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('filters_1', min_value=32, max_value=128, step=32),
                            (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('filters_2', min_value=64, max_value=256, step=64),
                            (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('filters_3', min_value=64, max_value=256, step=64),
                            (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,  # Adjust the number of trials as needed
    executions_per_trial=2,  # Multiple executions for more reliable results
    directory='my_dir',  # Optional directory to store results
    project_name='cifar10_tuning')

tuner.search(train_images, train_labels,
             epochs=2,  # Increase epochs for more thorough tuning
             validation_data=(test_images, test_labels))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model with more epochs
history = best_model.fit(train_images, train_labels, epochs=2, validation_data=(test_images, test_labels))

#Printing to external txt file
write_to_txt(history=history, filename='textout.txt')