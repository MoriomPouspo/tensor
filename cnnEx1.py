import tensorflow as tf
import csv

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



# Normalize pixel values to be between 0 and 1
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

epoch = 2
batch_size = 2
history = model.fit(train_images, train_labels, epochs=epoch,batch_size=5,
                    validation_data=(test_images, test_labels))

# Extract accuracy values from the training history
training_accuracy = 0.0
validation_accuracy = 0.0
loss=100
validation_loss = 100

for i in range(epoch):
    training_accuracy = max(training_accuracy, history.history['accuracy'][0])
    validation_accuracy = max(validation_accuracy,history.history['val_accuracy'][0])
    loss = min(loss,history.history['loss'][0])
    validation_loss = min(validation_loss,history.history['val_loss'][0])



# Open the CSV file in append mode
with open("csvDemo.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([epoch, batch_size, training_accuracy, validation_accuracy, loss, validation_loss])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
