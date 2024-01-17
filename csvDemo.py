import csv

# Sample data
data = [
    ['Name', 'Age', 'City'],
    ['John Doe', 25, 'New York'],
    ['Jane Smith', 30, 'San Francisco'],
    ['Bob Johnson', 22, 'Los Angeles']
]

# Specify the file name
output_file = 'csvDemo.csv'

# Writing data to the CSV file
with open(output_file, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Writing the header
    csv_writer.writerow(data[0])

    # Writing the remaining rows
    csv_writer.writerows(data[1:])

print(f'Data has been written to {output_file}')

"""
# Extract accuracy values from the training history
epoch = 2
training_accuracy = history.history['accuracy'][0]
validation_accuracy = history.history['val_accuracy'][0]

# Open the CSV file in append mode
with open("csvCNN.csv", mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([epoch, training_accuracy, validation_accuracy])
"""