import csv

# Sample data
data = [
    ['Name'],
    ['Moriom']
]

file_name = 'Pouspo.csv'

with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print(f'Data has been inserted to {file_name}')
