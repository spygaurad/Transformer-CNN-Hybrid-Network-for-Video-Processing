import os
import csv

root_directory = '/home/optimus/Downloads/Dataset/BDD 100K/images/100k/'
csv_directory = '/home/optimus/wiseyak/Transformer-CNN-Hybrid-Network-for-Video-Processing/Datasets/image2image'

# Create a list to store the file paths
file_paths = []

# Iterate over the folders (train, test, val)
for folder in ['train', 'test', 'val']:
    folder_path = os.path.join(root_directory, folder)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_paths.append(file_path)

# Create and save the CSV files
for folder, csv_file_name in [('train', 'train.csv'), ('test', 'test.csv'), ('val', 'valid.csv')]:
    csv_file_path = os.path.join(csv_directory, csv_file_name)
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['path'])
        for file_path in file_paths:
            if folder in file_path:
                writer.writerow([file_path])
