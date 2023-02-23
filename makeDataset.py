
# import os
# import gdown
# import zipfile
# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# import shutil
# import csv

# # First we download the youtube VOS dataset from the Google Drive's link provided
# def downloadDataset():
#     print("Downloading the dataset from google drive...")
#     file_id = "13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4"
#     output_file = "file.zip"

#     gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
#     print("Dataset Downloaded!")

#     print("Extracting data...")
#     destination_directory = 'Datasets/VOS/train'
#     try:
#         os.makedirs(destination_directory)
#     except:
#         print("Dataset Directory already exists.")
#         pass
#     with zipfile.ZipFile(output_file, "r") as zip_ref:
#         zip_ref.extractall(destination_directory)
#     print("Dataset extracted and stored into a folder")



# # Now we make our folder's structure, compatible with how our dataloader class does it.
# def manageFolderStructure():
#     #For images
#     source_directory = 'Datasets/VOS/train/train/JPEGImages/'
#     destination_directory = 'Datasets/VOS/train/images/'
#     os.makedirs(destination_directory)
#     subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
#     print(f"Moving the files from {source_directory} to {destination_directory}")
#     for subdirectory in tqdm(subdirectories):
#         shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
#     shutil.rmtree(source_directory)
#     print("Images Moved")

#     #For moving labels
#     source_directory = 'Datasets/VOS/train/train/annotations/'
#     destination_directory = 'Datasets/VOS/train/labels/'
#     os.makedirs(destination_directory)
#     print(f"Moving the files from {source_directory} to {destination_directory}")
#     subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
#     for subdirectory in tqdm(subdirectories):
#         shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
#     shutil.rmtree(source_directory)
#     print("Labels moved.")


# def createBinarySegmentationMask():
#     # Now we just generate a segmentation mask.
#     path = "Datasets/VOS/train/labels/"
#     savepath = "Datasets/VOS/train/masks/"
#     folders = os.listdir(path)
#     for folder in tqdm(folders):
#         files = os.listdir(f'{path}{folder}')
#         os.makedirs(f'{savepath}/{folder}')
#         for unitfile in files:
#             filepath = f'{path}{folder}/{unitfile}'
#             img = np.array(Image.open(filepath).convert("RGB"))
#             old_color = np.array([236, 95, 103])
#             new_color = np.array([255, 255, 255])
#             mask = np.all(img == old_color, axis=-1)
#             img[mask] = new_color
#             img[~mask] = 0
#             img = Image.fromarray(img, mode="RGB")
#             img.save(f"{savepath}/{folder}/{unitfile}.png")



# if __name__ == "__main__":
#     checkPathToFolder = 'Dataset/VOS/train'
#     if os.path.exists(checkPathToFolder) and os.path.isdir(checkPathToFolder):
#         print('Dataset Exists!')
#     else:
#         print('Preparing dataset...')
#         downloadDataset()
#         extractDataset()
#         manageFolderStructure()
#         createBinarySegmentationMask()


# import os
# import csv

# # Define the path to the directory containing the subdirectories and files
# path = "Datasets/VOS/train/images/"

# # Create a list of the subdirectories
# subdirs = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# # Create a list to store the file paths
# file_paths = []

# # Loop through each subdirectory
# for subdir in subdirs:
#     # Loop through each file in the subdirectory
#     for file in os.listdir(subdir):
#         # Get the full path to the file
#         file_path = os.path.join(subdir, file)
#         # Add the file path to the list
#         file_paths.append(file_path)

# # Write the file paths to a CSV file
# with open('data_image_train_VOS.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for file_path in file_paths:
#         writer.writerow([file_path])


import os
import csv

# Specify the path of the "Dataset" folder
dataset_path = "Datasets/VOS/train/images"

# Open the CSV file for writing
with open("data_sequential_VOS.csv", "w") as f:
    writer = csv.writer(f)

    # Iterate through all the folders in the "Dataset" folder
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        # Ignore if its not a directory
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        files.sort()
        # Iterate through all the files in the folder
        for i in range(len(files)):
            window_files = files[i:i+5]
            file_paths = [os.path.join(folder_path, x) for x in window_files]
            writer.writerow(file_paths)



input_file = 'data_sequential_VOS.csv'
output_file = 'sequential_VOS_data.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    for row in reader:
        if len(row) >= 5:
            writer.writerow(row)
