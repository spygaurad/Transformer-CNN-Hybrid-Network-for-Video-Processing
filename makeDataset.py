import os
import gdown
import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil


'''First we download the youtube VOS dataset from the Google Drive's link provided'''

# file_id = "13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4"
# output_file = "file.zip"
# gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)




'''Now we extract the downloaded file to the folder defined'''

# destination_directory = 'Datasets/VOS/train/'
# os.makedirs(destination_directory)
# with zipfile.ZipFile(output_file, "r") as zip_ref:
#     zip_ref.extractall(destination_directory)




'''Now we make the folder structure similar to what we are about to do below and for the further tasks that we're going to do in the future'''


# source_directory = 'Datasets/VOS/train/train/JPEGImages/'
# destination_directory = 'Datasets/VOS/train/images/'
# os.makedirs(destination_directory)
# subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
# print(f"Moving the files from {source_directory} to {destination_directory}")
# for subdirectory in tqdm(subdirectories):
#     shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
# shutil.rmtree(source_directory)
# print("Images Moved")


# source_directory = 'Datasets/VOS/train/train/annotations/'
# destination_directory = 'Datasets/VOS/train/labels/'
# os.makedirs(destination_directory)
# print(f"Moving the files from {source_directory} to {destination_directory}")
# subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
# for subdirectory in tqdm(subdirectories):
#     shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
# shutil.rmtree(source_directory)
# print("Labels moved.")

# shutil.rmtree('Datasets/VOS/train/train')




# '''Now we just generate a segmentation mask.'''

# path = "Datasets/VOS/train/labels/"
# savepath = "Datasets/VOS/train/masks/"
# folders = os.listdir(path)

# print(len(folders))

# for folder in tqdm(folders):
#     files = os.listdir(f'{path}{folder}')
#     os.makedirs(f'{savepath}/{folder}')
#     for unitfile in files:
#         filepath = f'{path}{folder}/{unitfile}'
#         img = np.array(Image.open(filepath).convert("RGB"))
#         old_color = np.array([236, 95, 103])
#         new_color = np.array([255, 255, 255])
#         mask = np.all(img == old_color, axis=-1)
#         img[mask] = new_color
#         img[~mask] = 0
#         img = Image.fromarray(img, mode="RGB")
#         img.save(f"{savepath}/{folder}/{unitfile}.png")


import os
import csv

# Specify the paths of the "test_images" and "test_masks" folders
image_path = "Datasets/Driving_Dataset/test_images/"
mask_path = "Datasets/Driving_Dataset/test_masks/"

# Get the lists of files in the "test_images" and "test_masks" folders
image_files = os.listdir(image_path)
mask_files = os.listdir(mask_path)

# Open the CSV file for writing
with open("data_test_car.csv", "w", newline='') as f:
    writer = csv.writer(f)


    # Iterate through the files and write the file paths to the CSV file
    for i in range(min(len(image_files), len(mask_files))):
        image_file = os.path.join(image_path, image_files[i])
        mask_file = os.path.join(mask_path, mask_files[i])
        writer.writerow([image_file, mask_file])


# Specify the paths of the "test_images" and "test_masks" folders
image_path = "Datasets/Driving_Dataset/train_images/"
mask_path = "Datasets/Driving_Dataset/train_masks/"

# Get the lists of files in the "test_images" and "test_masks" folders
image_files = os.listdir(image_path)
mask_files = os.listdir(mask_path)

# Open the CSV file for writing
with open("data_train_car.csv", "w", newline='') as f:
    writer = csv.writer(f)
    
    # Iterate through the files and write the file paths to the CSV file
    for i in range(min(len(image_files), len(mask_files))):
        image_file = os.path.join(image_path, image_files[i])
        mask_file = os.path.join(mask_path, mask_files[i])
        writer.writerow([image_file, mask_file])