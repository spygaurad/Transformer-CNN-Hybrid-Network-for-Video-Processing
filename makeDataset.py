
import os
import gdown
import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil
import csv

# First we download the youtube VOS dataset from the Google Drive's link provided
def downloadDataset():
    print("Downloading the dataset from google drive...")
    file_id = "13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4"
    output_file = "file.zip"

    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
    print("Dataset Downloaded!")

    print("Extracting data...")
    destination_directory = 'Datasets/VOS/train'
    try:
        os.makedirs(destination_directory)
    except:
        print("Dataset Directory already exists.")
        pass
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(destination_directory)
    print("Dataset extracted and stored into a folder")



# Now we make our folder's structure, compatible with how our dataloader class does it.
def manageFolderStructure():
    #For images
    source_directory = 'Datasets/VOS/train/train/JPEGImages/'
    destination_directory = 'Datasets/VOS/train/images/'
    os.makedirs(destination_directory)
    subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
    print(f"Moving the files from {source_directory} to {destination_directory}")
    for subdirectory in tqdm(subdirectories):
        shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
    shutil.rmtree(source_directory)
    print("Images Moved")

    #For moving labels
    source_directory = 'Datasets/VOS/train/train/annotations/'
    destination_directory = 'Datasets/VOS/train/labels/'
    os.makedirs(destination_directory)
    print(f"Moving the files from {source_directory} to {destination_directory}")
    subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
    for subdirectory in tqdm(subdirectories):
        shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
    shutil.rmtree(source_directory)
    print("Labels moved.")


def createBinarySegmentationMask():
    # Now we just generate a segmentation mask.
    path = "Datasets/VOS/train/labels/"
    savepath = "Datasets/VOS/train/masks/"
    folders = os.listdir(path)
    for folder in tqdm(folders):
        files = os.listdir(f'{path}{folder}')
        os.makedirs(f'{savepath}/{folder}')
        for unitfile in files:
            filepath = f'{path}{folder}/{unitfile}'
            img = np.array(Image.open(filepath).convert("RGB"))
            old_color = np.array([236, 95, 103])
            new_color = np.array([255, 255, 255])
            mask = np.all(img == old_color, axis=-1)
            img[mask] = new_color
            img[~mask] = 0
            img = Image.fromarray(img, mode="RGB")
            img.save(f"{savepath}/{folder}/{unitfile}.png")



if __name__ == "__main__":
    checkPathToFolder = 'Dataset/VOS/train'
    if os.path.exists(checkPathToFolder) and os.path.isdir(checkPathToFolder):
        print('Dataset Exists!')
    else:
        print('Preparing dataset...')
        downloadDataset()
        extractDataset()
        manageFolderStructure()
        createBinarySegmentationMask()


# import os
# import csv

# # Directory to search for files
# dir_path = "Datasets/VOS/train/"

# # CSV file to write the file paths
# csv_path = "data_image_train_VOS.csv"

# # Recursively search for files in directory
# def search_for_files(directory):
#     # Create an empty list to store the file paths
#     file_paths = []

#     # Recursively search through all subdirectories
#     for root, directories, files in os.walk(directory):
#         for filename in files:
#             # Get the absolute path of the file and append it to the list
#             filepath = os.path.join(root, filename)
#             file_paths.append(filepath)

#     # Return the list of file paths
#     return file_paths

# # Call the function to get the file paths
# file_paths = search_for_files(dir_path)

# # Write the file paths to a CSV file
# with open(csv_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     for path in file_paths:
#         writer.writerow([path])
