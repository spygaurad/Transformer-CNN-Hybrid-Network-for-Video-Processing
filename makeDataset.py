import os
import gdown
import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil
import csv

# First we download the youtube VOS dataset from the Google Drive's link provided
print("Downloading the dataset in zip format...")
file_id = "13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4"
output_file = "file.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
print("Dataset Downloaded!")


# Now we extract the downloaded file to a particular folder
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
source_directory = 'Datasets/VOS/train/train/JPEGImages/'
destination_directory = 'Datasets/VOS/train/images/'
os.makedirs(destination_directory)

subdirectories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
print(f"Moving the files from {source_directory} to {destination_directory}")
for subdirectory in tqdm(subdirectories):
    shutil.move(os.path.join(source_directory, subdirectory), destination_directory)
shutil.rmtree(source_directory)
print("Images Moved")