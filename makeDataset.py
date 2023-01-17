import os
import gdown
import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np


file_id = "13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4"
output_file = "file.zip"
destination_directory = 'Datasets/VOS/train/'


gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)

os.makedirs(destination_directory)


# Extract the folder
with zipfile.ZipFile(output_file, "r") as zip_ref:
    zip_ref.extractall(destination_directory)



path = "Datasets/VOS/train/labels/"
savepath = "Datasets/VOS/train/masks/"
folders = os.listdir(path)

print(len(folders))

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