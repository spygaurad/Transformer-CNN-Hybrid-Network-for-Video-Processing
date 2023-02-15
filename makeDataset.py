import os
import gdown
import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil
import csv

# '''First we download the youtube VOS dataset from the Google Drive's link provided'''
file_id = "13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4"
output_file = "file.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
