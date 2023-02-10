import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import csv
import random


DATA_SIZE = 4


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, trainingType, transforms=None):
        self.data = data
        self.transform = transforms
        self.trainingType = trainingType

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.trainingType.lower() == "supervised":
            img_path, mask_path = self.data[idx].split(",")[0], self.data[idx].split(",")[1]
            image = Image.open(img_path)
            image = self.transform(image)
            mask = Image.open(mask_path).convert("L")
            mask = self.transform(mask)
            return image, mask
        else:
            img_path = self.data[idx]
            image = Image.open(img_path)
            image = self.transform(image)
            return image
        
        

class DataLoader():

    def __init__(self, batch_size, trainingType, return_train_and_test):
        self.transform =  transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        self.batch_size = batch_size
        self.trainingType = trainingType
        self.return_train_and_test = return_train_and_test


    def make_data(self, csvfile):
        with open(csvfile, "r") as csv_file:
            datapaths = [row[:-1] for row in csv_file]
            random.shuffle(datapaths)
            return datapaths[0:len(datapaths)//DATA_SIZE]
            # return datapaths


    def load_data(self, trainDataCSV, testDataCSV):
        if self.return_train_and_test:
            assert testDataCSV != None , "Please enter the path to csv file for the TEST dataset too."

        traindata_paths = self.make_data(trainDataCSV)
        train_dataset = Dataset(traindata_paths, self.trainingType, self.transform)
        trainLoadedData = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        if self.return_train_and_test:
            testdata_paths = self.make_data(testDataCSV)
            test_dataset = Dataset(testdata_paths, self.trainingType, self.transform)
            testLoadedData = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            return trainLoadedData, testLoadedData
        else:
            return trainLoadedData




'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''




class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transforms):
        self.transform = transforms
        self.rows = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader)
            i = 0
            for row in reader:
                self.rows.append(row)
                if i > ((row_count//DATA_SIZE) + 1):
                    break
                i+=1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, row):
        img_paths = self.rows[row]
        images = []
        for items in img_paths:
            image = Image.open(items)
            image = self.transform(image)
            images.append(image)
        return images


class DataloaderSequential():
    
    def __init__(self, csv_file, batch_size) -> None:
        self.csv_file = csv_file
        self.transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        self.batch_size = batch_size
    
    def load_images(self):
        dataset = CSVDataset(self.csv_file, self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader



# l1 = DataloaderSequential("data_sequential_VOS.csv", 4).load_images()
# print(len(l1))
# for images in l1:
#     for image in images:
#         img = [np.moveaxis(image[i].numpy(), 0, 2) for i in range(4)]
#         img = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img]
#         vertical_stack = cv2.hconcat(img)
#         cv2.imshow('img', vertical_stack)
#         cv2.waitKey(500)
#     print("sequence_break")



