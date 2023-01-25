import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import csv






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
        self.transform =  transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.batch_size = batch_size
        self.trainingType = trainingType
        self.return_train_and_test = return_train_and_test


    def make_data(self, csvfile):
        with open(csvfile, "r") as csv_file:
            datapaths = [row[:-1] for row in csv_file]
            return datapaths


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
            i = 0
            for row in reader:
                self.rows.append(row)
                i+=1
                if i>99:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, row):
        img_path = self.rows[row]
        for items in img_path:
            image = Image.open(items)
            image = self.transform(image)
            return image


class DataLoaderSequential():

    def __init__(self, batch_size):
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.batch_size = batch_size

    def load_data(self):
        dataset = CSVDataset("data_sequential_VOS.csv", self.transform)
        l1 = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return l1




# l1 = DataLoaderSequential(4).load_data()

# for items in l1:
#     print(items)
#     image = items[0]
#     image = image.numpy()
#     img = np.moveaxis(image, 0, 2)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow('img', img)
#     cv2.waitKey(500)





# class DataLoader():

#     def __init__(self):
#         self.transform =  transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

#     def get_filenames(self):
#         path = 'Datasets/Driving_Dataset_Mini'
#         traindata = []
#         dataset = os.listdir(f'{path}/train_images')
#         dataset = sorted(dataset, key=self.sort_key)
#         for image in dataset:
#             traindata.append(f'{path}/train_images/{image}')
#         return traindata

#     def load_data(self, batch_size):
#         traindata = self.get_filenames()
#         train_dataset = Dataset(traindata, self.transform)
#         train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#         return train_dataloader

#     def sort_key(self, filename):
#         base, ext = filename.rsplit('.', 1)
#         num = int(re.sub(r'\D', '', base))
#         return num
















# class TrainDataset(torch.utils.data.Dataset):

#     def __init__(self, data, transforms=None):
#         self.dataset = data
#         self.transform = transforms

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img_path = self.dataset[idx]
#         image = Image.open(img_path)
#         image = self.transform(image)
#         return image



# class DataLoader():

#     def __init__(self):
#         self.transform =  transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#         self.datasetWindow = deque()

#     def get_filenames(self):
#         path = 'Datasets/Driving_Dataset_Mini'
#         traindata = []
#         dataset = os.listdir(f'{path}/train_images')
#         dataset = sorted(dataset, key=self.sort_key)
#         for image in dataset:
#             traindata.append(f'{path}/train_images/{image}')
#         return traindata

#     def load_data(self, batch_size):
#         traindata = self.get_filenames()
#         train_dataset = Dataset(traindata, self.transform)
#         train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#         return train_dataloader

#     def sort_key(self, filename):
#         base, ext = filename.rsplit('.', 1)
#         num = int(re.sub(r'\D', '', base))
#         return num














# create = DataLoader()
# l1 = create.load_data(1)
# for items in l1:
#     for item in range(0, len(items), 2):
#         image= items[item][0]
#         mask = items[item+1][0]
#         image = np.concatenate((image, mask), axis=2)
#         img = np.moveaxis(image, 0, 2)
#         cv2.imshow('img', img)
#         cv2.waitKey(1)



# create = DataLoader(batch_size=1, trainingType="unsupervised", return_train_and_test=False)
# l1 = create.load_data()
# for items in l1:
#     image = items[0]
#     image = image.numpy()
#     img = np.moveaxis(image, 0, 2)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow('img', img)
#     cv2.waitKey(1)
