import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.dataset = data
        self.transform = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, mask_path = self.dataset[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        mask = Image.open(mask_path).convert("L")
        mask = self.transform(mask)
        return image, mask




class DataLoader():

    def __init__(self):
        self.transform =  transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def make_data(self):
        path = 'Datasets/Driving_Dataset'
        traindata, testdata = [], []
        for image, label  in zip(os.listdir(f'{path}/train_images'), os.listdir(f'{path}/train_masks')):
            traindata.append([f'{path}/train_images/{image}', f'{path}/train_masks/{label}'])

        for image, label  in zip(os.listdir(f'{path}/test_images'), os.listdir(f'{path}/test_masks')):
            testdata.append([f'{path}/test_images/{image}', f'{path}/test_masks/{label}'])
        return traindata, testdata

    def load_data(self, batch_size):
        traindata, testdata = self.make_data()
        train_dataset = Dataset(traindata, self.transform)
        test_dataset = Dataset(testdata, self.transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader



# class Dataset(torch.utils.data.Dataset):
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

#     def make_data(self):
#         path = 'Datasets/Driving_Dataset'
#         traindata = []
#         for image in os.listdir(f'{path}/train_images'):
#             traindata.append(f'{path}/train_images/{image}')
#         return traindata

#     def load_data(self, batch_size):
#         traindata = self.make_data()
#         train_dataset = Dataset(traindata, self.transform)
#         train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#         return train_dataloader








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
