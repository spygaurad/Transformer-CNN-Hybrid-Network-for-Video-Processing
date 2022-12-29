import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm
from segmentationUNet import UNet


class Inference():

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.__load_network()
        self.batch_size = 1


    def __load_network(self):
        if self.model_name == "focusnet":
            model = FCT()
            check = torch.load('saved_model/model_focusnet.tar')
            model.load_state_dict(check['model_state_dict'])
        else:
            model = UNet(in_channels=3, out_channels=1)
            check = torch.load('saved_model/endoscopy_model_unet.tar')
            model.load_state_dict(check['model_state_dict'])
        model = model.to(self.device)
        return model


    def __create_edge(self, mask, image):
        mask = mask.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1/4, 0))
        image = np.array(image)

        for i in range(np.array(mask).shape[0]):
            for j in range(np.array(mask).shape[1]):
                if np.array(mask)[i][j] == 255:
                    image[i][j] = [255, 255, 255] 
        image = transforms.ToPILImage()(image)
        return image




    def __view_sample(self, x, y, y_pred, seq):
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        y_pred = torch.squeeze(y_pred)
        x = transforms.ToPILImage()(x)
        y = transforms.ToPILImage()(y).convert('L')
        y_pred = transforms.ToPILImage()(y_pred)
        stacked = self.__create_edge(y_pred, x)
        images = [x, y, y_pred, stacked]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height), (0, 200, 200))
        
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0] + 4
        new_im = new_im.save(f'inference_samples/image_{seq}.jpg')



    def infer(self):
        images = os.listdir('Datasets/segmented-images/test_images')
        masks = os.listdir('Datasets/segmented-images/test_masks')
        trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        model = (self.network).to(self.device)

        for i, (image, mask) in tqdm(enumerate(zip(images, masks))):
            x = trans(Image.open(f'Datasets/segmented-images/test_images/{image}'))
            y = trans(Image.open(f'Datasets/segmented-images/test_masks/{mask}'))
            x, y = x[None, :], y[None, :]  
            x, y = x.to(self.device), y.to(self.device)

            y_pred = model(x)
            self.__view_sample(x, y, y_pred, i)
    
model = "unet"
seg = Inference(model)
seg.infer()