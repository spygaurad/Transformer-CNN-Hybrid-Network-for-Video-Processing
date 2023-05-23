import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import random
import os 
import numpy as np
from scipy.ndimage import sobel

import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# from dataset import DataLoader
from metric import DiceLoss, JaccardScore

from PIL import Image
from torchvision import transforms

from tensorboardX import SummaryWriter 



    
class EncoderBlock(nn.Module):
    def __init__(self, blk, in_channels, out_channels):
        super().__init__()
        self.blk = blk
        self.conv1_a = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv1_b = nn.Conv2d(3, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) 

    def forward(self, x, scale_img="none"):
        if ((self.blk=="first") or (self.blk=="bottleneck")):
            x1 = self.relu(self.conv1_a(x))
            x1 = self.relu(self.conv2(x1))
        else:
            skip_x = self.relu(self.conv1_b(scale_img))
            x1 = torch.cat([skip_x, x], dim=1)
            x1 = self.relu(self.conv2(x1))
            x1 = self.relu(self.conv3(x1))
        out = self.maxpool(self.dropout(x1))
        return out




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        out = self.dropout(x1)
        return out




class DeepSupervisionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        out = self.sigmoid(self.conv3(x1))
        return out




class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [8, 16, 32, 64, 128, 512] 
        self.drp_out = 0.3
        self.scale_img = nn.AvgPool2d(2, 2)   

        self.block_1 = EncoderBlock("first", 3, filters[0])
        self.block_2 = EncoderBlock("second", filters[0], filters[1])
        self.block_3 = EncoderBlock("third", filters[1], filters[2])
        self.block_4 = EncoderBlock("fourth", filters[2], filters[3])
        self.block_5 = EncoderBlock("fifth", filters[3], filters[4])
        self.block_6 = EncoderBlock("bottleneck", filters[4], filters[5])


    def forward(self, x):
        # Multi-scale input
        scale_img_2 = self.scale_img(x)
        scale_img_3 = self.scale_img(scale_img_2)
        scale_img_4 = self.scale_img(scale_img_3)  
        scale_img_5 = self.scale_img(scale_img_4)

        x1 = self.block_1(x)
        x2 = self.block_2(x1, scale_img_2)
        x3 = self.block_3(x2, scale_img_3)
        x4 = self.block_4(x3, scale_img_4)
        x5 = self.block_5(x4, scale_img_5)
        x6 = self.block_6(x5)
        return x6



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [512, 128, 64, 32, 16, 8]
        self.drp_out = 0.3

        self.block_5 = DecoderBlock(filters[0], filters[1])
        self.block_4 = DecoderBlock(filters[1], filters[2])
        self.block_3 = DecoderBlock(filters[2], filters[3])
        self.block_2 = DecoderBlock(filters[3], filters[4])
        self.block_1 = DecoderBlock(filters[4], filters[5])
        self.ds = DeepSupervisionBlock(filters[5], 3)
        
    def forward(self, x):
        x = self.block_5(x)
        x = self.block_4(x)
        x = self.block_3(x)
        x = self.block_2(x)
        x = self.block_1(x)
        out9 = self.ds(x)
        return out9



class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return latent, output



data = (torch.rand(size=(1, 3, 256, 256)))
AE = AutoEncoder()
out = AE(data)
print(out[0].shape)
print(out[1].shape)










# class FCT_FLOW():

#     def __init__(self) -> None:
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         # self.device =   "cpu"
    

#     def save_sample(self, epoch, x, y, y_pred):
#         path = f'Training_Sneakpeeks/FCT'
#         try:
#             os.makedirs(path)
#         except:
#             pass
#         elements = [x, y, y_pred]
#         elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
#         for i, element in enumerate(elements):
#             element.save(f"{path}/{epoch}_{['input','actual','predicted'][i]}.jpg")



#     def train(self, batch_size, epochs, lr=0.001):
        
        
#         print("Loading Datasets...")
#         dl = DataLoader(batch_size=batch_size )
#         train_data, test_data = dl.load_data("car_train_data.csv", "car_test_data.csv")
#         print("Dataset Loaded... initializing parameters...")
        
        
#         model = FCT()
#         model.to(self.device)

#         optimizer = optim.AdamW(model.parameters(), lr)
#         dsc_loss = DiceLoss() 
#         # iou = JaccardScore()

#         writer = SummaryWriter(log_dir="logs")        
        
#         loss_train, loss_test, measur = [], [], []
#         start = 1
#         epochs = epochs+1
        
#         print(f"Starting to train for {epochs} epochs.")

#         for epoch in range(start, epochs):

#             _loss_train, _loss_test, _measure = 0, 0, 0
#             print(f"Training... at Epoch no: {epoch}")

#             num = random.randint(0, (len(train_data)//batch_size) - 1)

#             for i, (x, y) in enumerate(tqdm(train_data)):

#                 x, y = x.to(self.device), y.to(self.device)

#                 optimizer.zero_grad()

#                 y_pred = model(x)

#                 #taking the loss 
#                 loss = dsc_loss(y_pred, y)
#                 _loss_train += loss.item()

#                 #backprop algorithm
#                 loss.backward()
#                 optimizer.step()
#                 if i == num:
#                     self.save_sample(epoch, x, y, y_pred)

#             # writer.add_scalar("Testing Loss", _loss_test, epoch)
#             writer.add_scalar("Training Loss", _loss_train, epoch)
#             # writer.add_scalar("Evaluation Metric", _measure, epoch)
            
#             loss_train.append(_loss_train)
#             # loss_test.append(_loss_test)
#             # measur.append(_measure)

#             # print(f"Epoch: {epoch+1}, Training loss: {_loss_train}, Testing Loss: {_loss_test} || Jaccard Score : {_measure}")
#             print(f"Epoch: {epoch+1}, Training loss: {_loss_train}")
 
#             if loss_train[-1] == min(loss_train):
#                 print('Saving Model...')
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': loss_train
#                 }, f'saved_model/FCT_for_cars.tar')
#             print('\nProceeding to the next epoch...')



#     def infer(self):

#         model = self.network
#         model.load_state_dict(torch.load("saved_model/FCT_for_cars.tar")['model_state_dict'])
#         model = model.to(self.device)

#         path_for_image_inference = 'Datasets/Driving_Dataset/inference'
#         path_for_saving_inference_samples = "Inference_For_Cars/generated_images"

#         try:
#             os.makedirs(path_for_saving_inference_samples)
#         except:
#             pass

#         file_paths = [ f'{path_for_image_inference}/{x}' for x in os.listdir(path_for_image_inference)]
#         print(file_paths)
#         for i, image in tqdm(enumerate(file_paths)):
#             img = Image.open(image)
#             out = model(img)
#             out = np.array(out)
#             sobel_x = sobel(out, axis=0)
#             sobel_y = sobel(out, axis=1)
#             sobel_img = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
#             sobel_img = (sobel_img / np.max(sobel_img)) * 255
#             sobel_img = sobel_img.astype(np.uint8)
#             out = Image.fromarray(sobel_img)
#             img, out = img.convert("RGB"), out.convert("RGB")
#             result = Image.concatenate(img, out, axis=1)
#             result.save(f"{path_for_saving_inference_samples}/image_{i}")

    

# seg = FCT_FLOW()
# seg.train(batch_size=1, epochs=70) 
# seg.infer()