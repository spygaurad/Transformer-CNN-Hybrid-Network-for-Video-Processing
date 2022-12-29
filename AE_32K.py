import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np 
import torch.nn as nn
from dataset import DataLoader
import math
from tqdm import tqdm
from segmentationUNet import UNet
from pytorch_msssim import ms_ssim
from metric import DiceLoss, MixedLoss
from PIL import Image
import random



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out




class Encoder_32K(nn.Module):

    def __init__(self, block, layers):

        self.inplanes = 64

        super(Encoder_32K, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.conv2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 16, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(8)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.3)

        self.scale_img = nn.AvgPool2d(2, 2)





    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.dropout(self.conv5(x))))
        
        return x





class Decoder_32K(nn.Module):
    def __init__(self, outputDeterminer):
        super(Decoder_32K, self).__init__()

        self.outputDeterminer = outputDeterminer

        self.conv1 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)

        self.transConv1 = nn.ConvTranspose2d(256, 384, 2, 2, padding = 0)
        self.dbn2 = nn.BatchNorm2d(384)

        self.transConv2 = nn.ConvTranspose2d(384, 192, 2, 2, padding = 0)
        self.dbn3 = nn.BatchNorm2d(192)

        self.conv5 = nn.Conv2d(192, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 8, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(8)

        if self.outputDeterminer.lower() == "image":
            self.outputDeterminerConv = nn.Conv2d(8, 3, 3, padding=1)
            self.outputDeterminerNorm = nn.BatchNorm2d(3)
            self.finalactivation = nn.ReLU()
        elif self.outputDeterminer.lower() == "mask":
            self.outputDeterminerConv = nn.Conv2d(8, 1, 3, padding=1)
            self.outputDeterminerNorm = nn.BatchNorm2d(1)
            self.finalactivation = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))))))))
        x = self.relu(self.dbn2(self.transConv1(x)))
        x = self.relu(self.dbn3(self.transConv2(x)))
        x = self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(x))))))
        x = self.finalactivation(self.outputDeterminerNorm(self.outputDeterminerConv(x)))
        return x





class Autoencoder32K(nn.Module):
    def __init__(self, outputType):
        super(Autoencoder32K, self).__init__()
        self.encoder = Encoder_32K(Bottleneck, [3, 4])
        if outputType.lower() == "image":
            self.decoder = Decoder_32K("image")
        else:
            self.decoder = Decoder_32K("mask")

    def forward(self, x):
        bottleneck_32K = self.encoder(x)
        decoded = self.decoder(bottleneck_32K) 
        return decoded







def save_sample(epoch=0, x=None, mask=None, mask_pred=None, mode='train'):
    path = f'Training Sneakpeeks/latent_to_mask/{epoch}'
    elements = [x, mask, mask_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]

    if mode == 'train':
        elements[0] = elements[0].save(f"{path}_image.jpg")
        elements[1] = elements[1].save(f"{path}_mask.jpg")
        elements[2] = elements[2].save(f"{path}_mask_pred.jpg")
    elif mode == 'test':
        x = elements[0]
        x_hat = elements[1]
        images = [x, x_hat]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height), (0, 200, 200))
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0] + 4
        new_im.show()






def train(epochs, batch_size=4, lr=0.001):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    print("Loading Datasets...")
    train_dataloader, test_dataloader = DataLoader().load_data(batch_size)
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    #loading the model
    AE_model = Autoencoder32K("image").to("cuda:0")
    AE_model.load_state_dict(torch.load('saved_model/EncoderDecoder_road_image2image32K.tar')['model_state_dict'])

    #extracting the encoder from the autoencoder
    encoder = AE_model.encoder
    for params in encoder.parameters():
        params.requires_grad = False

    #using a new CNN decoder to train the latent
    decoder = Decoder_32K("mask")

    #a model comprised of forzen encoder and the decoder
    model = nn.Sequential(encoder, decoder).to("cuda:0")
    model.load_state_dict(torch.load('saved_model/latent_to_mask.tar')['model_state_dict'])



    #initializing the optimizer
    optimizerImg = optim.AdamW(model.parameters(), lr)
    # mseloss = torch.nn.MSELoss()

    nvidia_mix_loss = MixedLoss(0.5, 0.5)
    diceLoss = DiceLoss()
    loss_train = []
    start = 0
    epochs = epochs
    print(f"Parameters Initialized...")
    print(f"Starting to train for {epochs} epochs.")

    for epoch in range(start, epochs):

        print(f"Epoch no: {epoch+1}")
        _loss = 0
        num = random.randint(0, 200)

        for i, (image, mask) in enumerate(tqdm(train_dataloader)):

            #converting the image to cuda decice
            image = image.to("cuda:0")
            mask = mask.to("cuda:0") 

            #zero grading the optimizer
            optimizerImg.zero_grad()    

            #input the image into the model and getting the reconstructed image
            mask_pred = model(image)   

            #Loss functions for evaluation
            # loss = nvidia_mix_loss(image_pred, image)
            loss = diceLoss(mask_pred, mask)


            #adding a loss function 
            _loss += loss.item()    

            #backpropogation algorithm
            loss.backward()
            optimizerImg.step() 

            #saving a sample of this epoch
            if i == num:
                save_sample(epoch+1, image, mask, mask_pred, 'train')

        loss_train.append(_loss)

        #Saving the minimum loss wala model
        print(f"Epoch: {epoch+1}, Training loss: {_loss}")
        if loss_train[-1] == min(loss_train):
            print('Saving Model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizerImg.state_dict(),
                'loss': loss_train
            }, f'saved_model/latent_to_mask.tar')
        print('\nProceeding to the next epoch...')





def test_a_sample(batch_size=4):
    train_dataloader = DataLoader().load_data(batch_size)
    model = Autoencoder32K(outputType="Image")
    check = torch.load('saved_model/EncoderDecoder_road_image2image32K.tar')
    model.load_state_dict(check['model_state_dict'])
    model = model.to("cuda:0")

    num = random.randint(0, 100)
    for i, image in enumerate(train_dataloader):
        if i==num:
            image = image.to("cuda:0")
            image_pred = model(image)
            save_sample(epoch=None, x=image, img_pred=image_pred, mode='test')
            break

# train(70)
# test_a_sample()

