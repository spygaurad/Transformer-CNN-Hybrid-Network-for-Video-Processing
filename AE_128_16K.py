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
import os


BATCH_SIZE = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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

        self.conv4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 16, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(16)

        #initializing the vector for mean and standard deviation 
        '''
        self.fc_mu = nn.Linear(32768, 32768)
        self.fc_var = nn.Linear(32768, 32768)
        '''

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

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
        # x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)

        #breaking the latents into 4 chunks, linearly
        x = x.view(BATCH_SIZE, 4, 4096)

        #to run in a VAE setup, use the following part of code
        '''
        #flattening the layers
        x = x.view(x.size(0), -1)

        #getting the distribution of mean and the standard deviation
        mu = self.relu(self.fc_mu(x))
        sigma = self.relu(self.fc_var(x))

        #re-parameterizing the vector picking up samples from the distribution defined above
        x = self.reparameterize(mu, sigma)
        return x, mu, sigma
        '''
        #just return the latent if we're not using the VAE
        return x


    #reparameterization function which constructs a vector picking upsamples from the distributions that we have 
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return eps*std + mu




class Decoder_32K(nn.Module):
    def __init__(self, outputDeterminer):
        super(Decoder_32K, self).__init__()

        self.outputDeterminer = outputDeterminer
            
        self.conv1= nn.Conv2d(16, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)

        self.transConv1 = nn.ConvTranspose2d(256, 384, 2, 2, padding = 0)
        self.dbn2 = nn.BatchNorm2d(384)

        self.transConv2 = nn.ConvTranspose2d(384, 128, 2, 2, padding = 0)
        self.dbn3 = nn.BatchNorm2d(128)

        # self.transConv3= nn.ConvTranspose2d(192, 128, 2, 2, padding = 0)
        # self.dbn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
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

        #we now convert a broken linear vector to a volume of a desired shape
        # '''
        x = x.view(x.shape[0], 4, 4096)
        x = x.view(x.shape[0], 16, 32, 32)
        # '''
        x = self.finalactivation(self.outputDeterminerNorm(self.outputDeterminerConv(self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(self.relu(self.dbn3(self.transConv2(self.relu(self.dbn2(self.transConv1(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))))))))))))))))))))))
        return x




class Autoencoder4K(nn.Module):
    def __init__(self, outputType):
        super(Autoencoder4K, self).__init__()

        self.encoder = Encoder_32K(Bottleneck, [3, 4])

        if outputType.lower() == "image":
            self.decoder = Decoder_32K("image")
        else:
            self.decoder = Decoder_32K("mask")

    def forward(self, x):
        bottleneck_32K = self.encoder(x)
        decoded = self.decoder(bottleneck_32K) 
        return decoded





def save_sample(epoch=0, x=None, mask_pred=None, mode='train'):
    path = f'Training_Sneakpeeks/image_to_image_16k/'
    try:
        os.makedirs(path)
    except:
        pass
    elements = [x,  mask_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    for i, element in enumerate(elements):
        element.save(f"{path}{epoch}_{['image', 'image_pred'][i]}.jpg")





def train(epochs, batch_size=BATCH_SIZE, lr=0.0001):

    print(f"Using {DEVICE} device.")
    print("Loading Datasets...")
    train_dataloader = DataLoader(batch_size=batch_size, trainingType="semisupervised", return_train_and_test=False).load_data("data_image_train_VOS.csv", testDataCSV=None)
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    model = Autoencoder4K("image").to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    mseloss = torch.nn.MSELoss()
    loss_train = []
    start = 0
    epochs = epochs
    print(f"Parameters Initialized...")
    print(f"Starting to train for {epochs} epochs.")

    # try:

    #training for n epochs
    for epoch in range(start, epochs):

        print(f"Epoch no: {epoch+1}")
        _loss = 0
        num = random.randint(0, len(train_dataloader)//batch_size - 1)

        for i, image in enumerate(tqdm(train_dataloader)):

            #converting the image to cuda decice
            image = image.to(DEVICE)

            #zero grading the optimizer
            optimizer.zero_grad()

            #input the image into the model and getting the reconstructed image
            noise_image = image + torch.randn(image.size()).to(DEVICE)*0.05 + 0.0
            output = model(noise_image)

            #Loss functions for evaluation
            loss = mseloss(output, image)

            #adding a loss function 
            _loss += loss.item()

            #backpropogation algorithm
            loss.backward()
            optimizer.step() 

            #saving a sample of this epoch
            if epoch%5==0 and i==num:
                save_sample(epoch+1, image, output)
        loss_train.append(_loss)


        #Saving the minimum loss wala model
        print(f"Epoch: {epoch+1}, Training loss: {_loss}")
        if epoch%10 == 0 and loss_train[-1] == min(loss_train):
            print('Saving Model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train
            }, f'saved_model/autoencoder_16k_VOS_{epoch}.tar')
        print('\nProceeding to the next epoch...')

    return False




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
import os


BATCH_SIZE = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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

        self.conv4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 16, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(16)

        #initializing the vector for mean and standard deviation 
        '''
        self.fc_mu = nn.Linear(32768, 32768)
        self.fc_var = nn.Linear(32768, 32768)
        '''

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

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
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)

        #breaking the latents into 4 chunks, linearly
        x = x.view(x.shape[0], 4, 4096)

        #to run in a VAE setup, use the following part of code
        '''
        #flattening the layers
        x = x.view(x.size(0), -1)

        #getting the distribution of mean and the standard deviation
        mu = self.relu(self.fc_mu(x))
        sigma = self.relu(self.fc_var(x))

        #re-parameterizing the vector picking up samples from the distribution defined above
        x = self.reparameterize(mu, sigma)
        return x, mu, sigma
        '''
        #just return the latent if we're not using the VAE
        return x


    #reparameterization function which constructs a vector picking upsamples from the distributions that we have 
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return eps*std + mu




class Decoder_32K(nn.Module):
    def __init__(self, outputDeterminer):
        super(Decoder_32K, self).__init__()

        self.outputDeterminer = outputDeterminer
            
        self.conv1= nn.Conv2d(16, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)

        self.transConv1 = nn.ConvTranspose2d(256, 384, 2, 2, padding = 0)
        self.dbn2 = nn.BatchNorm2d(384)

        self.transConv2 = nn.ConvTranspose2d(384, 128, 2, 2, padding = 0)
        self.dbn3 = nn.BatchNorm2d(128)

        # self.transConv3= nn.ConvTranspose2d(192, 128, 2, 2, padding = 0)
        # self.dbn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
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

        #we now convert a broken linear vector to a volume of a desired shape
        # '''
        x = x.view(x.shape[0], 4, 4096)
        x = x.view(x.shape[0], 16, 32, 32)
        # '''
        x = self.finalactivation(self.outputDeterminerNorm(self.outputDeterminerConv(self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(self.relu(self.dbn3(self.transConv2(self.relu(self.dbn2(self.transConv1(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))))))))))))))))))))))
        return x




class Autoencoder4K(nn.Module):
    def __init__(self, outputType):
        super(Autoencoder4K, self).__init__()

        self.encoder = Encoder_32K(Bottleneck, [3, 4])

        if outputType.lower() == "image":
            self.decoder = Decoder_32K("image")
        else:
            self.decoder = Decoder_32K("mask")

    def forward(self, x):
        bottleneck_32K = self.encoder(x)
        decoded = self.decoder(bottleneck_32K) 
        return decoded





def save_sample(epoch=0, x=None, mask_pred=None, mode='train'):
    path = f'Training_Sneakpeeks/image_to_image_16k/'
    try:
        os.makedirs(path)
    except:
        pass
    elements = [x,  mask_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    for i, element in enumerate(elements):
        element.save(f"{path}{epoch}_{['image', 'image_pred'][i]}.jpg")





def train(epochs, batch_size=BATCH_SIZE, lr=0.0001):

    print(f"Using {DEVICE} device.")
    print("Loading Datasets...")
    train_dataloader = DataLoader(batch_size=batch_size, trainingType="semisupervised", image_size=128, return_train_and_test=False).load_data("data_image_train_VOS.csv", testDataCSV=None)
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    model = Autoencoder4K("image").to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    mseloss = torch.nn.MSELoss()
    loss_train = []
    start = 0
    epochs = epochs
    print(f"Parameters Initialized...")
    print(f"Starting to train for {epochs} epochs.")

    # try:

    #training for n epochs
    for epoch in range(start, epochs):

        print(f"Epoch no: {epoch+1}")
        _loss = 0
        num = random.randint(0, len(train_dataloader)//batch_size - 1)

        for i, image in enumerate(tqdm(train_dataloader)):

            #converting the image to cuda decice
            image = image.to(DEVICE)

            #zero grading the optimizer
            optimizer.zero_grad()

            #input the image into the model and getting the reconstructed image
            noise_image = image + torch.randn(image.size()).to(DEVICE)*0.05 + 0.0
            output = model(noise_image)

            #Loss functions for evaluation
            loss = mseloss(output, image)

            #adding a loss function 
            _loss += loss.item()

            #backpropogation algorithm
            loss.backward()
            optimizer.step() 

            #saving a sample of this epoch
            if epoch%5==0 and i==num:
                save_sample(epoch+1, image, output)
        loss_train.append(_loss)


        #Saving the minimum loss wala model
        print(f"Epoch: {epoch+1}, Training loss: {_loss}")
        if epoch%10 == 0 and loss_train[-1] == min(loss_train):
            print('Saving Model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train
            }, f'saved_model/autoencoder_16k_VOS_{epoch}.tar')
        print('\nProceeding to the next epoch...')

    return False





error = train(21, batch_size=BATCH_SIZE)
# if error:
#     torch.cuda.empty_cache()
#     train(epochs=61, batch_size=4)





            