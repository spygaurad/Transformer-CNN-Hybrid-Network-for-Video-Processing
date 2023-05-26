import torch
import torch.nn as nn
from UNet import UNet
from AutoEncoder import AutoEncoder

class Image2Image2Mask(nn.Module):

    def __init__(self):
        super(Image2Image2Mask, self).__init__()

        self.image2imageAE = AutoEncoder()
        self.unet = UNet()

    def forward(self, x):
        imageLatent, reconsImage = self.image2imageAE(x)
        segMask = self.unet(reconsImage)
        return imageLatent, reconsImage, segMask


# print("Combined Model")
# data = (torch.rand(size=(4, 3, 256, 256)))
# i2i2m = Image2Image2Mask()
# imageLatent, reconsImage, segMask = i2i2m(data)
# print("Latent's Shape: ", imageLatent.shape)
# print("Reconstructed Image's Shape: ", reconsImage.shape)
# print("Segmentation Mask's Shape: ", segMask.shape)
    