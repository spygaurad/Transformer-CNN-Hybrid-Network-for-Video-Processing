import torch
import torch.nn as nn


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
        out = self.relu(self.conv3(x1))
        return out




class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [8, 16, 32, 64, 128] 
        self.drp_out = 0.3
        self.scale_img = nn.AvgPool2d(2, 2)   

        self.block_1 = EncoderBlock("first", 3, filters[0])
        self.block_2 = EncoderBlock("second", filters[0], filters[1])
        self.block_3 = EncoderBlock("third", filters[1], filters[2])
        self.block_4 = EncoderBlock("fourth", filters[2], filters[3])
        self.block_5 = EncoderBlock("bottleneck", filters[3], filters[4])


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
        x5 = self.block_5(x4)
        return x5



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [128, 64, 32, 16, 8]
        self.drp_out = 0.3

        self.block_4 = DecoderBlock(filters[0], filters[1])
        self.block_3 = DecoderBlock(filters[1], filters[2])
        self.block_2 = DecoderBlock(filters[2], filters[3])
        self.block_1 = DecoderBlock(filters[3], filters[4])
        self.ds = DeepSupervisionBlock(filters[4], 3)
        
    def forward(self, x):
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



print("AUTOENCODER")
data = (torch.rand(size=(1, 3, 256, 256)))
AE = AutoEncoder()
img_out = AE(data)
print("Latent's Shape:", img_out[0].shape)
print("Output's ShapeL", img_out[1].shape)