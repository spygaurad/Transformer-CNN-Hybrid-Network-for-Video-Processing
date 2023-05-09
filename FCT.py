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

from metric import DiceLoss, JaccardScore

from PIL import Image
from torchvision import transforms

from tensorboardX import SummaryWriter 


class Attention(nn.Module):
    def __init__(self, channels, num_heads, proj_drop=0.0, kernel_size=3, stride_kv=1, stride_q=1, padding_kv="same", padding_q="same", attention_bias=True):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias, groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias, groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)
        self.attention = nn.MultiheadAttention(embed_dim=channels, bias=attention_bias, batch_first=True, num_heads=self.num_heads)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def _build_projection(self, x, qkv):
        if qkv == "q":
            x1 = self.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 =self.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)            
        elif qkv == "v":
            x1 = self.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)        
        return proj


    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")
        return q, k, v


    def forward(self, x):
       q, k, v = self.forward_conv(x)
       q = q.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
       k = k.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
       v = v.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
       q = q.permute(0, 2, 1)
       k = k.permute(0, 2, 1)
       v = v.permute(0, 2, 1)
       x1 = self.attention(query=q, value=v, key=k, need_weights=False)
       x1 = x1[0].permute(0, 2, 1)
       x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
    #    x1 = self.dropout(x1, self.proj_drop)
       return x1

    


class Transformer(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads, dpr=None, proj_drop=0.0, attention_bias=True, padding_q="same", padding_kv="same", stride_kv=1, stride_q=1):
        super().__init__()
        self.attention_output = Attention(channels=in_channels, num_heads=num_heads, proj_drop=proj_drop, padding_q=padding_q, padding_kv=padding_kv, stride_kv=stride_kv, stride_q=stride_q, attention_bias=attention_bias)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3




class Wide_Focus(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gelu(x1)
        x1 = self.dropout(x1)
        x2 = self.conv2(x)
        x2 = self.gelu(x2)
        x2 = self.dropout(x2)
        x3 = self.conv3(x)
        x3 = self.gelu(x3)
        x3 = self.dropout(x3)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = self.gelu(x_out)
        x_out = self.dropout(x_out)
        return x_out
        

    
class Block_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.blk = blk
        self.conv1_a = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv1_b = nn.Conv2d(3, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(in_channels=out_channels, out_channels=out_channels, num_heads=att_heads, dpr=dpr)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) 

    def forward(self, x, scale_img="none"):
        if ((self.blk=="first") or (self.blk=="bottleneck")):
            x1 = self.relu(self.conv1_a(x))
            x1 = self.relu(self.conv2(x1))
            x1 = self.maxpool(self.dropout(x1))
            out = self.trans(x1)
        else:
            skip_x = self.relu(self.conv1_b(scale_img))
            x1 = torch.cat([skip_x, x], dim=1)
            x1 = self.relu(self.conv2(x1))
            x1 = self.relu(self.conv3(x1))
            x1 = self.maxpool(self.dropout(x1))
            out = self.trans(x1)
        return out




class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(in_channels=out_channels, out_channels=out_channels, num_heads=att_heads, dpr=dpr)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, skip):
        x1 = self.upsample(x)
        x1 = self.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = self.dropout(x1)
        out = self.trans(x1)
        return out




class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
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



class FCTEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        att_heads = 4
        filters = [8, 16, 32, 64, 128] 
        blocks = len(filters)
        stochastic_depth_rate = 0.0
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]
        self.drp_out = 0.3
        self.scale_img = nn.AvgPool2d(2,2)   

        # model
        self.block_1 = Block_encoder_bottleneck("first", 3, filters[0], att_heads, dpr[0])
        self.block_2 = Block_encoder_bottleneck("second", filters[0], filters[1], att_heads, dpr[1])
        self.block_3 = Block_encoder_bottleneck("third", filters[1], filters[2], att_heads, dpr[2])
        self.block_4 = Block_encoder_bottleneck("fourth", filters[2], filters[3], att_heads, dpr[3])
        self.block_5 = Block_encoder_bottleneck("bottleneck", filters[3], filters[4], att_heads, dpr[4])

    def forward(self,x):
        scale_img_2 = self.scale_img(x)
        scale_img_3 = self.scale_img(scale_img_2)
        scale_img_4 = self.scale_img(scale_img_3)  
        x1 = self.block_1(x)
        x2 = self.block_2(x1, scale_img_2)
        x3 = self.block_3(x2, scale_img_3)
        x4 = self.block_4(x3, scale_img_4)
        x = self.block_5(x4)
        return x1, x2, x3, x4, x




class FCTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        att_heads = 4
        filters = [128, 64, 32, 16, 8] 
        blocks = len(filters)
        stochastic_depth_rate = 0.0
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]
        self.drp_out = 0.3 

        self.block_6 = Block_decoder(filters[0], filters[1], att_heads, dpr[1])
        self.block_7 = Block_decoder(filters[1], filters[2], att_heads, dpr[2])
        self.block_8 = Block_decoder(filters[2], filters[3], att_heads, dpr[3])
        self.block_9 = Block_decoder(filters[3], filters[4], att_heads, dpr[4])
        self.ds = DS_out(filters[4], 1)

    def forward(self, x1, x2, x3, x4, x): 
        x = self.block_6(x, x4)
        x = self.block_7(x, x3)
        x = self.block_8(x, x2)
        x = self.block_9(x, x1)
        out9 = self.ds(x)
        return out9




class FCT(nn.Module):
    def __init__(self):
        super(FCT, self).__init__()
        self.encoder = FCTEncoder()
        self.decoder = FCTDecoder()
    
    def forward(self, x):
        x1, x2, x3, x4, bottleneck_latent = self.encoder(x)
        output = self.decoder(x1, x2, x3, x4, bottleneck_latent)
        return output

model = FCT().to('cuda')
image = torch.rand(4, 3, 256, 256).to('cuda')
output = model(image)
print(output.shape)

