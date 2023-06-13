import os
import numpy as np
import random
import torch
import torch.nn as nn
import math
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision import transforms


from metric import MixedLoss
from AutoEncoder import AutoEncoder
from dataset import DataloaderSequential
from TransformerEncoder import TransformerEncoder
'''
    How do we send an image to a transformer? 

    Well, we ecode the image of size [3, 256, 256] to a size of 65536 dimensional vector

    The latent of each Image is of 65536 dimension. It is linearly broken down into 16 chunks of dimension EMBEDDED_DIMENSION. 
    There will be total 16 vectors. They will be stacked on top of each other. It will look something like this: 
    from [batch, 65536] to [batch, 16, 4096].
    It will be than covered by two tensor tokens, which are static random tensors. One will be added on top of 
    the chunk and one on the bottom. creating a size of [batch, 18, 4096]. 
    It is represneted as a sequence as:

    <i> l1_1 l2_1 ... l16_1 </i> <i> l1_2 l2_2 ... l16_2 </i> <i> l1_3 l2_3 ... l16_3 </i> <i> l1_4 l2_4 ... l16_4 </i>   

    A sequence of 16 vectors represent an image. So to reprenst a sequence of 7 images, we need 112 vectors.
    For the sequence of 112 vectors, we need tokens which separates one image's latent from another. Since 1 image is 
    represnted by 16 latents, we need to keep those tokens 16-distance apart and we need total of 14 such vectors. 

    Total number of vectors that we will be resulting in is 126
'''

SEQUENCE_LENGTH = 5
EMBEDDED_DIMENSION = 4096
CHUNK_UNITS = 64
BATCH_SIZE = 16
MODEL_NAME = "CNNTRANSFORMER_UNIFIED_V1"
# DEVICE =  "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



encoderdecoder = AutoEncoder()
encoderdecoder.load_state_dict(torch.load('')['model_state_dict'])



class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.encoder = encoderdecoder.encoder
        for params in self.encoder.parameters():
            params.requires_grad = False
        
    def forward(self, x):
        cnn_latent = self.encoder(x)
        return cnn_latent
    


class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, dropout):
        super(Transformer_Encoder, self).__init__()
        # self.transformerencoder = TransformerEncoder(input_dim=input_dim, hidden_dim=input_dim, num_layers=num_layers, num_heads=num_heads, dropout=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        # transformer_latent = self.transformerencoder(x, mask=None)
        transformer_latent = self.transformer_encoder(x, mask=mask)
        return transformer_latent



class CNN_Decoder(nn.Module):
    def __init__(self):
        super(CNN_Decoder, self).__init__()
        self.decoder = encoderdecoder.decoder 

    def forward(self, latent):
        out = self.decoder(latent)
        return out




class VideoSegmentationNetwork(nn.Module):

    def __init__(self):

        super(VideoSegmentationNetwork, self).__init__()
         
        #the pre trained encoder which encodes the input into total number of 32K parameters 
        self.cnnencoder = CNN_Encoder()

        #loading the transformer encoder class
        self.transenc = Transformer_Encoder(input_dim=EMBEDDED_DIMENSION, num_layers=4, num_heads=8, dropout=0.1)

        #the CNN decoder which is slightly pre-trained but is fine tuned to decode the transformer's output
        self.cnndecoder = CNN_Decoder()

        #get the tensor of size [sequence_length, embedding dimension] which is encoded like... (see the method implementation)
        self.positions = self.__get_positional__tensor().to(DEVICE)

        #the two learnable tokens which separates one frame's latent sequence with another frame's sequence of latents
        self.sof = nn.Parameter(torch.randn(EMBEDDED_DIMENSION)).expand(BATCH_SIZE, 1, -1).to(DEVICE)
        self.eof = nn.Parameter(torch.randn(EMBEDDED_DIMENSION)).expand(BATCH_SIZE, 1, -1).to(DEVICE)




    def forward(self, x, epoch):

        latents = []
        image_preds = []

        # sending the input to the cnn encoder
        maskFrameNo = 4
        for i in range(x.shape[0]):
            if i == maskFrameNo:
                l = torch.zeros(BATCH_SIZE, EMBEDDED_DIMENSION*CHUNK_UNITS).to(DEVICE)
            else:
                l = self.cnnencoder(x[i])
            l = l.permute(0, 2, 1)
            latents.append(l)

        #before sending to the transformer, this is the pre-processing we need
        num_zeros = int(0.25 * latents.shape[1])
        zero_indices = torch.randperm(latents.shape[1])[:num_zeros]
        latents[:, zero_indices, :] = 0
        latents = latents.permute(1, 0, 2)


        attention_mask = self.get_mask_seq_cat(first_seq_len=256, second_seq_len=64).to(DEVICE)

        # sending the latents predicted to the transformer
        latents_pred = self.transenc(latents, mask=attention_mask)
        
        latents_pred = torch.cat([x[:, 1:-1, :] for x in torch.split(latents_pred, CHUNK_UNITS+2, dim=1)], dim=1)
        latents_pred = latents_pred.reshape(SEQUENCE_LENGTH, BATCH_SIZE, CHUNK_UNITS, EMBEDDED_DIMENSION)
        for i in range(latents_pred.shape[0]):
            l_hat = self.__unstack_and_merge__(latents_pred[i])
            image_preds.append(self.cnndecoder(l_hat))

        image_preds = torch.stack(image_preds)
        return image_preds


    def get_positional_encoding(self, seq_len, embedding_dim, learnable=False):
        if not learnable:
            pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
            pos_embedding = torch.zeros(seq_len, embedding_dim)
            pos_embedding[:, 0::2] = torch.sin(pos * div)
            pos_embedding[:, 1::2] = torch.cos(pos * div)
        else:
            pos_embedding = nn.Parameter(torch.randn(seq_len, embedding_dim, requires_grad=True))
        return pos_embedding

        
    def __get_positional__tensor(self, embedding_dim=EMBEDDED_DIMENSION):
        pos_embedding_chunk = self.get_positional_encoding(CHUNK_UNITS, EMBEDDED_DIMENSION, learnable=False)
        pos_embedding_frame = self.get_positional_encoding(SEQUENCE_LENGTH, EMBEDDED_DIMENSION, learnable=True)
        pos_tensor = []
        for i in range(pos_embedding_frame.shape[0]):
            temp = pos_embedding_chunk
            temp = torch.cat((pos_embedding_frame[i:i+1], temp), dim=0)
            temp = torch.cat((temp, pos_embedding_frame[i:i+1]), dim=0)
            pos_tensor.append(temp)
        pos_tensor = torch.cat(pos_tensor, dim=0)
        return pos_tensor

    def __split_and_stack__(self, x):
        latent_sequence = x.view(BATCH_SIZE, CHUNK_UNITS, -1)
        return latent_sequence


    def __unstack_and_merge__(self, x):
        chunks = x.split(1, dim=1)
        chunks = [chunk.squeeze(dim=1) for chunk in chunks]
        merged_x = torch.cat(chunks, dim=1)
        return merged_x

















def train(epochs, lr=1e-6):

    print(f"Using {DEVICE} device.")
    print("Loading Datasets...")
    train_data = DataloaderSequential(csv_file="data_sequential_VOS.csv", batch_size=BATCH_SIZE, imageSize=256).load_images()
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    #loading the model
    model = VideoSegmentationNetwork().to(DEVICE)
    
    #checking the size of the model
    

    # model.load_state_dict(torch.load('saved_model/transformer_full_model.tar')['model_state_dict'])


    #initializing the optimizer for transformer
    optimizerTransformer = optim.Adam(model.parameters(), lr)
    #initializing the optimizer for CNN decoder. It will learn in 10% of the rate that the transformer is learning in

    #loss function
    # nvidia_mix_loss = MixedLoss(0.5, 0.5)
    mseloss = nn.MSELoss()

    writer = SummaryWriter(log_dir="logs")     

    loss_train = []

    print("Parameters Initialized...")
    print(f"Starting to train for {epochs} epochs...")

    for epoch in range(epochs):

        print(f"Epoch no: {epoch+1}")
        _loss = 0
        num = random.randint(0, (len(train_data)//BATCH_SIZE) - 1)
        accumulation_steps = 4

        for i, image in enumerate(tqdm(train_data)):

            #zero gradding the optimizer
            optimizerTransformer.zero_grad()

            image = torch.stack(image).to(DEVICE)

            #input the image into the model
            imagePred = model(image, epoch)
            #here, we take our output as the latent which is just on the third frame 

            # loss = nvidia_mix_loss(imagePred, image)
            loss = mseloss(imagePred, image)

            #getting the loss value
            _loss += loss.item()
            
            #gradient accumulation
            if i%accumulation_steps==0:
                loss.backward()
                optimizerTransformer.step()
                optimizerTransformer.zero_grad()
            else:
                loss.backward()
            
            #saving the sample in each epoch
            if i%num==0 and epoch%5==0: 
                [__save_sample__(epoch+1, image[j], imagePred[j], str(j+1)) for j in range(SEQUENCE_LENGTH)]

        writer.add_scalar("Training Loss", _loss, epoch)

        loss_train.append(_loss)

        print(f"Epoch: {epoch+1}, Training loss: {_loss}")

        if epoch%50==0:
            print('Saving Model...')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizerTransformer.state_dict(), 'loss': loss_train} , f'saved_model/transformer_model_{epoch}.tar')
            # torch.save({'epoch': epoch, 'model_state_dict': decoderModel.state_dict(), 'optimizer_state_dict': optimizerCNNDecoder.state_dict(), 'loss': loss_train} , f'saved_model/CNN_decoder_model{epoch}.tar')
        print('\nProceeding to the next epoch...')



def __save_sample__(epoch, x, img_pred, iter):
    path = f'Training_Sneakpeeks/Transformer_Training/'
    try:
        os.makedirs(path)
    except:
        pass
    elements = [x, img_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    for i, element in enumerate(elements):
        try:
            element.save(f"{path}{epoch}_{iter}_{['image', 'image_trans_pred'][i]}.jpg")
        except:
            pass


train(epochs=500)

# vsn = VideoSegmentationNetwork().to(DEVICE)

# input_tensor = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, 3, 256, 256).to(DEVICE)
# imagePred = vsn(input_tensor, 1)
# print(imagePred.shape)
    


# print(f'Input tensor {input_tensor.shape} -> {splitandstack.shape} -> {unstackandmerge.shape}') 