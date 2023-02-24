from collections import deque
import torch
import torch.nn as nn
import math
import torch.optim as optim
from AE_128_16K import Autoencoder4K
from dataset import DataloaderSequential
from TransformerEncoder import TransformerEncoder
from collections import deque
from metric import MixedLoss
import numpy as np
import random
from tqdm import tqdm
import os
from torchvision import transforms
from tensorboardX import SummaryWriter



SEQUENCE_LENGTH = 5
CHUNK_LENGTH = 4
BATCH_SIZE = 8
EMBEDDED_DIMENSION = 4096
# DEVICE =  "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



encoderdecoder = Autoencoder4K(outputType="image")
encoderdecoder.load_state_dict(torch.load('saved_model/autoencoder_16k_VOS_30.tar')['model_state_dict'])
for params in encoderdecoder.encoder.parameters():
    params.requires_grad = False



class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.encoder = encoderdecoder.encoder
        
    def forward(self, x):
        bottleneck_4K = self.encoder(x)
        return bottleneck_4K


#generate a code
class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, dropout):
        super(Transformer_Encoder, self).__init__()
        # self.transformerencoder = TransformerEncoder(input_dim=input_dim, hidden_dim=input_dim, num_layers=num_layers, num_heads=num_heads, dropout=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # transformer_latent = self.transformerencoder(x, mask=None)
        transformer_latent = self.transformer_encoder(x, mask=None)
        return transformer_latent



class CNN_Decoder(nn.Module):
    def __init__(self):
        super(CNN_Decoder, self).__init__()
        self.decoder = encoderdecoder.decoder 

    def forward(self, x):
        out = self.decoder(x)
        return out




class VideoSegmentationNetwork(nn.Module):

    def __init__(self):

        super(VideoSegmentationNetwork, self).__init__()
         
        #the pre trained encoder which encodes the input into total number of 32K parameters 
        self.cnnencoder = CNN_Encoder()

        #loading the custom transformer encoder class
        # self.transenc = Transformer_Encoder(input_dim=EMBEDDED_DIMENSION, num_layers=2, num_heads=2)
        self.transenc = Transformer_Encoder(input_dim=EMBEDDED_DIMENSION, num_layers=4, num_heads=4, dropout=0.1)

        #the CNN decoder which is slightly pre-trained but is fine tuned to decode the transformer's output
        self.cnndecoder = CNN_Decoder()

        #the buffer object where we store the sequence of sequences of frame's latents for a given frame index
        self.sequence_window = deque()

        #counts which instance of frame that we are looking at in the sequence window. Signifies the index of the window in which we currently are present.
        self.sequence_counter = -1

        #generate a sinusoidal positional embedding
        self.positions = self.__positionalencoding__(d_model=EMBEDDED_DIMENSION, length=SEQUENCE_LENGTH*CHUNK_LENGTH).to(DEVICE)


    def forward(self, x):

        latents = []
        image_preds = []

        # sending the input to the cnn encoder
        # maskFrameNo = 2
        maskFrameNo = random.randint(1, SEQUENCE_LENGTH)
        for i in range(x.shape[0]):
            if i == maskFrameNo:
                l = torch.nn.init.xavier_normal_(torch.empty(BATCH_SIZE, EMBEDDED_DIMENSION*CHUNK_LENGTH)).to(DEVICE)
            else:
                l = self.cnnencoder(x[i])
            l = self.__split_and_stack__(l)
            latents.append(l)
        #before sending to the transformer, this is the pre-processing we need
        latents = torch.stack(latents).permute(1, 0, 2, 3)
        latents = latents.reshape(latents.shape[0], latents.shape[1]*latents.shape[2], latents.shape[3])
        latents += self.positions

        # sending the latents predicted to the transformer
        latents_pred = self.transenc(latents)
        
        #decoding all the sequence of the latents
        # latents_pred = latents_pred.reshape(SEQUENCE_LENGTH, BATCH_SIZE, ,EMBEDDED_DIMENSION)
        latents_pred = latents_pred.reshape(SEQUENCE_LENGTH, BATCH_SIZE, CHUNK_LENGTH, EMBEDDED_DIMENSION)
        for i in range(latents_pred.shape[0]):
            l_hat = self.__unstack_and_merge__(latents_pred[i])
            image_preds.append(self.cnndecoder(l_hat))

        image_preds = torch.stack(image_preds)
        return image_preds


    def __positionalencoding__(self, d_model, length):
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe


    def __split_and_stack__(self, x):
        latent_sequence = x.view(BATCH_SIZE, CHUNK_LENGTH, -1)
        return latent_sequence


    def __unstack_and_merge__(self, x):
        chunks = x.split(1, dim=1)
        chunks = [chunk.squeeze(dim=1) for chunk in chunks]
        merged_x = torch.cat(chunks, dim=1)
        return merged_x





def train(epochs, lr=1e-6):
    
    print(f"Using {DEVICE} device...")
    print("Loading Datasets...")
    train_data = DataloaderSequential(csv_file="data_sequential_VOS.csv", batch_size=BATCH_SIZE, imageSize=128).load_images()
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    #loading the model
    model = VideoSegmentationNetwork().to(DEVICE)
    # model.load_state_dict(torch.load('saved_model/transformer_full_model_4k_0.tar')['model_state_dict'])

    #initializing the optimizer for transformer
    optimizerTransformer = optim.AdamW(model.parameters(), lr)

    #loss function
    # nvidia_mix_loss = MixedLoss(0.5, 0.5)
    mseloss = torch.nn.MSELoss()

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
            
            optimizerTransformer.zero_grad()

            image = torch.stack(image).to(DEVICE)

            #input the image into the model
            imagePred = model(image)

            # MS-SSIM loss + MSE Loss for model evaluation

            loss = mseloss(imagePred, image)

            #getting the loss's number
            _loss += loss.item()

            if i % accumulation_steps==0:
                loss.backward()
                optimizerTransformer.step()
                optimizerTransformer.zero_grad()
            else:
                loss.backward()

            #saving a sample in each epoch
            if i%num==0: 
                [__save_sample__(epoch+1, image[j], imagePred[j], str(j+1)) for j in range(SEQUENCE_LENGTH)]

        writer.add_scalar("Training Loss", _loss, i)
        loss_train.append(_loss)

        print(f"Epoch: {epoch+1}, Training loss: {_loss}")

        if epoch%10==0:
            print('Saving Model...')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizerTransformer.state_dict(), 'loss': loss_train} , f'saved_model/transformer_full_model_16K_{epoch}.tar')
            # torch.save({'epoch': epoch, 'model_state_dict': decoderModel.state_dict(), 'optimizer_state_dict': optimizerCNNDecoder.state_dict(), 'loss': loss_train} , f'saved_model/CNN_decoder_model{epoch}.tar')
        print('\nProceeding to the next epoch...')



def __save_sample__(epoch, x, img_pred, iter):
    path = f'Training_Sneakpeeks/Transformer_Training_16K/'
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

# vsn = VideoSegmentationNetwork()

# for i in range(100):
#     input_tensor = torch.randn(BATCH_SIZE, 3, 256, 256)
#     state, throughput0, throughput1 = vsn(x=input_tensor)
#     print(f'Iteration: {i+1}')
#     if state:
#         print(throughput1.shape)
    


# print(f'Input tensor {input_tensor.shape} -> {splitandstack.shape} -> {unstackandmerge.shape}') 
