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


#Defining the gobal variables
SEQUENCE_LENGTH = 5
EMBEDDED_DIMENSION = 512
CHUNK_LENGTH = 64
BATCH_SIZE = 8
MODEL_NAME = "Transformer_Training_16K_UnifiedModel"
# DEVICE =  "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#Loading the previously trained encoder decoder model
encoderdecoder = Autoencoder4K(outputType="image")
# encoderdecoder.load_state_dict(torch.load('saved_model/autoencoder_16k_VOS_40_512D.tar')['model_state_dict'])


#We freeze the parameters in the CNN Encoder
class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.encoder = encoderdecoder.encoder
        # for params in self.encoder.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        bottleneck_4K = self.encoder(x)
        return bottleneck_4K


#The transformer Encoder layer, to learn the sequential information in between frames
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



#The transformer Decoder layer, to autoregressively predict the posterior frames
class Transformer_Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # self.fc_out = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, target, memory):
        output = self.transformer_decoder(target, memory)
        # output = self.dropout(self.fc_out(output))
        return output.permute(1, 0, 2)



#The CNN Decoder layer, which decodes the latent from the transformer
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

        #loading the transformer encoder class
        self.transenc = Transformer_Encoder(input_dim=EMBEDDED_DIMENSION, num_layers=4, num_heads=8, dropout=0.1)

        #loading the transformer decoder class
        # self.transdec = Transformer_Decoder(output_dim=EMBEDDED_DIMENSION, hidden_dim=EMBEDDED_DIMENSION*4, num_layers=4, num_heads=8, dropout=0.1)

        #the CNN decoder which is slightly pre-trained but is fine tuned to decode the transformer's output
        self.cnndecoder = CNN_Decoder()

        #generate a sinusoidal positional embedding
        self.positions = self.__get_positional__tensor().to(DEVICE)


    def forward(self, x, epoch=None):
        # '''
        latents = []
        image_preds = []

        #sending the inputs the CNN encoder
        for i in range(x.shape[0]):
            l = self.cnnencoder(x[i])
            l = l.permute(0, 2, 1)
            latents.append(l)

        #before sending to the transformer, this is the pre-processing we need
        latents = torch.concat(latents, axis=1)

        #mask random k% of the items in the sequence 
        num_zeros = int(0.15 * latents.shape[1])
        zero_indices = torch.randperm(latents.shape[1])[:num_zeros]
        latents[:, zero_indices, :] = 0

        #add the positional embedding
        latents += self.positions

        mask = torch.ones(batch_size, seq_length, seq_length)

        # Set the attention mask for the last 64 elements of each sequence to 0,
        # so they will be ignored by the Transformer encoder
        mask[:, 256:, :] = 0

        # Create a triangular mask for the attention weights for the last 64 elements of each sequence
        triangular_mask = torch.tril(torch.ones(seq_length, seq_length), diagonal=-256)
        triangular_mask = triangular_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        attention_mask = mask.unsqueeze(1) * triangular_mask.unsqueeze(-1)

        latents_pred = self.transenc(latents, mask=attention_mask)



        #decoding all the sequence of the latents
        chunks = torch.chunk(latents_pred, SEQUENCE_LENGTH, dim=1)
        for chunk in chunks:
            chunk = chunk.permute(0, 2, 1)
            image_preds.append(self.cnndecoder(chunk))

        image_preds = torch.stack(image_preds)
        return image_preds
        '''

        latents = []
        image_preds = []

        #sending the inputs the CNN encoder
        for i in range(x.shape[0]):
            l = self.cnnencoder(x[i])
            l = l.permute(0, 2, 1)
            latents.append(l)

        #before sending to the transformer, this is the pre-processing we need
        latents = torch.concat(latents, axis=1)

        #mask random k% of the items in the sequence 
        num_zeros = int(0.15 * latents.shape[1])
        zero_indices = torch.randperm(latents.shape[1])[:num_zeros]
        latents[:, zero_indices, :] = 0

        #add the positional embedding
        latents += self.positions

        mem = latents[:, :256, :]
        tgt = latents[:, 256:320, :]
        latents_pred = self.transdec(tgt.permute(1, 0, 2), mem.permute(1, 0, 2))

        imgPred = self.cnndecoder(latents_pred)

        return imgPred
        '''



    def get_positional_encoding(self, seq_len, embedding_dim):
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pos_embedding = torch.zeros(seq_len, embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div)
        pos_embedding[:, 1::2] = torch.cos(pos * div)
        return pos_embedding

    def __get_positional__tensor(self, embedding_dim=EMBEDDED_DIMENSION):
        pos_tensor1 = self.get_positional_encoding(seq_len=CHUNK_LENGTH, embedding_dim=EMBEDDED_DIMENSION)
        pos_tensor2 = self.get_positional_encoding(seq_len=SEQUENCE_LENGTH, embedding_dim=EMBEDDED_DIMENSION)
        # pos_embedding = nn.Parameter(torch.randn(SEQUENCE_LENGTH, EMBEDDED_DIMENSION, requires_grad=True, device=DEVICE))
        pos_tensor1 = torch.cat([pos_tensor1 for i in range(SEQUENCE_LENGTH)], dim=0)
        pos_tensor2 = torch.cat([pos_tensor2[:, i].repeat_interleave(CHUNK_LENGTH).unsqueeze(1) for i in range(pos_tensor2.shape[1])], dim=1)
        return pos_tensor1 + pos_tensor2


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
    data_size = len(train_data//BATCH_SIZE)
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    #loading the model
    model = VideoSegmentationNetwork().to(DEVICE)
    # model.load_state_dict(torch.load('saved_model/transformer_full_model_16K_10.tar')['model_state_dict'])

    #initializing the optimizer for transformer
    optimizerTransformer = optim.AdamW(model.transenc.parameters(), lr)

    
    # nvidia_mix_loss = MixedLoss(0.5, 0.5)
    mseloss = nn.MSELoss()

    writer = SummaryWriter(log_dir="logs")     

    loss_train = []

    print("Parameters Initialized...")
    print(f"Starting to train for {epochs} epochs...")

    for epoch in range(epochs):

        print(f"Epoch no: {epoch+1}")
        _loss = 0
        num = random.randint(0, data_size - 1)
        # num = random.randint(0, )
        accumulation_steps = 4

        for i, image in enumerate(tqdm(train_data)):
            
            optimizerTransformer.zero_grad()

            image = torch.stack(image).to(DEVICE)

            noise_image = image + torch.randn(image.size()).to(DEVICE)*0.05 + 0.01
            
            #input the image into the model
            imagePred = model(noise_image, epoch)

            # We add a temporal loss for the model too, for the model to learn temporal dependencies in the inputs
            frameloss = mseloss(imagePred, image[-1, :, :, :, :])
            loss = frameloss

            #getting the loss's number
            _loss += loss.item()

            if i % accumulation_steps==0:
                loss.backward()
                optimizerTransformer.step()
                optimizerTransformer.zero_grad()
            else:
                loss.backward()

            #saving a sample in each epoch
            if epoch%5==0 and i==num: 
                __save_sample__(epoch+1, image[-1], noise_image[-1], imagePred, 5)
                # __save_sample__(epoch+1, image, imagePred, 1)

        _loss = _loss/data_size
        writer.add_scalar("Training Loss", _loss, epoch)

        loss_train.append(_loss)

        print(f"Epoch: {epoch+1}, Training loss: {_loss}")

        if epoch%10==0:
            print('Saving Model...')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizerTransformer.state_dict(), 'loss': loss_train} , f'saved_model/{MODEL_NAME}_{epoch}.tar')
            # torch.save({'epoch': epoch, 'model_state_dict': decoderModel.state_dict(), 'optimizer_state_dict': optimizerCNNDecoder.state_dict(), 'loss': loss_train} , f'saved_model/CNN_decoder_model{epoch}.tar')
        print('\nProceeding to the next epoch...')


def __save_sample__(epoch, x, _x, img_pred, iter):
    path = f'Training_Sneakpeeks/{MODEL_NAME}/'
    try:
        os.makedirs(path)
    except:
        pass
    elements = [x, _x, img_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0, :, :, :])) for element in elements]
    for i, element in enumerate(elements):
        try:
            element.save(f"{path}{epoch}_{iter}_{['image', 'input', 'image_trans_pred'][i]}.jpg")
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
