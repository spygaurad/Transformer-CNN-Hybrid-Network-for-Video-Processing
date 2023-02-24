from collections import deque
import torch
import torch.nn as nn
import math
import torch.optim as optim
from AE_256_32K import Autoencoder32K
from dataset import DataLoaderSequential
from TransformerEncoder import TransformerEncoder
from collections import deque
from metric import MixedLoss
import numpy as np
import random
from tqdm import tqdm
import os
from torchvision import transforms
from tensorboardX import SummaryWriter


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
CHUNK_LENGTH = 8
BATCH_SIZE = 8
# DEVICE =  "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



encoderdecoder = Autoencoder32K(outputType="image")
encoderdecoder.load_state_dict(torch.load('saved_model/autoencoder_32K_VOS_60.tar')['model_state_dict'])
# for params in encoderdecoder.parameters():
#     params.requires_grad = False



class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.encoder = encoderdecoder.encoder
        
    def forward(self, x):
        bottleneck_32K = self.encoder(x)
        return bottleneck_32K
    

#generate a code 

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Transformer_Encoder, self).__init__()
        # self.transformerencoder = TransformerEncoder(input_dim=input_dim, hidden_dim=input_dim, num_layers=num_layers, num_heads=num_heads, dropout=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
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
        self.transenc = Transformer_Encoder(input_dim=EMBEDDED_DIMENSION, hidden_dim=EMBEDDED_DIMENSION, num_layers=2, num_heads=4, dropout=0.1)

        #the CNN decoder which is slightly pre-trained but is fine tuned to decode the transformer's output
        self.cnndecoder = CNN_Decoder()

        #the two learnable tokens which separates one frame's latent sequence with another frame's sequence of latents
        # self.sof = nn.Parameter(torch.randn(EMBEDDED_DIMENSION)).expand(BATCH_SIZE, 1, -1).to(DEVICE)
        # self.eof = nn.Parameter(torch.randn(EMBEDDED_DIMENSION)).expand(BATCH_SIZE, 1, -1).to(DEVICE)

        #get the tensor of size [sequence_length, embedding dimension] which is encoded like... (see the method implementation)
        self.positionalTensor = self.__get_positional__tensor().to(DEVICE)

        #the buffer object where we store the sequence of sequences of frame's latents for a given frame index
        self.sequence_window = deque()

        #counts which instance of frame that we are looking at in the sequence window. Signifies the index of the window in which we currently are present.
        self.sequence_counter = -1



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

    

    def __get_latent__chunks__(self):
        chunks = torch.Tensor(self.sequence_window)
        chunks = chunks.permute(1, 0, 2, 3)
        chunks = chunks.reshape(chunks.shape[0], chunks.shape[1]*chunks.shape[2], chunks.shape[-1]) 
        return chunks


    def __sequence_counter__(self):
        self.sequence_counter += 1
        if self.sequence_counter > 4:
            self.sequence_counter = 0
        return self.sequence_counter


    def __reshape_split_and_stack__(self, x):
        # x = x.view(x.shape[0], -1)
        latent_sequence = x.view(BATCH_SIZE, CHUNK_LENGTH, -1)
        # sequence = torch.cat((torch.cat((self.sof, latent_sequence), dim=1), self.eof), dim=1)
        return latent_sequence


    def __reshape_unstack_and_merge__(self, x):
        # x = x[:, 1:-1]
        chunks = x.split(1, dim=1)
        chunks = [chunk.squeeze(dim=1) for chunk in chunks]
        merged_x = torch.cat(chunks, dim=1)
        # merged_x = merged_x.view(BATCH_SIZE, 8, 64, 64)
        return merged_x


    def __positionalencoding__(self, d_model, length):
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe


    def __get_positional__tensor(self):
        ''' 
            A = [A1, A2, A3, A4]
            B = [B1, B2, B3, B4, B5]
            T = [ B1, A1, A2, A3, A4, B1, B2, A1, A2, A3, A4, B2, B3, A1, A2, A3, A4, B3,.... B5, A1, A2, A3, A4, B5 ]
        '''
        PE_latentSequence = self.__positionalencoding__(EMBEDDED_DIMENSION, CHUNK_LENGTH*SEQUENCE_LENGTH) 
        # PE_imageSequence = self.__positionalencoding__(EMBEDDED_DIMENSION, SEQUENCE_LENGTH)
        # T = []
        # for seq in PE_imageSequence:
        #     t = torch.cat((seq.unsqueeze(dim=0), PE_latentSequence, seq.unsqueeze(dim=0)))
        #     T.append(t)
        # positional_tensor = torch.cat(T, dim=0)
        return PE_latentSequence




def train(epochs, lr=1e-6):

    print(f"Using {DEVICE} device.")
    print("Loading Datasets...")
    train_data = DataLoaderSequential(csv_file="data_sequential_VOS.csv", batch_size=BATCH_SIZE, image_size=256).load_data()
    print("Dataset Loaded.")
    print("Initializing Parameters...")

    #loading the model
    model = VideoSegmentationNetwork().to(DEVICE)
    # model.load_state_dict(torch.load('saved_model/transformer_full_model.tar')['model_state_dict'])

    #the CNN's encoder and decoder model
    decodermodel = model.cnndecoder
    encodermodel = model.cnnencoder



    #initializing the optimizer for transformer
    optimizerTransformer = optim.AdamW(model.parameters(), lr)
    #initializing the optimizer for CNN decoder. It will learn in 10% of the rate that the transformer is learning in

    #loss function
    nvidia_mix_loss = MixedLoss(0.5, 0.5)

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
            imagePred = model(image)
            #here, we take our output as the latent which is just on the third frame 

            loss = nvidia_mix_loss(imagePred, image)

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
            if i%num==0: 
                [__save_sample__(epoch+1, image[j], imagePred[j], str(j+1)) for j in range(SEQUENCE_LENGTH)]

        writer.add_scalar("Training Loss", _loss, i)
        loss_train.append(_loss)

        print(f"Epoch: {epoch+1}, Training loss: {_loss}")

        if epoch%50==0:
            print('Saving Model...')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizerTransformer.state_dict(), 'loss': loss_train} , f'saved_model/transformer_model_{epoch}.tar')
            # torch.save({'epoch': epoch, 'model_state_dict': decoderModel.state_dict(), 'optimizer_state_dict': optimizerCNNDecoder.state_dict(), 'loss': loss_train} , f'saved_model/CNN_decoder_model{epoch}.tar')
        print('\nProceeding to the next epoch...')



def __save_sample__(epoch, x, img_pred):
    path = f'Training_Sneakpeeks/Transformer_Training/'
    try:
        os.makedirs(path)
    except:
        pass
    elements = [x, img_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    for i, element in enumerate(elements):
        element.save(f"{path}{epoch}_{['image', 'image_trans_pred'][i]}.jpg")


train(epochs=500)

# vsn = VideoSegmentationNetwork()

# for i in range(100):
#     input_tensor = torch.randn(BATCH_SIZE, 3, 256, 256)
#     state, throughput0, throughput1 = vsn(x=input_tensor)
#     print(f'Iteration: {i+1}')
#     if state:
#         print(throughput1.shape)
    


# print(f'Input tensor {input_tensor.shape} -> {splitandstack.shape} -> {unstackandmerge.shape}') 
