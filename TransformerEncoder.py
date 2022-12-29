import torch
import torch.nn as nn
import math


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, input_dim)

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropoutLayer = nn.Dropout(0.1) 
  

    def __reshape_to_batches__(self, x):
        ''' 
            Here, we divide the size of the input vector by the total number of heads.
            New vectors will be created, each with the resulting size of input_vector//heads
            They will be projected along a new axis, on the third index
            It will be than moved to the first index.
            It than will be projected into the axis with the batch-dimension

            The batch size will be up-scaled by the number of heads we have and the input_dimension will be downscaled by it 
        '''
        batch_size, seq_len, input_dim = x.size()
        sub_dim = input_dim // self.num_heads
        x = x.reshape(batch_size, seq_len, self.num_heads, sub_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * self.num_heads, seq_len, sub_dim)
        return x
    def __reshape_from_batches__(self, x):
        batch_size, seq_len, input_dim = x.size()
        batch_size //= self.num_heads
        out_dim = input_dim * self.num_heads
        x = x.reshape(batch_size, self.num_heads, seq_len, input_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, seq_len, out_dim)
        return x



    def attention(self, q, k, v, mask):
        # computing the dot product of the query and the key and scaling it by input dimension^-1
        dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input_dim)

        # masking the dot product
        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, -1e9)

        # the softmax of the dot products
        weights = self.dropoutLayer(self.softmax(dot_product))

        # multiplying the scaled dot product of queries and keys with the values
        output = torch.matmul(weights, v)
        return output




    def forward(self, q, k, v, mask=None):
        # converting the input into query, key and the value
        q, k, v = self.query_layer(q), self.key_layer(k), self.value_layer(v)
        q, k, v = self.activation(q), self.activation(k), self.activation(v)

        #splitting it into number of heads now 
        q = self.__reshape_to_batches__(q)
        k = self.__reshape_to_batches__(k)
        v = self.__reshape_to_batches__(v)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        #attention function
        attention = self.attention(q, k, v, mask)

        #reshaping from the batch dimension (similar to concatinating)
        attention = self.__reshape_from_batches__(attention)

        #finally feeding it through the output layer
        output = self.output_layer(attention)

        return output



class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        if input_dim % num_heads != 0:
            raise ValueError(f"Input dimension must be divisible by number of heads. Here, Input dimension = {input_dim} is not divisible by number of heads = {num_heads}")

        # create the custom multi-head attention layers
        self.attention_layers = nn.ModuleList([MultiheadAttention(input_dim, num_heads) for _ in range(num_layers)])
        self.feedforward_layers = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)) for _ in range(num_layers)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])

    def forward(self, input, mask):
        # input is of shape (batch_size, seq_len, input_dim)
        # mask is of shape (batch_size, seq_len)

        # apply the attention and feedforward layers in a loop
        for i in range(self.num_layers):
            input = self.attention_layers[i](input, input, input, mask)
            input = self.layer_norm[i](input)
            input = self.feedforward_layers[i](input)
            input = self.layer_norm[i](input)
            input = input * math.sqrt(0.5)
            input = nn.Dropout(self.dropout)(input)

        # return the output of the encoder
        return input




# transenc = TransformerEncoder(input_dim=4096, hidden_dim=4096, num_layers=1, num_heads=4, dropout=0.1)
# inp = torch.rand(size=(4, 128, 4096))
# out = transenc(inp, mask=None)
# print(inp.shape, out.shape)