import torch

batch_size = 1
seq_len = 5
num_heads = 8

# create a matrix of ones for the first 256 elements
attn_mask = torch.ones(seq_len, seq_len)

# create a lower triangular matrix of ones for the last 64 elements
for i in range(seq_len - 2, seq_len):
    attn_mask[i, :i] = 1
    attn_mask[i, i+1:] = 0

# repeat the attention mask for each sample in the batch
attn_mask = attn_mask.unsqueeze(0).repeat(batch_size, 1, 1)

# create a tensor of zeros for the attention weights
# we'll use this tensor to fill in the diagonal elements of the attention matrix
attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len)

# fill in the diagonal elements of the attention matrix with the attention mask
for i in range(num_heads):
    attn_weights[:, i, :, :] = attn_mask

# transpose the attention weights tensor to match the expected input of the Transformer
attn_weights = attn_weights.transpose(1, 2)
print(attn_weights)import torch

batch_size = 8
sequence_length = 320
hidden_size = 512

# Create a mask tensor of shape (batch_size, sequence_length)
mask = torch.ones((batch_size, sequence_length), dtype=torch.bool)

# Set the last 64 elements of each sequence to False to mask them
mask[:, -64:] = False

# Create a causal mask tensor for the last 64 elements
causal_mask = torch.tril(torch.ones((64, 64), dtype=torch.bool))

# Repeat the causal mask tensor for each batch
causal_mask = causal_mask.repeat(batch_size, 1, 1)

# Concatenate the two mask tensors along the sequence length dimension
mask = torch.cat((mask.unsqueeze(-1), causal_mask.unsqueeze(-1)), dim=-1)

# Expand the mask tensor to shape (batch_size, 1, sequence_length, sequence_length+64)
mask = mask.unsqueeze(1).expand(batch_size, 1, sequence_length, sequence_length+64)

# Convert the mask tensor to a float tensor and negate it to create the attention mask
attention_mask = (~mask).type(torch.float)