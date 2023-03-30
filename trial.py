import torch

batch_size = 1
sequence_length = 5
hidden_size = 3

# Create a mask tensor of shape (batch_size, sequence_length)
mask = torch.ones((batch_size, sequence_length), dtype=torch.bool)

# Set the last 64 elements of each sequence to False to mask them
mask[:, -3:] = False

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