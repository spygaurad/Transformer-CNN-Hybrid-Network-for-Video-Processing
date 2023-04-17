import torch

# Create a latent tensor of shape (batch_size, sequence_length, embedding_dimension)
latent = torch.randn((2, 9, 4))

# Get the batch size and sequence length from the latent tensor
batch_size, sequence_length, _ = latent.size()

# Create a mask tensor of shape (sequence_length, sequence_length) initialized with ones
mask = torch.ones((sequence_length, sequence_length))

# Set the upper triangle elements (including the diagonal) to False to prevent attending to itself and elements after it
mask = mask.triu_(1).type(torch.bool)

# Expand the mask tensor to shape (batch_size, sequence_length, sequence_length)
mask = mask.unsqueeze(0).expand(batch_size, sequence_length, sequence_length)

# Convert the mask tensor to a float tensor and negate it to create the attention mask
attention_mask = (~mask).type(torch.float)

# Apply the attention mask to the latent tensor
latent = latent * attention_mask.unsqueeze(-1)

# Print the masked latent tensor
print(latent)
