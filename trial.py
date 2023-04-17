import torch

# Create a mask tensor of shape (batch_size, sequence_length)
batch_size = 2
sequence_length = 9
n_elements_to_mask = 3
mask = torch.ones((batch_size, sequence_length))

# Set the last 64 elements of each sequence to 0 to mask them
mask[:, -n_elements_to_mask:] = 0

# Create a mask of shape (batch_size, sequence_length, sequence_length) with diagonal elements set to 1
# This mask will attend to everything before the element, but not to itself and anything after it
attention_mask = torch.tril(torch.ones((sequence_length, sequence_length)))

# Repeat the mask tensor for each batch and multiply it with the original mask tensor
# This will create the final attention mask that attends to everything before the element, but not to itself and anything after it
attention_mask = mask.unsqueeze(-1) * attention_mask.unsqueeze(0)

# Convert the attention mask to a float tensor
attention_mask = attention_mask.type(torch.float)

# Apply the attention mask to the latent tensor
latent = torch.randn((batch_size, sequence_length, 4))  # Example latent tensor

masked_latent = latent*attention_mask

# Now you can pass the masked_latent tensor along with the attention_mask tensor to the transformer encoder
