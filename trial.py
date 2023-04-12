import torch

# Create a mask tensor of shape (batch_size, sequence_length)
batch_size = 4
sequence_length = 9
n_elements_to_mask = 3
mask = torch.ones((batch_size, sequence_length))

# Set the last 64 elements of each sequence to False to mask them
mask[:, -n_elements_to_mask:] = 0

# Create a causal mask tensor for the last 64 elements
causal_mask = torch.tril(torch.ones((n_elements_to_mask, n_elements_to_mask)))

# Expand the causal mask tensor to shape (batch_size, n_elements_to_mask, n_elements_to_mask)
causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

# Concatenate the two mask tensors along the sequence length dimension
mask = torch.cat((mask.unsqueeze(-1), causal_mask), dim=-1)

# Convert the mask tensor to a float tensor and negate it to create the attention mask
attention_mask = (~mask).type(torch.float)

# Apply the attention mask to the latent tensor
latent = torch.randn((batch_size, sequence_length, 512))  # Example latent tensor
masked_latent = latent * attention_mask.unsqueeze(-1)

# Now you can pass the masked_latent tensor along with the attention_mask tensor to the transformer encoder
