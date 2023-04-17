import torch
import random

# # Create a latent tensor of shape (batch_size, sequence_length, embedding_dimension)
# latent = torch.randn((2, 9, 4))

# # Get the batch size and sequence length from the latent tensor
# batch_size, sequence_length, _ = latent.size()

# # Create a mask tensor of shape (sequence_length, sequence_length) initialized with ones
# mask = torch.ones((sequence_length, sequence_length))

# # Set the upper triangle elements (including the diagonal) to False for the last 3 elements of the sequence
# mask[-3:, :] = 0

# # Expand the mask tensor to shape (batch_size, sequence_length, sequence_length) 
# mask = mask.unsqueeze(0).expand(batch_size, sequence_length, sequence_length).bool()

# # Convert the mask tensor to a float tensor and negate it to create the attention mask
# attention_mask = (~mask).type(torch.float)

# # Apply the attention mask to the latent tensor
# latent = latent * attention_mask.unsqueeze(-1)

# # Print the masked latent tensor
# print(latent)



def get_last_seq_pad_mask(second_seq_len=512, third_seq_len=512, bs=1):
        last_sequence_pad_mask = torch.zeros(second_seq_len+third_seq_len, dtype=torch.bool)
        last_sequence_pad_mask = last_sequence_pad_mask.repeat(bs, 1)
        return last_sequence_pad_mask


def get_mask_seq_cat(first_seq_len=130, second_seq_len=128):

        def add_additional_top_mask(mat_mask, mp):
            for seq in range(mat_mask.shape[0]):
                position = [random.randint(1,first_seq_len-2) for _ in range(mp)]

                for pos in position:
                    mat_mask[seq][pos] = float('-inf')
                       
            
            return mat_mask
        
        def add_additional_bottom_mask(mat_mask, mp):
            for seq in range(mat_mask.shape[0]):
                position = [random.randint(1, first_seq_len-2) for _ in range(mp)]
                position += [random.randint(first_seq_len+1, first_seq_len+second_seq_len-2) for _ in range(mp)]
                #For masking the text part code needs to here
                
                for pos in position:
                    mat_mask[seq][pos] = float('-inf')
                      
            return mat_mask


        second_mask = torch.triu(torch.full((second_seq_len, second_seq_len), float('-inf')), diagonal=1)
        first_mask = torch.zeros(second_seq_len, first_seq_len)

        bottom_mask = torch.cat([first_mask, second_mask], axis=-1)

        # top_mask = torch.clone(bottom_mask[0])
        # top_mask[first_seq_len] = float('-inf')
        # top_mask = top_mask.unsqueeze(0).repeat(first_seq_len,1)

        top_mask = bottom_mask[0].unsqueeze(0).repeat(first_seq_len,1)
        mask_ua = torch.cat([top_mask, bottom_mask], axis=0)
        return mask_ua

        # # Additional masking attention in image & text segment
        # bottom_mask = add_additional_bottom_mask(bottom_mask, int(0.1 * first_seq_len))
        # # top_mask = add_additional_top_mask(top_mask, int(0.1 * first_seq_len))
        # mask_a = torch.cat([top_mask, bottom_mask], axis=0)

        # if random.random() < 0.5:
        #     return mask_a
        # return mask_ua

mask = get_mask_seq_cat(256, 64)
print(mask)