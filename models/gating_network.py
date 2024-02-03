import torch
import torch.nn as nn

class GatingNetwork(nn.Module):

    def __init__(self, input_dims):
        super(GatingNetwork, self).__init__()
        ## what will the input dim be? state_space + right_latent_dim*2 + left_latent_dim*2
        self.ff = nn.Linear(input_dims, 2)

    def forward(self, state, latent):
        if isinstance(latent, tuple):
            left_latent = latent[0]
            right_latent = latent[1]
        else:
            raise ValueError
        inputs = torch.cat((left_latent, right_latent, state), dim=-1)
        return torch.sigmoid(self.ff(inputs))

