import torch
import torch.nn as nn
import torch.nn.functional as F

## consider adding an encoder to get gating network latent
class GatingNetwork(nn.Module):

    def __init__(self, input_dims):
        super(GatingNetwork, self).__init__()
        ## input dim = state_space + right_latent_dim*2 + left_latent_dim*2
        ## option to use state or latent?
        self.ff = nn.Linear(input_dims, 2)

    def forward(self, state, left_latent, right_latent):

        inputs = torch.cat((left_latent, right_latent, state), dim=-1)
        outputs = F.softmax(self.ff(inputs), dim=-1)
        return outputs[...,:1], outputs[...,1:]

