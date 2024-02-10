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

class StepGatingNetwork:

    def __init__(self, num_steps, update_size, left_init = 0.01):
        self.num_steps = num_steps
        self.update_size = update_size
        self.left = left_init
        self.right = 1 - left_init

    def __call__(self, state, left_latent, right_latent):
        inputs = torch.cat((left_latent, right_latent, state), dim=-1)
        outputs = torch.zeros((*inputs.size()[:-1], 2))
        outputs[...,:1] = self.left
        outputs[...,1:] = self.right
        return outputs[...,:1], outputs[...,1:]
    
    def step(self):
        ## apply a stepping function for gating network values
        return None


