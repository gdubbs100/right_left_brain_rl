import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, gating_schedule_type, gating_schedule_update, min_right_value, init_right_value):
        self.gating_schedule_type = gating_schedule_type
        assert self.gating_schedule_type in ['addative', 'multiplicative']
        self.gating_schedule_update = gating_schedule_update
        self.right = init_right_value
        self.left = 1 - self.right
        self.min_right_value = min_right_value

    def __call__(self, state, left_latent, right_latent):
        inputs = torch.cat((left_latent, right_latent, state), dim=-1)
        outputs = torch.zeros((*inputs.size()[:-1], 2))
        outputs[...,:1] = self.left
        outputs[...,1:] = self.right
        outputs = outputs.to(device)
        outputs.requires_grad = False
        return outputs[...,:1], outputs[...,1:]
    
    def step(self):

        # update right
        if self.gating_schedule_type == 'addative':
            self.right = np.max([self.right - self.gating_schedule_update, self.min_right_value])
        
        elif self.gating_schedule_type == 'multiplicative':
            self.right = np.max([self.right * self.gating_schedule_update, self.min_right_value])
        
        # update left
        self.left = 1 - self.right


