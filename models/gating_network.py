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
    
    ## can you update this to take arbitrary inputs?
    def forward(self, state, left_latent, right_latent):

        inputs = torch.cat((left_latent, right_latent, state), dim=-1)
        outputs = F.softmax(self.ff(inputs), dim=-1)
        return outputs[...,:1], outputs[...,1:]

class GatingEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size, take_action, take_state):
        super(GatingEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.take_action = take_action
        self.take_state = take_state

        self.gru = nn.GRU(input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=1,
                )
        
        self.fc = nn.Linear(self.input_dim, self.output_dim)

        ## not sure what this does, taken from encoder
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, action, state, left_reward_err, right_reward_err, hidden_state):

        if self.take_action and self.take_state:
            x = torch.cat((action, state, left_reward_err, right_reward_err, hidden_state), dim = -1)
        elif self.take_action and not self.take_state:
            x = torch.cat((action, left_reward_err, right_reward_err, hidden_state), dim = -1)
        elif not self.take_action and self.take_state:
            x = torch.cat((state, left_reward_err, right_reward_err, hidden_state), dim = -1)
        else:
            x = torch.cat((left_reward_err, right_reward_err, hidden_state), dim = -1)

        hidden_state, _ = self.gru(x)
        # not quite sure why we do this - based on encoder
        _hidden_state = hidden_state.clone()
        latent = self.fc(_hidden_state) ## to be consistent with other code, apply Relu in agent
        return latent, hidden_state
    
    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size):

        # we start out with a hidden state of zero
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        ## for consistency - apply relu in agent
        latent = self.fc(hidden_state)

        return latent, hidden_state

    

class EncoderGatingNetwork():

    def __init__(self, input_dim, output_dim, hidden_size, take_action, take_state):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.take_action = take_action
        self.take_state = take_state
        self.encoder = GatingEncoder(
            self.input_dim, self.output_dim, self.hidden_size, self.take_action, self.take_state
        )
        self.gate = nn.Linear(self.output_dim, 2)
    
    def prior(self, batch_size):
        return self.encoder.prior(batch_size)
    
    def reset_hidden(self, hidden_state, done):
        return self.encoder.reset_hidden(hidden_state, done)
    
    def gating_function(self, x):
        outputs = F.softmax(self.gate(x), dim=-1)
        return outputs[...,:1], outputs[...,1:]
    

class StepGatingNetwork:

    def __init__(self, gating_schedule_type, gating_schedule_update, min_right_value, init_right_value):
        self.gating_schedule_type = gating_schedule_type
        assert self.gating_schedule_type in ['addative', 'multiplicative']
        self.gating_schedule_update = gating_schedule_update
        self.right = init_right_value
        self.left = 1 - self.right
        self.min_right_value = min_right_value

    def gating_function(self, state, left_latent, right_latent):
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


