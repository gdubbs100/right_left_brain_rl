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
        
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        ## not sure what this does, taken from encoder
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, action, state, left_reward_err, right_reward_err, left_gate_value, right_gate_value, hidden_state):

        if self.take_action and self.take_state:
            x = torch.cat(
                (action, state, left_reward_err, right_reward_err, left_gate_value, right_gate_value, hidden_state), 
                dim = -1)
        elif self.take_action and not self.take_state:
            x = torch.cat(
                (action, left_reward_err, right_reward_err, left_gate_value, right_gate_value, hidden_state), 
                dim = -1)
        elif not self.take_action and self.take_state:
            x = torch.cat(
                (state, left_reward_err, right_reward_err, left_gate_value, right_gate_value, hidden_state), 
                dim = -1)
        else:
            x = torch.cat(
                (left_reward_err, right_reward_err, left_gate_value, right_gate_value, hidden_state), 
                dim = -1)

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

    

class EncoderGatingNetwork(nn.Module):

    def __init__(self, take_action, take_state, dim_action, dim_state):
        super().__init__()
        self.take_action = take_action
        self.take_state = take_state
        self.dim_action = dim_action
        self.dim_state = dim_state

        ## TODO: incorporate gating values?
        ## input dim should be:
        # left_error(1) + right_error(1) + action_dim (4) + state_dim (40) = 46
        # +gateing_vals(2) = 48
        self.latent_dim = 4 # latent is input dim - named latent to be consistent with rest of code
        if self.take_action:
            self.latent_dim += self.dim_action
        if self.take_state:
            self.latent_dim += self.dim_state
        self.hidden_size = self.latent_dim
        self.latent_dim *= 2 # hidden size is input
        self.encoder = GatingEncoder(
            input_dim=self.latent_dim, 
            output_dim=self.latent_dim, 
            hidden_size=self.hidden_size, 
            take_action=self.take_action, 
            take_state=self.take_state
        )
        self.gate = nn.Linear(self.latent_dim, 2)
    
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

        ## dummy variables
        self.hidden_size = 1
        self.latent_dim = 1

    def gating_function(self, gate_latent):
        # inputs = torch.cat((left_latent, right_latent, state), dim=-1)
        outputs = torch.zeros((*gate_latent.size()[:-1], 2))
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

    ## dummy functions
    def encoder(self, action, state, left_reward_err, right_reward_err, left_gate_value, right_gate_value, hidden_state):
        return torch.zeros_like(hidden_state).to(device), torch.zeros_like(hidden_state).to(device)
    
    def prior(self, num_processes):
        return torch.zeros((1, num_processes, self.latent_dim)).to(device), torch.zeros((1, num_processes, self.hidden_size)).to(device)
    
    def reset_hidden(self, hidden_state, done):
        return torch.zeros_like(hidden_state).to(device)
    
    


