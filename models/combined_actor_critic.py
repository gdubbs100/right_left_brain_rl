import torch.nn as nn

class ActorCritic(nn.Module):

    def __init__(self, policy, encoder):
        super().__init__()
        self.policy = policy
        self.encoder = encoder
    
    def get_actor_params(self):
        return self.policy.get_actor_params()

    def get_critic_params(self):
        return self.policy.get_critic_params()

    def forward_actor(self, inputs):
        return self.policy.forward_actor(inputs)

    def forward_critic(self, inputs):
        return self.policy.forward_critic(inputs)
    
    def act(self, state, latent, belief=None, task=None, deterministic = False):
        return self.policy.act(state, latent, belief, task, deterministic)

    def get_value(self, state, latent, belief=None, task=None):
        value, _ = self.policy.forward(state, latent, belief, task)
        return value

    def evaluate_actions(self, state, latent, belief, task, action):
        """Call policy eval, set task, belief to None"""
        return self.policy.evaluate_actions(state, latent, belief, task, action)
    
        ## TODO: what to do about 'sample'? check what this arg is?
    # def forward(self, actions, states, rewards, hidden_state, return_prior=False, sample=True, detach_every=None):
    #     # really want this to take the inputs for the encoder and then output the outputs of the policy
    #     # we only want to get the prior when there are no previous rewards, actions or hidden states
    #     # should only occur at the very start of the continual learning process
    #     if hidden_state is None:
    #         # print('Hidden state is None!!:', hidden_state)
    #         _, latent_mean, latent_logvar, hidden_state = self.encoder.prior(states.shape[1]) # check that this gets the batch size?
    #     else:
    #         _, latent_mean, latent_logvar, hidden_state = self.encoder(actions, states, rewards, hidden_state, return_prior, sample, detach_every)
        
    #     latent_mean = F.relu(latent_mean)
    #     latent_logvar = F.relu(latent_logvar)
    #     latent = torch.cat((latent_mean, latent_logvar), dim=-1).reshape(1, -1)
    #     # none for belief and task
    #     return self.policy(states, latent, None, None), hidden_state, latent
    
    # def prior(self, num_processes):
    #     return self.encoder.prior(num_processes)