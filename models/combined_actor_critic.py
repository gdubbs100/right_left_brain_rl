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
    
class BiCameralActorCritic(nn.Module):

    def __init__(self, left_policy, left_encoder, right_policy, right_encoder):
        super().__init__()
        self.left_actor_critic = ActorCritic(left_policy, left_encoder)
        self.right_actor_critic = ActorCritic(right_policy, right_encoder)
        self.gating_network = None
    
    @property
    def encoder(self):
        return None
    
    @property
    def prior(self):
        return None
    
    @property
    def policy(self):
        return None
    
    # def act(self, state, latent, belief=None, task=None, deterministic=False)
    #     right_action = self.right_actor_critic.policy.act(state, latent, belief, task, deterministic)
    #     left_action = self.left_actor_critic.policy.act(state, latent, belief, task, deterministic)
    #     right_gate_value, left_gate_value = self.gating_network(state, latent, belief, task, deterministic)
    #     return right_gate_value * right_action + left_gate_value * left_action
    def act(self, state, latent, belief=None, task=None, deterministic=False):
        ## should these be right / left latents?
        right_action_mean = self.right_actor.policy.dist.fc_mean(latent)
        left_action_mean = self.left_actor.policy.dist.fc_mean(latent)
        right_gate_value, left_gate_value = self.gating_network(state, latent, belief, task, deterministic)
        combined_features = right_gate_value * right_action_mean + left_gate_value * left_action_mean
        ## figure out how to do this - perhaps create a bihem dist class?
        dist = FixedNormal(combined_features, std)
        actions = dist.sample()
        return actions
    # def get_actor_params(self):
    #     return self.policy.get_actor_params()

    # def get_critic_params(self):
    #     return self.policy.get_critic_params()

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
