import torch
import torch.nn as nn
from models.policy import FixedNormal
from models.gating_network import GatingNetwork

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
        self.gating_network = GatingNetwork(40 + left_encoder.latent_dim * 2 + right_encoder.latent_dim * 2)

        # placeholder for now
        self.std = torch.tensor([0.5])
    
    def encoder(self, action, state, reward, hidden_state):
        if isinstance(hidden_state, tuple):
            left_hidden_state = hidden_state[0]
            right_hidden_state = hidden_state[1]
        else:
            raise ValueError

        _, left_latent_mean, left_latent_logvar, left_hidden_state = self.left_actor_critic.encoder(action, state, reward, left_hidden_state)
        _, right_latent_mean, right_latent_logvar, right_hidden_state = self.right_actor_critic.encoder(action, state, reward, right_hidden_state)
        ## TODO: add gating encoder?
        
        return (left_latent_mean, left_latent_logvar, left_hidden_state), (right_latent_mean, right_latent_logvar, right_hidden_state)
    
    def prior(self, num_processes):
        _, left_latent_mean, left_latent_logvar, left_hidden_state = self.left_actor_critic.encoder.prior(num_processes)
        _, right_latent_mean, right_latent_logvar, right_hidden_state = self.right_actor_critic.encoder.prior(num_processes)
        ## TODO: add gating encoder?
        
        return (left_latent_mean, left_latent_logvar, left_hidden_state), (right_latent_mean, right_latent_logvar, right_hidden_state)
    

    def policy(self, state, latent, belief=None, task=None, deterministic=False):

        if isinstance(latent, tuple):
            left_latent = latent[0]
            right_latent = latent[1]
        else:
            raise ValueError
        
        # get left hemisphere input to distribution
        left_value, left_actor_features = self.left_actor_critic.policy(state=state, latent=left_latent, belief=belief, task=task)
        left_action_mean = self.left_actor_critic.policy.dist.fc_mean(left_actor_features)
        
        # get right hemisphere input to distribution        
        right_value, right_actor_features = self.right_actor_critic.policy(state=state, latent=right_latent, belief=belief, task=task)
        right_action_mean = self.right_actor_critic.policy.dist.fc_mean(right_actor_features)
        
        # maybe gate network should take task? take combined latents and current state?
        right_gate_value, left_gate_value = self.gating_network(state, latent)

        # combine action and value estimate
        combined_action_means = left_gate_value * left_action_mean + right_gate_value * right_action_mean
        combined_values = left_gate_value * left_value + right_gate_value * right_value
        
        # use 'self.std' for now
        dist = FixedNormal(combined_action_means, self.std)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample() ## assumes not deterministic
        return combined_values, actions, dist
    
    def act(self, state, latent, belief=None, task=None, deterministic=False):
        values, actions, _ = self.policy(state, latent, None, None, deterministic = deterministic)
        return values, actions

    def get_value(self, state, latent, belief=None, task=None):
        value, _, _ = self.policy(state, latent, belief, task)
        return value
    
    def evaluate_actions(self, state, latent, belief, task, action):
        """
        Gets the distribution of the entire network
        """
        values, _, dist = self.policy(state, latent, None, None)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy

    def evaluate_actions_by_hemisphere(self, state, latent, belief, task, action):
        """Call policy eval, set task, belief to None"""

        ## get left v, log_probs, entropy
        left_value, left_logprobs, left_entropy = self.left_actor_critic.evaluate_actions(state, latent, None, None, action)
        ## get right value, log_probs, entropy
        right_value, right_logprobs, right_entropy = self.right_actor_critic.evaluate_actions(state, latent, None, None, action)

        return (left_value, left_logprobs, left_entropy), (right_value, right_logprobs, right_entropy)
