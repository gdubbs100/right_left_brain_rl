import torch
import torch.nn as nn
import numpy as np
from models.policy import FixedNormal
from models.gating_network import GatingNetwork, StepGatingNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
class BiHemActorCritic(nn.Module):

    def __init__(
            self, 
            left_policy, 
            left_encoder, 
            right_policy, 
            right_encoder, 
            dim_state,
            dim_action, 
            init_std, 
            use_gating_schedule = False,
            gating_schedule_type = None,
            gating_schedule_update = None,
            min_right_value = None,
            init_right_value = None
        ):
        super().__init__()
        self.left_actor_critic = ActorCritic(left_policy, left_encoder)
        self.right_actor_critic = ActorCritic(right_policy, right_encoder)

        if use_gating_schedule:
            self.gating_network = StepGatingNetwork(
                gating_schedule_type=gating_schedule_type, 
                gating_schedule_update=gating_schedule_update, 
                min_right_value=min_right_value, 
                init_right_value=init_right_value
            )
        else:
            self.gating_network = GatingNetwork(
                dim_state + left_encoder.latent_dim * 2 + right_encoder.latent_dim * 2
            )
        self.logstd = nn.Parameter(np.log(torch.zeros(dim_action) + init_std))
        self.min_std = torch.tensor([1.0e-6]).to(device)
        # self.max_std = torch.tensor([1.0e6]).to(device)
        # self.std = torch.tensor([init_std]).to(device)
    
    def encoder(self, action, state, reward, hidden_state, return_prior = False, sample = False, detach_every = None):
        if isinstance(hidden_state, tuple):
            left_hidden_state = hidden_state[0]
            right_hidden_state = hidden_state[1]
        else:
            raise ValueError

        _, left_latent_mean, left_latent_logvar, left_hidden_state = self.left_actor_critic.encoder(
            action, 
            state, 
            reward, 
            left_hidden_state, 
            return_prior = return_prior,
            sample = sample,
            detach_every=detach_every
        )
        _, right_latent_mean, right_latent_logvar, right_hidden_state = self.right_actor_critic.encoder(
            action, 
            state, 
            reward, 
            right_hidden_state, 
            return_prior = return_prior,
            sample = sample,
            detach_every=detach_every
        )
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
        left_value, left_actor_features = self.left_actor_critic.policy(
            state=state, latent=left_latent, belief=belief, task=task
        )
        left_action_mean = self.left_actor_critic.policy.dist.fc_mean(left_actor_features)
        
        # get right hemisphere input to distribution        
        right_value, right_actor_features = self.right_actor_critic.policy(
            state=state, latent=right_latent, belief=belief, task=task
        )
        right_action_mean = self.right_actor_critic.policy.dist.fc_mean(right_actor_features)
        
        # maybe gate network should take task? take combined latents and current state?
        left_gate_value, right_gate_value = self.gating_network(state, left_latent, right_latent)

        # combine action and value estimate
        combined_action_means = left_gate_value * left_action_mean + right_gate_value * right_action_mean
        combined_values = left_gate_value * left_value + right_gate_value * right_value
        
        # use 'self.std' for now
        std = torch.max(self.min_std, self.logstd.exp())
        dist = FixedNormal(combined_action_means, std)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample() ## assumes not deterministic

        return combined_values, actions, dist, (left_gate_value, right_gate_value)
    
    def act(self, state, latent, belief=None, task=None, deterministic=False):
        values, actions, _, gating_values = self.policy(state, latent, None, None, deterministic = deterministic)
        return values, actions, gating_values

    def get_value(self, state, latent, belief=None, task=None):
        value, _, _, _ = self.policy(state, latent, belief, task)
        return value
    
    def evaluate_actions(self, state, latent, belief, task, action):
        """
        Gets the distribution of the entire network
        """
        values, _, dist, gating_values = self.policy(state, latent, None, None)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy, gating_values

    def evaluate_actions_by_hemisphere(self, state, latent, belief, task, action):
        """Call policy eval, set task, belief to None"""
        if isinstance(latent, tuple):
            left_latent = latent[0]
            right_latent = latent[1]
        else:
            raise ValueError

        ## get left v, log_probs, entropy
        left_value, left_logprobs, left_entropy = self.left_actor_critic\
            .evaluate_actions(state, left_latent, None, None, action)
        ## get right value, log_probs, entropy
        right_value, right_logprobs, right_entropy = self.right_actor_critic\
            .evaluate_actions(state, right_latent, None, None, action)

        return (left_value, left_logprobs, left_entropy), (right_value, right_logprobs, right_entropy)
