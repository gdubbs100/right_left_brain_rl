"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])

def value_checker(value_preds):
    if isinstance(value_preds, list):
        return value_preds[0].detach()
    else:
        return value_preds.detach()

### TODO: change the name
class CustomOnlineStorage(object):
    def __init__(self,
                #  args, 
                 num_steps, num_processes,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards
                ):

        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)

        self.latent_dim = latent_dim
        self.latent = []
        # hidden states of RNN (necessary if we want to re-compute embeddings)
        self.hidden_size = hidden_size
        ## TODO: this is why we have double zeros at the start...
        self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)
        # self.hidden_states = torch.zeros(num_steps, num_processes, hidden_size)

        self.beliefs = None
        self.tasks = None
        # next_state will include s_N when state was reset, skipping s_0
        # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        self.next_state = torch.zeros(num_steps, num_processes, state_dim)

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.to_device()

    def to_device(self, device = device):

        self.prev_state = self.prev_state.to(device)
        self.latent = [t.to(device) for t in self.latent]
        self.hidden_states = self.hidden_states.to(device)
        self.next_state = self.next_state.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.done = self.done.to(device)
        self.masks = self.masks.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)

    def insert(self,
               state,
               belief,
               task,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               done,
               hidden_states=None,
               latent = None
               ):
        self.prev_state[self.step + 1].copy_(state)
        self.latent.append(latent.detach().clone())
        self.hidden_states[self.step+1].copy_(hidden_states.detach())

        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.done[self.step + 1].copy_(done)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        ## TODO: should we copy the last state over? this is just an RL2 meta-training thing?
        ## set to torch.zeros_like for now
        self.prev_state[0].copy_(torch.zeros_like(self.prev_state[-1]))
        self.latent = []
        self.hidden_states[0].copy_(torch.zeros_like(self.hidden_states[-1]))
        self.done[0].copy_(torch.zeros_like(self.done[-1]))
        self.masks[0].copy_(torch.zeros_like(self.masks[-1]))
        self.action_log_probs = None

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae, use_proper_time_limits=use_proper_time_limits)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae, use_proper_time_limits):

        if use_proper_time_limits:
            ## don't want to use this at all
            raise NotImplementedError
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_state) * self.num_processes
    
    def before_update(self, policy):
        # this is about building the computation graph during training
        _, action_log_probs, _, = policy.evaluate_actions(self.prev_state[:-1],
                                                         torch.cat(self.latent[:-1]),
                                                         None,
                                                         None,
                                                         self.actions)
        self.action_log_probs = action_log_probs.detach()

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]
            cat_latent = torch.cat(self.latent[:-1])
            latent_batch = cat_latent.reshape(-1, *cat_latent.size()[2:])[indices]
            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, actions_batch, latent_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
            

class BiHemOnlineStorage(object):
    def __init__(self,
                 num_steps, num_processes,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 gate_hidden_size, left_hidden_size, right_hidden_size,
                 gate_latent_dim, left_latent_dim, right_latent_dim,
                 normalise_rewards
                ):

        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)

        self.gate_latent_dim = gate_latent_dim
        self.left_latent_dim = left_latent_dim
        self.right_latent_dim = right_latent_dim
        self.gate_latent = []
        self.left_latent = []
        self.right_latent = []
        # hidden states of RNN (necessary if we want to re-compute embeddings)
        self.gate_hidden_size = gate_hidden_size
        self.left_hidden_size = left_hidden_size
        self.right_hidden_size = right_hidden_size
        self.gate_hidden_states = torch.zeros(num_steps + 1, num_processes, self.gate_hidden_size)
        self.left_hidden_states = torch.zeros(num_steps + 1, num_processes, self.left_hidden_size)
        self.right_hidden_states = torch.zeros(num_steps + 1, num_processes, self.right_hidden_size)

        self.beliefs = None
        self.tasks = None
        # next_state will include s_N when state was reset, skipping s_0
        # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        self.next_state = torch.zeros(num_steps, num_processes, state_dim)

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.left_value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.right_value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.to_device()

    def to_device(self, device = device):

        self.prev_state = self.prev_state.to(device)
        self.gate_latent = [t.to(device) for t in self.gate_latent]
        self.left_latent = [t.to(device) for t in self.left_latent]
        self.right_latent = [t.to(device) for t in self.right_latent]
        self.gate_hidden_states = self.gate_hidden_states.to(device)
        self.left_hidden_states = self.left_hidden_states.to(device)
        self.right_hidden_states = self.right_hidden_states.to(device)
        self.next_state = self.next_state.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.done = self.done.to(device)
        self.masks = self.masks.to(device)
        self.value_preds = self.value_preds.to(device)
        self.left_value_preds = self.left_value_preds.to(device)
        self.right_value_preds = self.right_value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)

    def insert(self,
               state,
               belief,
               task,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               done,
               hidden_states=None,
               latent = None
               ):
        self.prev_state[self.step + 1].copy_(state)

        ## handle latents:
        if isinstance(latent, tuple):
            left_latent = latent[0]
            right_latent = latent[1]
        else:
            raise ValueError
        self.left_latent.append(left_latent.detach().clone())
        self.right_latent.append(right_latent.detach().clone())

        ## handle hidden_states
        if isinstance(hidden_states, tuple):
            left_hidden_states = hidden_states[0].squeeze()
            right_hidden_states = hidden_states[1].squeeze()
        else:
            raise ValueError
        self.left_hidden_states[self.step+1].copy_(left_hidden_states.detach())
        self.right_hidden_states[self.step+1].copy_(right_hidden_states.detach())

        ## rest of the inputs
        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, tuple):
            combined_values = value_preds[0]
            left_values = value_preds[1]
            right_values = value_preds[2]

            self.value_preds[self.step].copy_(value_checker(combined_values))
            self.left_value_preds[self.step].copy_(value_checker(left_values))
            self.right_value_preds[self.step].copy_(value_checker(right_values))
            # if isinstance(value_preds, list):
            #     self.value_preds[self.step].copy_(value_preds[0].detach())
            # else:
            #     self.value_preds[self.step].copy_(value_preds.detach())
        else:
            raise ValueError
        self.masks[self.step + 1].copy_(masks)
        self.done[self.step + 1].copy_(done)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        ## TODO: should we copy the last state over? this is just an RL2 meta-training thing?
        ## set to torch.zeros_like for now
        self.prev_state[0].copy_(torch.zeros_like(self.prev_state[-1]))
        self.gate_latent = []
        self.left_latent = []
        self.right_latent = []
        self.gate_hidden_states[0].copy_(torch.zeros_like(self.gate_hidden_states[-1]))
        self.left_hidden_states[0].copy_(torch.zeros_like(self.left_hidden_states[-1]))
        self.right_hidden_states[0].copy_(torch.zeros_like(self.right_hidden_states[-1]))
        self.done[0].copy_(torch.zeros_like(self.done[-1]))
        self.masks[0].copy_(torch.zeros_like(self.masks[-1]))
        self.action_log_probs = None

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae, use_proper_time_limits=use_proper_time_limits)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae, use_proper_time_limits):

        if use_proper_time_limits:
            raise NotImplementedError
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_state) * self.num_processes
    
    def before_update(self, policy):
        # this is about building the computation graph during training
        gate_latent = torch.cat(self.gate_latent[:-1])
        left_latent = torch.cat(self.left_latent[:-1])
        right_latent = torch.cat(self.right_latent[:-1])
        _, action_log_probs, _, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                         (gate_latent, left_latent, right_latent),
                                                         None,
                                                         None,
                                                         self.actions)
        self.action_log_probs = action_log_probs.detach()

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]

            gate_latent = torch.cat(self.gate_latent[:-1])
            gate_latent_batch = gate_latent.reshape(-1, *gate_latent.size()[2:])[indices]

            left_latent = torch.cat(self.left_latent[:-1])
            left_latent_batch = left_latent.reshape(-1, *left_latent.size()[2:])[indices]

            right_latent = torch.cat(self.right_latent[:-1])
            right_latent_batch = right_latent.reshape(-1, *right_latent.size()[2:])[indices]
            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            left_preds_batch = self.left_value_preds[:-1].reshape(-1, 1)[indices]
            right_preds_batch = self.right_value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, actions_batch, \
                (gate_latent_batch, left_latent_batch, right_latent_batch), \
                (value_preds_batch, left_preds_batch, right_preds_batch), \
                return_batch, old_action_log_probs_batch, adv_targ
            
