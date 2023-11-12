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

### TODO: change the name
class CustomOnlineStorage(object):
    def __init__(self,
                #  args, 
                 num_steps, num_processes,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards):

        # self.args = args
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
        # self.latent_samples = []
        # self.latent_mean = []
        # self.latent_logvar = []
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
        # if self.args.pass_latent_to_policy:
        #     # latent variables (of VAE)
        #     self.latent_dim = latent_dim
        #     self.latent_samples = []
        #     self.latent_mean = []
        #     self.latent_logvar = []
        #     # hidden states of RNN (necessary if we want to re-compute embeddings)
        #     self.hidden_size = hidden_size
        #     self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)
        #     # next_state will include s_N when state was reset, skipping s_0
        #     # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        #     self.next_state = torch.zeros(num_steps, num_processes, state_dim)

        # else:
        #     self.latent_mean = None
        #     self.latent_logvar = None
        #     self.latent_samples = None
        # if self.args.pass_belief_to_policy:
        #     self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        # else:
        #     self.beliefs = None
        # if self.args.pass_task_to_policy:
        #     self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim)
        # else:
        #     self.tasks = None

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        ## for metaworld its not clear if this is required
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

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
        # self.latent_samples = [t.to(device) for t in self.latent_samples]
        # self.latent_mean = [t.to(device) for t in self.latent_mean]
        # self.latent_logvar = [t.to(device) for t in self.latent_logvar]
        self.hidden_states = self.hidden_states.to(device)
        self.next_state = self.next_state.to(device)


        # if self.args.pass_state_to_policy:
        #     self.prev_state = self.prev_state.to(device)
        # if self.args.pass_latent_to_policy:
        #     self.latent_samples = [t.to(device) for t in self.latent_samples]
        #     self.latent_mean = [t.to(device) for t in self.latent_mean]
        #     self.latent_logvar = [t.to(device) for t in self.latent_logvar]
        #     self.hidden_states = self.hidden_states.to(device)
        #     self.next_state = self.next_state.to(device)
        # if self.args.pass_belief_to_policy:
        #     self.beliefs = self.beliefs.to(device)
        # if self.args.pass_task_to_policy:
        #     self.tasks = self.tasks.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.done = self.done.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
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
               bad_masks,
               done,
               hidden_states=None,
               latent = None
               ):
        self.prev_state[self.step + 1].copy_(state)
        self.latent.append(latent.detach().clone())
        ##TODO: what is going on here? why step+1?
        self.hidden_states[self.step+1].copy_(hidden_states.detach())

        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
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
        self.bad_masks[0].copy_(torch.zeros_like(self.bad_masks[-1]))
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
### TODO: update as appropriate
    def before_update(self, policy):
        # this is about building the computation graph during training
        # latent = utl.get_latent_for_policy(self.args,
        #                                    latent_sample=torch.stack(
        #                                        self.latent_samples[:-1]) if self.latent_samples is not None else None,
        #                                    latent_mean=torch.stack(
        #                                        self.latent_mean[:-1]) if self.latent_mean is not None else None,
        #                                    latent_logvar=torch.stack(
        #                                        self.latent_logvar[:-1]) if self.latent_mean is not None else None)
        # might need to update this for combined policy /

        _, action_log_probs, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                         torch.stack(self.latent[:-1]),
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
            
    def recurrent_generator(self, advantages, num_mini_batch):
        # num_processes = self.rewards.size(1)
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        # batch_size = num_processes * num_steps
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
