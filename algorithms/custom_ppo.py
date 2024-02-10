import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import warnings
from utils import custom_helpers as utl


class CustomPPO:
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 max_grad_norm = 0.5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 context_window = None
                 ):
        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.max_grad_norm = max_grad_norm

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss
        self.context_window = context_window

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)

    def update(self, policy_storage):

        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # recompute embeddings (to build computation graph)
        self._recompute_embeddings(policy_storage, sample=False, update_idx=0,
                            detach_every= self.context_window if self.context_window is not None else None)

        # update the normalisation parameters of policy inputs before updating
        # don't think I need this
        # self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:

                state_batch, actions_batch, latent_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = \
                    self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch,
                                                       belief=None, task=None,
                                                       action=actions_batch)
                
                ratio = torch.exp(action_log_probs -
                            old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # zero out the gradients
                self.optimiser.zero_grad()

                # compute policy loss and backprop
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()

                # clip gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                # update
                self.optimiser.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()


                # recompute embeddings (to build computation graph) during updates
                self._recompute_embeddings(policy_storage, sample=False, update_idx=e + 1,
                                             detach_every= self.context_window if self.context_window is not None else None)

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch

    def act(self, state, latent, belief, task, deterministic = False):
        return self.actor_critic.act(state, latent, belief, task, deterministic)
    
    def get_value(self, state, latent, belief, task):
        return self.actor_critic.get_value(state, latent, belief, task)
    
    def get_latent(self, action, state, reward, hidden_state, return_prior = False):
        _, latent_mean, latent_logvar, hidden_state = self.actor_critic.encoder(action, state, reward, hidden_state, return_prior = return_prior)
        latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim = -1)
        ## assume always add non-linearity to latent
        return F.relu(latent[None,:]), hidden_state
    
    def get_prior(self, num_processes):
        _, latent_mean, latent_logvar, hidden_state = self.actor_critic.encoder.prior(num_processes)
        latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim=-1)
        ## assume always add non-linearity to latent
        return F.relu(latent), hidden_state
    
    ## This is a bit inconsistent with the rest of the getting of latents and stuff
    def _recompute_embeddings(self, policy_storage, sample, update_idx, detach_every):
        latent = [policy_storage.latent[0].detach().clone()]
        latent[0].requires_grad = True

        h = policy_storage.hidden_states[0].detach()
        for i in range(policy_storage.actions.shape[0]):
            # reset hidden state of the GRU when we reset the task
            h = self.actor_critic.encoder.reset_hidden(h, policy_storage.done[i])
            # not sure why this is i + 1?
            # h = self.actor_critic.encoder.reset_hidden(h, policy_storage.done[i + 1])

            _, tm, tl, h = self.actor_critic.encoder(
                policy_storage.actions.float()[i:i + 1],
                policy_storage.next_state[i:i + 1],
                policy_storage.rewards_raw[i:i + 1],
                h,
                sample=sample,
                return_prior=False,
                detach_every=detach_every
            )

            ## apply the nonlinearity manually
            latent.append(F.relu(torch.cat((tm, tl), dim = -1)[None,:]))

        if update_idx == 0:
            try:
                assert (torch.cat(policy_storage.latent) - torch.cat(latent)).sum() == 0

            except AssertionError:
                warnings.warn('You are not recomputing the embeddings correctly!')

        
        policy_storage.latent = latent




class BiHemPPO:
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 max_grad_norm = 0.5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 gating_alpha=0,
                 gating_beta=0,
                 context_window = None
                 ):
        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.max_grad_norm = max_grad_norm

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        self.gating_alpha = gating_alpha
        self.gating_beta = gating_beta

        self.context_window = context_window

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)


    def update(self, policy_storage):

        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # recompute embeddings (to build computation graph)
        self._recompute_embeddings(policy_storage, sample=False, update_idx=0,
                            detach_every= self.context_window if self.context_window is not None else None)

        # update the normalisation parameters of policy inputs before updating
        # don't think I need this
        # self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        gating_penalty_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:

                state_batch, actions_batch, latent_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, (left_gate_value, right_gate_value) = \
                    self.actor_critic.evaluate_actions(
                        state = state_batch, latent=latent_batch,
                        belief = None, task = None,
                        action = actions_batch
                    )


                ## calc action loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                ## calc value loss
                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()


                # zero out the gradients
                self.optimiser.zero_grad()

                # compute gating penalty (use logs for stability) (?)
                # gating_penalty = (
                #     np.log(self.gating_beta + 1.0e-5) + \
                #     self.gating_alpha * (torch.log(right_gate_value) - torch.log(left_gate_value))
                # ).mean()

                gating_penalty = (
                    self.gating_beta * \
                    (right_gate_value / (left_gate_value + 1.0e-5))**self.gating_alpha
                ).mean()


                # compute policy loss and backprop
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + gating_penalty

                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()

                # clip gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                # update
                self.optimiser.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                gating_penalty_epoch += gating_penalty.item()
                loss_epoch += loss.item()


                # recompute embeddings (to build computation graph) during updates
                self._recompute_embeddings(policy_storage, sample=False, update_idx=e + 1,
                                             detach_every= self.context_window if self.context_window is not None else None)

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        gating_penalty_epoch /= num_updates
        loss_epoch /= num_updates
        

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, gating_penalty_epoch, loss_epoch

    def act(self, state, latent, belief, task, deterministic = False):
        return self.actor_critic.act(state, latent, belief, task, deterministic)
    
    def get_value(self, state, latent, belief, task):
        return self.actor_critic.get_value(state, latent, belief, task)
    
    def get_latent(self, action, state, reward, hidden_state, return_prior = False):
        left, right = self.actor_critic.encoder(action, state, reward, hidden_state, return_prior = return_prior)
        left_latent = torch.cat((left[0].clone(), left[1].clone()), dim = -1)
        right_latent = torch.cat((right[0].clone(), right[1].clone()), dim = -1)
        ## assume always add non-linearity to latent
        latent = (F.relu(left_latent[None, :]), F.relu(right_latent[None, :]))
        hidden_state = (left[-1], right[-1])
        return latent, hidden_state
    
    def get_prior(self, num_processes):
        left, right = self.actor_critic.prior(num_processes)
        left_latent = torch.cat((left[0].clone(), left[1].clone()), dim = -1)
        right_latent = torch.cat((right[0].clone(), right[1].clone()), dim = -1)
        ## assume always add non-linearity to latent
        latent = (F.relu(left_latent), F.relu(right_latent))
        hidden_state = (left[-1], right[-1])
        return latent, hidden_state
    
    ## This is a bit inconsistent with the rest of the getting of latents and stuff
    def _recompute_embeddings(self, policy_storage, sample, update_idx, detach_every):
        left_latent = [policy_storage.left_latent[0].detach().clone()]
        left_latent[0].requires_grad = True

        right_latent = [policy_storage.right_latent[0].detach().clone()]
        ## we don't want the right latent to have a grad!!
        ## may still need to do more - e.g. freeze the weights of the right network
        # right_latent[0].requires_grad = True

        left_h = policy_storage.left_hidden_states[0].detach()
        right_h = policy_storage.right_hidden_states[0].detach()
        for i in range(policy_storage.actions.shape[0]):
            # reset hidden state of the GRU when we reset the task
            left_h = self.actor_critic.left_actor_critic.encoder.reset_hidden(left_h, policy_storage.done[i])
            right_h = self.actor_critic.right_actor_critic.encoder.reset_hidden(right_h, policy_storage.done[i])
            # not sure why this is i + 1?
            # h = self.actor_critic.encoder.reset_hidden(h, policy_storage.done[i + 1])

            left, right = self.actor_critic.encoder(
                policy_storage.actions.float()[i:i + 1],
                policy_storage.next_state[i:i + 1],
                policy_storage.rewards_raw[i:i + 1],
                (left_h, right_h),
                sample=sample,
                return_prior=False,
                detach_every=detach_every
            )
            left_h, right_h = left[-1], right[-1]


            ## apply the nonlinearity manually
            left_latent.append(F.relu(torch.cat((left[0], left[1]), dim = -1)[None,:]))
            right_latent.append(F.relu(torch.cat((right[0], right[1]), dim = -1)[None,:]))

        if update_idx == 0:
            try:
                assert (torch.cat(policy_storage.left_latent) - torch.cat(left_latent)).sum() == 0
                assert (torch.cat(policy_storage.right_latent) - torch.cat(right_latent)).sum() == 0

            except AssertionError:
                warnings.warn('You are not recomputing the embeddings correctly!')
        
        policy_storage.left_latent = left_latent
        ## probably don't need to do this as we are not attaching gradients to the right...
        policy_storage.right_latent = right_latent


    # def calculate_loss_by_hemisphere(
    #         self, 
    #         values,
    #         action_log_probs,
    #         return_batch, 
    #         old_action_log_probs_batch, 
    #         value_preds_batch, 
    #         adv_targ
    #     ):


    #     ratio = torch.exp(action_log_probs -
    #                 old_action_log_probs_batch)
    #     surr1 = ratio * adv_targ
    #     surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
    #     action_loss = -torch.min(surr1, surr2)#.mean()

    #     if self.use_huber_loss and self.use_clipped_value_loss:
    #         value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
    #                                                                                     self.clip_param)
    #         value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
    #         value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
    #         value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)#.mean()
    #     elif self.use_huber_loss:
    #         value_loss = F.smooth_l1_loss(values, return_batch)
    #     elif self.use_clipped_value_loss:
    #         value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
    #                                                                                     self.clip_param)
    #         value_losses = (values - return_batch).pow(2)
    #         value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
    #         value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)#.mean()
    #     else:
    #         value_loss = 0.5 * (return_batch - values).pow(2)#.mean()

    #     ## not aggregated!
    #     return action_loss, value_loss

### stuff for bi-hemisphere log probs!
                    # (left_values, left_action_log_probs, left_entropy), \
                # (right_values, right_action_log_probs, right_entropy) = \
                #     self.actor_critic.evaluate_actions_by_hemisphere(state=state_batch, latent=latent_batch,
                #                                        belief=None, task=None,
                #                                        action=actions_batch)
                
                # ## call gating network to create computation graph for gating values
                # (left_gate_value, right_gate_value) = \
                #     self.actor_critic.gating_network(state_batch, latent_batch[0], latent_batch[1])
                ## calculate loss by hemispheres
                # left_action_loss, left_value_loss = self.calculate_loss_by_hemisphere(
                #     left_values, left_action_log_probs, return_batch,
                #     old_action_log_probs_batch, value_preds_batch, adv_targ
                # )

                # right_action_loss, right_value_loss = self.calculate_loss_by_hemisphere(
                #     right_values, right_action_log_probs, return_batch,
                #     old_action_log_probs_batch, value_preds_batch, adv_targ
                # )

                ## use MOE loss function - no losses so far have been aggregated prior to weighting
                # value_loss = \
                #     (left_gate_value * left_value_loss + right_gate_value + right_value_loss).mean()
                
                # action_loss = \
                #     (left_gate_value * left_action_loss + right_gate_value + right_action_loss).mean()
                
                # dist_entropy = (left_gate_value * left_entropy + right_gate_value * right_entropy).mean()


