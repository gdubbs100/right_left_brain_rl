import os
import time

import gym
import numpy as np
import torch

from algorithms.custom_ppo import CustomPPO
from algorithms.custom_storage import CustomOnlineStorage
# from environments.parallel_envs import make_vec_envs
# from models.policy import Policy
# from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.custom_logger import CustomLogger
from environments.custom_env_utils import prepare_parallel_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ContinualLearner:
    """
    Continual learning class - handles training process for continual learning
    """
    def __init__(self, seed, envs, agent, num_processes, rollout_len, steps_per_env, logger):

        # self.args = args
        ## TODO: set a seed, look at below function
        utl.seed(seed, False)

        ## initialise the envs
        ## TODO: initialise envs in he init
        self.envs = envs
        self.envs = prepare_parallel_envs(
            envs,
            steps_per_env,
            num_processes,
            device
        )

        # set params for runs
        self.num_processes = num_processes
        # steps_per_env = 2000 # might be passed to envs during env creation
        self.rollout_len = rollout_len

        # network
        self.agent = agent

        self.storage = CustomOnlineStorage(
                    self.rollout_len, 
                    self.num_processes, 
                    self.envs.observation_space.shape[0]+1, 
                    0, # what's this? 
                    0, # what's this?
                    self.envs.action_space, 
                    self.agent.actor_critic.encoder.hidden_size, 
                    self.agent.actor_critic.encoder.latent_dim, 
                    False # what's this?
                )
        
        ## TODO: add a logger
        self.logger = logger

        # # calculate number of updates and keep count of frames/iterations
        # self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        # self.frames = 0
        # self.iter_idx = -1

        # # initialise tensorboard logger
        # # self.logger = TBLogger(self.args, self.args.exp_label)

    def train(self):
        """ Main Training loop """
        start_time = time.time() # use this in logger?

        ## TODO: should this be some sort of assert? (perhaps where args are established)
        # print(
        #     steps_per_env % num_processes * rollout_len == 0,
        #     steps_per_env - num_processes * rollout_len >= 0
        # )
        
        ## TODO: replace res with tensorboard
        res = dict()
        eps = 0

        # steps limit is parameter for whole continual env
        while self.envs.get_env_attr('cur_step') < self.envs.get_env_attr('steps_limit'):

            step = 0
            obs = self.envs.reset() # we reset all at once as metaworld is time limited
            current_task = self.envs.get_env_attr("cur_seq_idx") # perhaps sort out dictionary mapping for name / task id
            # episode_reward = 0
            episode_reward = []
            done = [False for _ in range(self.num_processes)]

            ## TODO: determine how frequently to get prior - do at start of each episode for now
            with torch.no_grad():
                _, latent_mean, latent_logvar, hidden_state = self.agent.actor_critic.encoder.prior(self.num_processes)

                assert len(self.storage.latent) == 0  # make sure we emptied buffers

                self.storage.hidden_states[:1].copy_(hidden_state)
                latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim=-1)
                self.storage.latent.append(latent)

            while not all(done):
                value, action = self.agent.act(obs, latent, None, None)
                next_obs, reward, done, info = self.envs.step(action)
                assert all(done) == any(done), "Metaworld envs should all end simultaneously"

                obs = next_obs

                ## TODO: review this
                # episode_reward += #reward.sum() / self.num_processes
                episode_reward.append(reward)
                ## TODO: do I even need masks? - check how advantages are calculated
                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if _done else [1.0] for _done in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                # don't care for metaworld
                bad_masks = torch.FloatTensor([[0.0] for _done in done]).to(device)

                # TODO: check if this needs to be done - how is it done in other loops
                # reset hidden state if done
                if all(done):
                    hidden_state = self.agent.actor_critic.encoder.reset_hidden(hidden_state, masks_done)

                _, latent_mean, latent_logvar, hidden_state = self.agent.actor_critic.encoder(action, obs, reward, hidden_state, return_prior = False)
                latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim = -1)[None,:]

                
                self.storage.next_state[step] = obs.clone()
                self.storage.insert(
                    state=obs.squeeze(),
                    belief=None, # could I get rid of belief?
                    task=None, # could I get rid of task?
                    actions=action.double(),
                    rewards_raw=reward.squeeze(0),
                    rewards_normalised=reward.squeeze(0),#rew_normalised, don't use
                    value_preds=value.squeeze(0),
                    masks=masks_done.squeeze(0), # do I even need these?
                    bad_masks=bad_masks.squeeze(0), 
                    done=torch.from_numpy(done)[:,None].float(),
                    hidden_states = hidden_state.squeeze(),
                    latent = latent,
                )

                step += 1

            # TODO: convert to tensorboard
            # res[eps] = (*self.agent.update(self.storage), current_task, episode_reward.cpu().detach().numpy())
            ## Log loss
            ## TODO: tidy into function
            value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch = self.agent.update(self.storage)
            self.logger.add('value_loss', value_loss_epoch, eps)
            self.logger.add('action_loss', action_loss_epoch, eps)
            self.logger.add('entropy_loss', dist_entropy_epoch, eps)
            self.logger.add('total_loss', loss_epoch, eps)
            self.logger.add('current_task', current_task, eps)

            ## Log reward quantiles
            ## TODO: tidy into function
            quantiles = [0.05, 0.1, 0.2, 0.3, 0.5]
            reward_quantiles = torch.quantile(torch.stack(episode_reward).cpu(), torch.tensor(quantiles))
            quantile_dict = dict(zip(['q_' + str(q) for q in quantiles], reward_quantiles))
            self.logger.add_multiple('reward_quantiles', quantile_dict, eps)

            ## Log success
            # self.logger.add('success', np.sum([i['success'] for i in info]), eps)

            # clears out old data
            self.storage.after_update()
            eps+=1
        end_time = time.time()
        print(f"completed in {end_time - start_time}")
        self.envs.close()

        ## TODO: replace this with tensorboard
        # return res

    def evaluate(self, current_task, test_processes = 10):

        ## TODO: 
        ## need to consider how to log these
        ## want each task to have its own line
        ## would like each task to have proportion of successes, some reward metric
        ## may not log all metrics in tensorflow
        ## log at episode level but also get tasks to share a plot

        ## TODO: create vectorised envs
        ## num runs given by test_processes
        test_envs = prepare_parallel_envs(
            self.envs, 
            self.rollout_len,
            test_processes, 
            device
        )
        start_time = time.time() # use this in logger?
        eps = 0

        # steps limit is parameter for whole continual env
        while test_envs.get_env_attr('cur_step') < test_envs.get_env_attr('steps_limit'):

            step = 0
            obs = test_envs.reset() # we reset all at once as metaworld is time limited
            episode_reward = []
            successes = []
            done = [False for _ in range(test_processes)]

            ## TODO: determine how frequently to get prior - do at start of each episode for now
            with torch.no_grad():
                _, latent_mean, latent_logvar, hidden_state = self.agent.actor_critic.encoder.prior(test_processes)
                latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim=-1)


            while not all(done):
                with torch.no_grad():
                    _, action = self.agent.act(obs, latent, None, None)
                next_obs, reward, done, info = test_envs.step(action)
                assert all(done) == any(done), "Metaworld envs should all end simultaneously"

                obs = next_obs

                ## combine all rewards
                episode_reward.append(reward)
                # if we succeed at all then the task is successful
                successes.append(torch.tensor([i['success'] for i in info]))

                # ## TODO: Don't think I need this for eval
                # # create mask for episode ends
                # masks_done = torch.FloatTensor([[0.0] if _done else [1.0] for _done in done]).to(device)
                # # reset hidden state if done
                # if all(done):
                #     with torch.no_grad():
                #         hidden_state = self.agent.actor_critic.encoder.reset_hidden(hidden_state, masks_done)

                with torch.no_grad():
                    _, latent_mean, latent_logvar, hidden_state = self.agent.actor_critic.encoder(action, obs, reward, hidden_state, return_prior = False)
                    latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim = -1)[None,:]

                step += 1

            ## log the results here
            ## current task
            self.logger.add('current_task', current_task, eps)
            ## TODO: tidy into function
            quantiles = [0.05, 0.1, 0.2, 0.3, 0.5]
            reward_quantiles = torch.quantile(torch.stack(episode_reward).cpu(), torch.tensor(quantiles))
            quantile_dict = dict(zip(['q_' + str(q) for q in quantiles], reward_quantiles))
            self.logger.add_multiple('reward_quantiles', quantile_dict, eps)

            ## Log success - if success then 1 divided by number of envs ()
            stacked_successes = torch.stack(successes).max(0)[0].sum() / test_processes
            self.logger.add('success', stacked_successes, eps)

            eps+=1
        end_time = time.time()
        print(f"eval completed in {end_time - start_time}")
        test_envs.close()



    # def log(self, run_stats, train_stats, start_time):

    #     # --- visualise behaviour of policy ---

    #     if (self.iter_idx + 1) % self.args.vis_interval == 0:
    #         ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
    #         utl_eval.visualise_behaviour(args=self.args,
    #                                      policy=self.policy,
    #                                      image_folder=self.logger.full_output_folder,
    #                                      iter_idx=self.iter_idx,
    #                                      ret_rms=ret_rms,
    #                                      encoder=self.vae.encoder,
    #                                      reward_decoder=self.vae.reward_decoder,
    #                                      state_decoder=self.vae.state_decoder,
    #                                      task_decoder=self.vae.task_decoder,
    #                                      compute_rew_reconstruction_loss=self.vae.compute_rew_reconstruction_loss,
    #                                      compute_state_reconstruction_loss=self.vae.compute_state_reconstruction_loss,
    #                                      compute_task_reconstruction_loss=self.vae.compute_task_reconstruction_loss,
    #                                      compute_kl_loss=self.vae.compute_kl_loss,
    #                                      tasks=self.train_tasks,
    #                                      )

    #     # --- evaluate policy ----

    #     if (self.iter_idx + 1) % self.args.eval_interval == 0:

    #         ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
    #         returns_per_episode = utl_eval.evaluate(args=self.args,
    #                                                 policy=self.policy,
    #                                                 ret_rms=ret_rms,
    #                                                 encoder=self.vae.encoder,
    #                                                 iter_idx=self.iter_idx,
    #                                                 tasks=self.train_tasks,
    #                                                 )

    #         # log the return avg/std across tasks (=processes)
    #         returns_avg = returns_per_episode.mean(dim=0)
    #         returns_std = returns_per_episode.std(dim=0)
    #         for k in range(len(returns_avg)):
    #             self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
    #             self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
    #             self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
    #             self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)

    #         print(f"Updates {self.iter_idx}, "
    #               f"Frames {self.frames}, "
    #               f"FPS {int(self.frames / (time.time() - start_time))}, "
    #               f"\n Mean return (train): {returns_avg[-1].item()} \n"
    #               )

    #     # --- save models ---

    #     if (self.iter_idx + 1) % self.args.save_interval == 0:
    #         save_path = os.path.join(self.logger.full_output_folder, 'models')
    #         if not os.path.exists(save_path):
    #             os.mkdir(save_path)

    #         idx_labels = ['']
    #         if self.args.save_intermediate_models:
    #             idx_labels.append(int(self.iter_idx))

    #         for idx_label in idx_labels:

    #             torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
    #             torch.save(self.vae.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
    #             if self.vae.state_decoder is not None:
    #                 torch.save(self.vae.state_decoder, os.path.join(save_path, f"state_decoder{idx_label}.pt"))
    #             if self.vae.reward_decoder is not None:
    #                 torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder{idx_label}.pt"))
    #             if self.vae.task_decoder is not None:
    #                 torch.save(self.vae.task_decoder, os.path.join(save_path, f"task_decoder{idx_label}.pt"))

    #             # save normalisation params of envs
    #             if self.args.norm_rew_for_policy:
    #                 rew_rms = self.envs.venv.ret_rms
    #                 utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
    #             # TODO: grab from policy and save?
    #             # if self.args.norm_obs_for_policy:
    #             #     obs_rms = self.envs.venv.obs_rms
    #             #     utl.save_obj(obs_rms, save_path, f"env_obs_rms{idx_label}")

    #     # --- log some other things ---

    #     if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):

    #         self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
    #         self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)

    #         self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
    #         self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)

    #         self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
    #         self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
    #         self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
    #         self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

    #         self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
    #         if hasattr(self.policy.actor_critic, 'logstd'):
    #             self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
    #         self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
    #         self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

    #         self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
    #         self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)

    #         # log the average weights and gradients of all models (where applicable)
    #         for [model, name] in [
    #             [self.policy.actor_critic, 'policy'],
    #             [self.vae.encoder, 'encoder'],
    #             [self.vae.reward_decoder, 'reward_decoder'],
    #             [self.vae.state_decoder, 'state_transition_decoder'],
    #             [self.vae.task_decoder, 'task_decoder']
    #         ]:
    #             if model is not None:
    #                 param_list = list(model.parameters())
    #                 param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
    #                 self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
    #                 if name == 'policy':
    #                     self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
    #                 if param_list[0].grad is not None:
    #                     param_grad_mean = np.mean([param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
    #                     self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)
