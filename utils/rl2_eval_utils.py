import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

# from environments.custom_env_utils import prepare_base_envs, prepare_parallel_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_rl2(
        env_name,
        policy,
        iter_idx,
        encoder,
        num_episodes,
        num_processes,
        num_explore = None,
        deterministic = False
        ):
    
    if num_explore is None:
        num_explore = num_episodes + 1

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    success_for_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name,
                         seed=73 * 42 + iter_idx,
                         num_processes=num_processes,
                         gamma=0.99,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=False,
                         ret_rms=None,
                         tasks=None,
                         add_done_info=True,
                         )

    num_steps = envs._max_episode_steps
    # print(f"Max episode steps = {num_steps}")

    # reset environments
    state = envs.reset().float().to(device)#utl.reset_env(envs, args)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    # reset latent state to prior
    _, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)

    for episode_idx in range(num_episodes):

        ## reset hidden state after exploration episodes
        if episode_idx > num_explore:
            print(episode_idx, "resetting hidden to adapted hidden state")
            fake_dones = torch.zeros(num_processes).to(device)
            encoder.reset_hidden(adapted_hidden, fake_dones)

        for step_idx in range(num_steps):

            with torch.no_grad():
                latent = torch.cat((latent_mean, latent_logvar), dim=-1).squeeze()
                _, action = policy.act(None, latent, None, None, deterministic=deterministic)

            # observe reward and next obs
            state, reward, done, infos = envs.step(action.detach())
            
            # handles case where we normalise rewards
            if isinstance(reward, list):
                reward = reward[0].to(device)
            else:
                reward = reward.to(device)

            # save stuff from info
            done_mdp = [info['done_mdp'] for info in infos]
            successes = torch.tensor([info['success'] for info in infos]).to(device)


            # update the hidden state
            _, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                              next_obs=state,
                                                                              action=action,
                                                                              reward=reward,
                                                                              done=None,
                                                                              hidden_state=hidden_state)
            
            # saves hidden state after initial exploration (-1 because zero index)
            if episode_idx == (num_explore - 1):
                adapted_hidden = hidden_state.clone()

            # get successes
            success_for_episode[range(num_processes), task_count] += successes.view(-1)
            # add rewards
            returns_per_episode[range(num_processes), task_count] += reward.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                state = envs.reset().float().to(device)

    envs.close()

    # normalise to 1/0
    success_for_episode /= (success_for_episode + 1.0e-10)

    return returns_per_episode[:, :num_episodes], success_for_episode[:, :num_episodes]


def combine_results(task_list, rewards, successes):

    reward_df = pd.DataFrame(
        rewards.detach().cpu().numpy().tolist(),
        index = task_list
        )\
        .melt(
            var_name='episode',
            value_name = 'mean_rewards',
            ignore_index=False
        ).reset_index().rename(columns = {'index':'tasks'})
    
    success_df = pd.DataFrame(
        successes.detach().cpu().numpy().tolist(),
        index = task_list
        )\
        .melt(
            var_name='episode',
            value_name = 'successes',
            ignore_index=False
        ).reset_index().rename(columns = {'index':'tasks'})
    
    out = reward_df.merge(success_df, on=['tasks','episode'])
    return out

def plot_results(combined_results, log_dir, file_name):
    fig, _ = plt.subplots(figsize = (15, 7))

    # show the range of rewards
    reward_plot = sns.lineplot(
        data = combined_results,
        x = 'episode',
        y = 'mean_rewards',
        hue = 'tasks',
    )

    # add red dots to successful runs
    successes = sns.scatterplot(
        data = combined_results.query('successes==1'),
        x = 'episode',
        y = 'mean_rewards',
        legend = False,
        c = 'red',
        s = 50,
        alpha = 0.75
    )
    fig.savefig(f'{log_dir}/{file_name}.png')


# class rl2_agent:
#     """
#     Simple agent wrapper to run RL2 eval
#     """
#     def __init__(self, network):
#         self.actor_critic = network

#     def act(self, obs, latent, task = None, belief = None):
#         return self.actor_critic.act(obs, latent, task, belief)

# def eval_rl2(agent, tasks, benchmark, task_set, num_eval, num_sequential_rounds):
#     raw_envs = prepare_base_envs(tasks, benchmark, task_set)
#     envs = prepare_parallel_envs(
#         raw_envs,
#         500*num_sequential_rounds, 
#         num_eval,
#         device
#     )
#     results = {
#         task:{
#             'reward_mean': 0,
#             'reward_std': 0,
#             'successes' :0
#         } for task in tasks
#     }
#     with torch.no_grad():
#         _, latent_mean, latent_logvar, hidden_state = agent.actor_critic.encoder.prior(num_eval)
#         latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim=-1)
        
#     while envs.get_env_attr('cur_step') < envs.get_env_attr('steps_limit'):

#         obs = envs.reset() # we reset all at once as metaworld is time limited
#         current_task = tasks[envs.get_env_attr('cur_seq_idx')]
#         episode_reward = []
#         successes = []
#         done = [False for _ in range(num_eval)]

#         ## TODO: determine how frequently to get prior - do at start of each episode for now
#         # with torch.no_grad():
#         #     _, latent_mean, latent_logvar, hidden_state = agent.actor_critic.encoder.prior(num_eval)
#         #     latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim=-1)


#         while not all(done):
#             with torch.no_grad():
#                 _, action = agent.act(obs, latent, None, None)
#             next_obs, reward, done, info = envs.step(action)
#             assert all(done) == any(done), "Metaworld envs should all end simultaneously"

#             obs = next_obs

#             ## combine all rewards
#             episode_reward.append(reward)
#             # if we succeed at all then the task is successful
#             successes.append(torch.tensor([i['success'] for i in info]))

#             with torch.no_grad():
#                 _, latent_mean, latent_logvar, hidden_state = agent.actor_critic.encoder(action, obs, reward, hidden_state, return_prior = False)
#                 latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim = -1)[None,:]

#         ## log the results here
#         rewards_to_log = torch.stack(episode_reward).squeeze().cpu()
#         results[current_task]['reward_mean'] = rewards_to_log.mean().numpy()
#         results[current_task]['reward_std'] = rewards_to_log.std().numpy()
#         results[current_task]['successes'] = (torch.stack(successes).max(0)[0].sum() / num_eval).numpy()

#     cleaned_results = clean_results(results)
#     return cleaned_results

# def clean_results(results_dict):
#     df = pd.DataFrame(results_dict).T.reset_index().rename(columns={'index':'task'})
#     df.loc[:,'task'] = df.loc[:,'task'].astype('category')
#     df.loc[:,'reward_mean'] = df.loc[:,'reward_mean'].astype('float')
#     df.loc[:,'reward_std'] = df.loc[:,'reward_std'].astype('float')
#     df.loc[:,'successes'] = df.loc[:,'successes'].astype('float')
#     return df
    