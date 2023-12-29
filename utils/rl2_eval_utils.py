import pandas as pd
import torch

from environments.custom_env_utils import prepare_base_envs, prepare_parallel_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class rl2_agent:
    """
    Simple agent wrapper to run RL2 eval
    """
    def __init__(self, network):
        self.actor_critic = network

    def act(self, obs, latent, task = None, belief = None):
        return self.actor_critic.act(obs, latent, task, belief)
    

def eval_rl2(agent, tasks, benchmark, task_set, num_eval):
    raw_envs = prepare_base_envs(tasks, benchmark, task_set)
    envs = prepare_parallel_envs(
        raw_envs,
        500, 
        num_eval,
        device
    )
    results = {
        task:{
            'reward_mean': 0,
            'reward_std': 0,
            'successes' :0
        } for task in tasks
    }
    while envs.get_env_attr('cur_step') < envs.get_env_attr('steps_limit'):

        obs = envs.reset() # we reset all at once as metaworld is time limited
        current_task = tasks[envs.get_env_attr('cur_seq_idx')]
        episode_reward = []
        successes = []
        done = [False for _ in range(num_eval)]

        ## TODO: determine how frequently to get prior - do at start of each episode for now
        with torch.no_grad():
            _, latent_mean, latent_logvar, hidden_state = agent.actor_critic.encoder.prior(num_eval)
            latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim=-1)


        while not all(done):
            with torch.no_grad():
                _, action = agent.act(obs, latent, None, None)
            next_obs, reward, done, info = envs.step(action)
            assert all(done) == any(done), "Metaworld envs should all end simultaneously"

            obs = next_obs

            ## combine all rewards
            episode_reward.append(reward)
            # if we succeed at all then the task is successful
            successes.append(torch.tensor([i['success'] for i in info]))

            with torch.no_grad():
                _, latent_mean, latent_logvar, hidden_state = agent.actor_critic.encoder(action, obs, reward, hidden_state, return_prior = False)
                latent = torch.cat((latent_mean.clone(), latent_logvar.clone()), dim = -1)[None,:]

        ## log the results here
        rewards_to_log = torch.stack(episode_reward).squeeze().cpu()
        results[current_task]['reward_mean'] = rewards_to_log.mean().numpy()
        results[current_task]['reward_std'] = rewards_to_log.std().numpy()
        results[current_task]['successes'] = (torch.stack(successes).max(0)[0].sum() / num_eval).numpy()

    cleaned_results = clean_results(results)
    return cleaned_results

def clean_results(results_dict):
    df = pd.DataFrame(results_dict).T.reset_index().rename(columns={'index':'task'})
    df.loc[:,'task'] = df.loc[:,'task'].astype('category')
    df.loc[:,'reward_mean'] = df.loc[:,'reward_mean'].astype('float')
    df.loc[:,'reward_std'] = df.loc[:,'reward_std'].astype('float')
    df.loc[:,'successes'] = df.loc[:,'successes'].astype('float')
    return df
    