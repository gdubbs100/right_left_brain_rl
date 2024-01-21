"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import json

import numpy as np
import torch

# get configs
from config.metaworld_config import \
    args_ML10_rl2, args_CustomML10_rl2, args_ML1_rl2
from config.gridworld import \
    args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from config.pointrobot import \
    args_pointrobot_multitask, args_pointrobot_varibad, args_pointrobot_rl2, args_pointrobot_humplik
from config.mujoco import \
    args_cheetah_dir_multitask, args_cheetah_dir_expert, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_multitask, args_cheetah_vel_expert, args_cheetah_vel_rl2, args_cheetah_vel_varibad, \
    args_cheetah_vel_avg, \
    args_ant_dir_multitask, args_ant_dir_expert, args_ant_dir_rl2, args_ant_dir_varibad, \
    args_ant_goal_multitask, args_ant_goal_expert, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_ant_goal_humplik, \
    args_walker_multitask, args_walker_expert, args_walker_avg, args_walker_rl2, args_walker_varibad, \
    args_humanoid_dir_varibad, args_humanoid_dir_rl2, args_humanoid_dir_multitask, args_humanoid_dir_expert
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_from_checkpoint', default = None)
    parser.add_argument('--env-type', default='gridworld_varibad')
    args, rest_args = parser.parse_known_args()
    # save for later
    load_model_from_checkpoint = args.load_model_from_checkpoint
    # print(args, rest_args)

    if args.load_model_from_checkpoint is None:
        env = args.env_type

        # --- MetaWorld ---
        if env == 'ML10_rl2':
            args = args_ML10_rl2.get_args(rest_args)
            assert args.num_processes % 10 == 0, "num_processes should be a multiple of 10 for ML10 envs"
        elif env == 'CustomML10_rl2':
            args = args_CustomML10_rl2.get_args(rest_args)
            assert args.num_processes % 10 == 0, "num_processes should be a multiple of 10 for ML10 envs"
        elif env == 'ML1_rl2':
            args = args_ML1_rl2.get_args(rest_args)
        else:
            raise Exception("Invalid Environment")
        args.load_model_from_checkpoint = load_model_from_checkpoint
    else:
        
        print(f"loading data from {args.load_model_from_checkpoint}")
        with open(args.load_model_from_checkpoint + '/config.json', 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()


if __name__ == '__main__':
    main()
