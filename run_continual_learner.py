import metaworld
import argparse
import datetime
import json
import os
import torch

from continualworld_utils.constants import TASK_SEQS
from continuallearner import ContinualLearner

from utils.helpers import boolean_argument

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    ### get args
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type = str, default = '', help = 'Give a name to your run for ease of logging')

    parser.add_argument('--algorithm', type = str, default='left_only', help='type of algorithm to run, choose: left_only, right_only, bicameral')
    parser.add_argument('--run_folder', help = 'folder from which to load a network - requires config.json if left_only or bicameral, requires policy and encoder networks if right_only')
    parser.add_argument('--log_folder', default='./logs/continual_learning/')
    parser.add_argument('--env_name', type = str, default = 'push-v2',
                        help='tasks to train on - choose one of: ["push-v2", "reach-v2", "pick-place-v2", "door-open-v2", "button-press-v2", "faucet-open-v2", "reach-wall-v2", "push-wall-v2", "pick-place-wall-v2"]')

    ## num processes and env steps
    parser.add_argument('--steps_per_env', type=int, default=1e6, help="Number of steps each environment is trained on. CW uses 1m")
    parser.add_argument('--rollout_len', type=int, default=500, help="rollout len for each episode. MW default is 500")
    parser.add_argument('--num_processes', type=int, default=20, help="number of parallel processes to run - effectively controls batch size as one update is run per each episode x num_processes")
    parser.add_argument('--randomization', type=str, default='deterministic', help='randomisation setting for CW must be one of: deterministic, random_init_all, random_init_fixed20, random_init_small_box')

    ## Arguments for bicameral algorithm
    ##NOTE: There are some parameter settings here that conflict - add checks?
    parser.add_argument('--init_std', type = float, default=0.5, help = 'standard deviation for bicameral action dist')
    parser.add_argument('--use_gating_penalty', type = boolean_argument, default = False, help="apply a gating penalty to the loss function")
    parser.add_argument('--gating_alpha', type=float, default=0.75, help="exponent of gating penalty")
    parser.add_argument('--gating_beta', type = float, default=0.1, help="multiplicative parameter for gating penalty")
    parser.add_argument('--use_action_in_gate', type = boolean_argument, default = False, help='pass action to gating encoder')
    parser.add_argument('--use_state_in_gate', type = boolean_argument, default=False, help='pass state to gating encoder')
    parser.add_argument('--use_gating_schedule', type = boolean_argument, default = False, help="use an update schedule for gating values")
    parser.add_argument('--gating_schedule_type', type = str, default='addative', help='set to addative or multiplicative. Addative subtracts a constant, multiplcative reduces by a factor.')
    parser.add_argument('--gating_schedule_update', type=float, default=0.05, help="Constant or factor to reduce right gating value by when using schedule")
    parser.add_argument('--min_right_value', type=float, default = 0.05, help = 'minimum gating value for the right network when stepping')
    parser.add_argument('--init_right_value', type=float, default=0.95, help="Initialisation value for right gate when using scheduler")
    parser.add_argument('--step_gate_every', type=int, default=10, help = 'frequency of gating scheduler updates - expressed in network updates')
        
    ## PPO params
    parser.add_argument('--ppo_clip_param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--ppo_epoch', type=int, default=16, help="PPO update epochs")
    parser.add_argument('--num_mini_batch', type=int, default=4, help="num minibatches per update")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help = "learning rate for network")
    parser.add_argument('--entropy_coef', type= float, default=5e-3, help="entropy coefficient for the policy")
    parser.add_argument('--gamma', type=float, default=0.99, help = "discount rate")
    parser.add_argument('--tau', type=float, default=0.97, help="discount rate for GAE")
    parser.add_argument('--normalise_rewards', type=boolean_argument, default=True, help="normalise rewards")
    parser.add_argument('--context_window', type = int, default=None, help="Determines window size for bptt. None means use max")

    parser.add_argument('--value_loss_coef', type=float, default=1, help="coefficient applied to the value loss")
    parser.add_argument('--use_huberloss', type=boolean_argument, default=False, help="use huber loss instead of MSE")
    parser.add_argument('--use_clipped_value_loss', type=boolean_argument, default=False, help="clip the value loss")

    ## optimiser args
    parser.add_argument('--optimiser', type=str, default='adam', help='just choose adam')
    parser.add_argument('--eps', type=float, default =1.0e-8, help='eps param for adam optimiser')

    ## admin stuff
    parser.add_argument('--seed', type=int, default=73, help="set the seed for maximum reproducibility")
    parser.add_argument('--eval_every', type=int, default=10, help="logging frequency where integer value is number of updates")

    args, rest_args = parser.parse_known_args()

    ## do a check for s/p
    assert args.steps_per_env % args.num_processes == 0, "steps_per_env must be divisible by num processes"
    print(
        f"Running with {args.num_processes} processes for {args.steps_per_env / args.num_processes} each for a total of {args.steps_per_env} steps per env."
    )
    ## check environment specs
    # assert args.envs in ['CW20', 'CW10'], f"env is not one of CW10, CW20. env is: {args.envs}"

    ## effective steps per env is the required amount of steps that each paralell env is run for
    effective_steps_per_env = args.steps_per_env / args.num_processes
    tasks = [args.env_name]#TASK_SEQS[args.envs]

    ## run continual learner
    continual_learner = ContinualLearner(
        args.seed, 
        tasks,  
        args.num_processes,
        args.rollout_len,
        effective_steps_per_env,
        args.normalise_rewards,
        args.log_folder, 
        gamma = args.gamma, 
        tau = args.tau, 
        eval_every = args.eval_every, 
        randomization=args.randomization,
        args = args # pass args through to logger if you want
    )

    ## run it
    continual_learner.train()

if __name__ == '__main__':
    main()