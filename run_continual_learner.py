import metaworld
import argparse
import datetime
import json
import os
import torch

from algorithms.custom_ppo import CustomPPO
from models.combined_actor_critic import ActorCritic
from continualworld_utils.constants import TASK_SEQS
from continuallearner import ContinualLearner

from utils.helpers import boolean_argument

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    ### get args
    parser = argparse.ArgumentParser()

    ## TODO: how to randomly initialise a network!?
    parser.add_argument('--run_folder', help = 'location from which to load a network')
    parser.add_argument('--log_folder', default='./logs/continual_learning/')

    ## num processes and env steps
    parser.add_argument('--steps_per_env', type=int, default=1e6, help="Number of steps each environment is trained on. CW uses 1m")
    parser.add_argument('--rollout_len', type=int, default=500, help="rollout len for each episode. MW default is 500")
    parser.add_argument('--num_processes', type=int, default = 20, help="number of parallel processes to run - effectively controls batch size as one update is run per each episode x num_processes")
    parser.add_argument('--randomization', type=str, default='deterministic', help='randomisation setting for CW must be one of: deterministic, random_init_all, random_init_fixed20, random_init_small_box')
    
    ## PPO params
    parser.add_argument('--ppo_clip_param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--ppo_epoch', type=int, default=16, help="PPO update epochs")
    parser.add_argument('--num_mini_batch', type=int, default=4, help="num minibatches per update")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help = "learning rate for network")
    parser.add_argument('--entropy_coef', type= float, default=5e-3, help="entropy coefficient for the policy")
    parser.add_argument('--gamma', type=float, default=0.99, help = "discount rate")
    parser.add_argument('--tau', type=float, default=0.97, help="discount rate for GAE")
    parser.add_argument('--normalise_rewards', type=boolean_argument, default=True, help="normalise rewards")

    parser.add_argument('--value_loss_coef', type=float, default=1, help="coefficient applied to the value loss")
    parser.add_argument('--use_huberloss', type=boolean_argument, default=False, help="use huber loss instead of MSE")
    parser.add_argument('--use_clipped_value_loss', type=boolean_argument, default=False, help="clip the value loss")

    ## optimiser args
    parser.add_argument('--optimiser', type=str, default='adam', help='just choose adam')
    parser.add_argument('--eps', type=float, default =1.0e-8, help='eps param for adam optimiser')

    # context_window=None ## make an arg??

    ## admin stuff
    parser.add_argument('--seed', type=int, default=42, help="set the seed for maximum reproducibility")
    parser.add_argument('--log_every', type=int, default=10, help="logging frequency where integer value is number of updates")


    # TASK_SEQS['CW10'][:1], 


    args, rest_args = parser.parse_known_args()

    ## do a check for s/p
    assert args.steps_per_env % args.num_processes == 0, "steps_per_env must be divisible by num processes"
    print(
        f"Running with {args.num_processes} processes for {args.steps_per_env / args.num_processes} each for a total of {args.steps_per_env} steps per env."
    )

    ## set arg variables 
    policy_loc = args.run_folder + '/models/policy.pt'
    encoder_loc = args.run_folder + '/models/encoder.pt'

    ## create networks / agents
    # get RL2 trained policy for example
    # RUN_FOLDER = './logs/logs_CustomML10-v2/rl2_73__13:01_12:00:05' ## make an arg
    policy_net = torch.load(policy_loc)
    encoder_net = torch.load(encoder_loc)
    ac = ActorCritic(policy_net, encoder_net)
    agent = CustomPPO(
        actor_critic=ac,
        value_loss_coef = args.value_loss_coef,
        entropy_coef = args.entropy_coef,
        policy_optimiser=args.optimiser,
        policy_anneal_lr=False, # remove
        train_steps = 3, # remove
        lr = args.learning_rate,
        eps= args.eps, # for adam optimiser
        clip_param = args.ppo_clip_param, 
        ppo_epoch = args.ppo_epoch, 
        num_mini_batch=args.num_mini_batch,
        use_huber_loss = args.use_huberloss,
        use_clipped_value_loss=args.use_clipped_value_loss,
        context_window=None ## make an arg??
    )

    ## effective steps per env is the required amount of steps that each paralell env is run for
    effective_steps_per_env = args.steps_per_env / args.num_processes

    ## run continual learner
    continual_learner = ContinualLearner(
        args.seed, 
        TASK_SEQS['CW10'][:1], 
        agent, 
        args.num_processes,
        args.rollout_len,
        effective_steps_per_env,
        args.normalise_rewards,
        args.log_folder, # make location an arg, but perhaps reconsider how this is created, put under logs foler
        'dummy',
        gamma = args.gamma, 
        tau = args.tau, 
        log_every = args.log_every, 
        randomization=args.randomization 
    )

    ## run it
    continual_learner.train()

if __name__ == '__main__':
    main()