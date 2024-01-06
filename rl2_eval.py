import metaworld
import argparse
import datetime
import json
import os

from environments.custom_metaworld_benchmark import CustomML10

from utils.rl2_eval_utils import *
from models.combined_actor_critic import ActorCritic

# from models import encoder, policy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder')
    parser.add_argument('--log_folder',default='./logs/rl2_eval')
    parser.add_argument('--num_eval_rounds', default=16)
    parser.add_argument('--num_sequential_rounds', default=1)
    parser.add_argument('--benchmark', default='custom')

    args, rest_args = parser.parse_known_args()

    run_folder = args.run_folder
    log_folder = args.log_folder + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S')
    num_eval_rounds = int(args.num_eval_rounds)
    num_sequential_rounds = int(args.num_sequential_rounds)
    eval_benchmark = args.benchmark

    ## create the log folder if it doesn't
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    ## log the cmd input
    config = {k: v for (k, v) in vars(args).items()}
    config['device'] = str(device)
    with open(os.path.join(log_folder + '/config.json'), 'w') as file:
        json.dump(config, file, indent=2)

    ## Create RL2 agent
    policy_net = torch.load(run_folder + '/models/policy.pt')
    encoder_net = torch.load(run_folder + '/models/encoder.pt')
    ac = ActorCritic(policy_net, encoder_net)
    agent = rl2_agent(ac)

    # get benchmark and tasks
    if eval_benchmark == 'custom':
        benchmark = CustomML10()
    elif eval_benchmark == 'ml10':
        benchmark = metaworld.ML10()
    else:
        raise ValueError(f"{benchmark} is not available, please choose: custom or ml10")

    # get task names
    train_tasks = list(benchmark.train_classes.keys())
    test_tasks = list(benchmark.test_classes.keys())

    # run the evaluation
    train_results = eval_rl2(agent, train_tasks, benchmark, 'train', num_eval_rounds, num_sequential_rounds)
    test_results = eval_rl2(agent, test_tasks, benchmark, 'test', num_eval_rounds, num_sequential_rounds)

    # save the results
    train_results.to_csv(log_folder + '/train_results.csv')
    test_results.to_csv(log_folder + '/test_results.csv')


if __name__=='__main__':
    main()

