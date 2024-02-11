import metaworld
import argparse
import datetime
import json
import os

from environments.custom_metaworld_benchmark import CustomML10, ML3

from utils.rl2_eval_utils import *
from utils.helpers import boolean_argument

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder')
    parser.add_argument('--log_folder', default='./logs/rl2_eval/')
    parser.add_argument('--num_episodes', default = 10)
    parser.add_argument('--num_rounds', default = 1)
    parser.add_argument('--num_explore', default = None)
    parser.add_argument('--deterministic', type=boolean_argument, default=True)
    parser.add_argument('--benchmark', default='ML10')

    args, rest_args = parser.parse_known_args()

    run_folder = args.run_folder
    log_folder = args.log_folder + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S')
    num_rounds = int(args.num_rounds)
    num_episodes = int(args.num_episodes)
    num_explore = int(args.num_explore) if args.num_explore is not None else args.num_explore
    train_benchmark = args.benchmark + '-v2'
    test_benchmark = args.benchmark + '_test-v2'
    deterministic = args.deterministic

    ## create the log folder if it doesn't
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    ## log the cmd input
    config = {k: v for (k, v) in vars(args).items()}
    config['device'] = str(device)
    with open(os.path.join(log_folder + '/config.json'), 'w') as file:
        json.dump(config, file, indent=2)

    ## Create RL2 agent
    policy_net = torch.load(run_folder + '/models/policy.pt')
    encoder_net = torch.load(run_folder + '/models/encoder.pt')

    # # get benchmark and tasks
    if args.benchmark == 'CustomML10':
        benchmark = CustomML10()

    elif args.benchmark == 'ML10':
        benchmark = metaworld.ML10()

    elif args.benchmark == 'ML1':
        benchmark = metaworld.ML1('reach-v2')
    elif args.benchmark == 'ML3':
        benchmark = ML3()
    else:
        raise ValueError(f"{args.benchmark} is not available, please choose: CustomML10 or ML10 ML1")

    # get task names
    train_tasks = list(benchmark.train_classes.keys())
    test_tasks = list(benchmark.test_classes.keys())

    # run the evaluation
    train_all_results = pd.DataFrame()
    test_all_results = pd.DataFrame()
    for i in range(num_rounds):
        print(f'Evaluating on train set for run {i}')
        train_reward, train_success = evaluate_rl2(
            env_name = train_benchmark, 
            policy = policy_net,
            iter_idx = i,
            encoder = encoder_net,
            num_episodes=num_episodes,
            num_processes=len(train_tasks),
            num_explore = num_explore,
            deterministic = deterministic
        )

        train_results = combine_results(train_tasks, train_reward, train_success)
        train_all_results = pd.concat([train_results, train_all_results])

        print(f'Evaluating on test set for iter {i}')
        test_reward, test_success = evaluate_rl2(
            env_name = test_benchmark, 
            policy = policy_net,
            iter_idx = i,
            encoder = encoder_net,
            num_episodes=num_episodes,
            num_processes=len(test_tasks),
            num_explore = num_explore,
            deterministic = False
        )

        test_results = combine_results(test_tasks, test_reward, test_success)
        test_all_results = pd.concat([test_results, test_all_results])


    # save the results
    print(f'saving results to {log_folder}')
    train_all_results.to_csv(log_folder + '/train_results.csv')
    test_all_results.to_csv(log_folder + '/test_results.csv')

    ## save plots
    plot_results(train_all_results, log_folder, 'train_results')
    plot_results(test_all_results, log_folder, 'test_results')

    print(
        'train results:\n',
        'by task:\n',
        train_all_results.loc[:, ['tasks', 'mean_rewards', 'successes']].groupby('tasks').mean(),
        'by episode:\n',
        train_all_results.loc[:, ['episode','mean_rewards', 'successes']].groupby('episode').mean(),
        '\n===================',
        'test results:\n',
        'by task:\n',
        test_all_results.loc[:, ['tasks', 'mean_rewards', 'successes']].groupby('tasks').mean(),
        'by episode:\n',
        test_all_results.loc[:, ['episode','mean_rewards', 'successes']].groupby('episode').mean(),
        '\n==================='
    )


if __name__=='__main__':
    main()

