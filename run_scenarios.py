import subprocess
import argparse


SCRIPT = './run_continual_learner.py'
parser = argparse.ArgumentParser()

parser.add_argument('--algorithm', type = str, default = 'random', help = 'Give a name to your run for ease of logging')
parser.add_argument('--steps_per_env', type=str, default='100', help = 'number of iters')
parser.add_argument('--log_folder', type=str, default='logs/baselines/random_agent', help = 'where to save results')
parser.add_argument('--randomization', type=str, default='random_init_all', help = 'whether env randomises on reset')
parser.add_argument('--run_folder', type=str, default='rl2_baseline/bicameral_rl2')
parser.add_argument('--num_processes', type=str, default='8')

args, _ = parser.parse_known_args()
envs = [
    'pick-place-v2',
    'reach-v2',
    'push-v2',
    # 'reach-wall-v2',
    'push-wall-v2',
    'pick-place-wall-v2',
    # 'door-open-v2', 
    'button-press-v2', 
    'faucet-open-v2'
]

base_args = [
    '--run_folder', args.run_folder,
    '--randomization', args.randomization,
    '--algorithm', args.algorithm, 
    '--steps_per_env', args.steps_per_env,
    '--log_folder', args.log_folder,
    '--num_processes', args.num_processes
]

for env in envs:
    print(f"RUNNING: {env}")
    additional_args = [
        '--env_name', env, 
        '--run_name', f'baseline_{args.algorithm}_agent_{args.randomization}_env_{env}',
        ]
    command = ["python", SCRIPT] + additional_args + base_args
    subprocess.run(command)
    