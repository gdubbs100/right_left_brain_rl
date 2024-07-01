import datetime
import json
import os
import csv

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomLogger:
    def __init__(self, log_dir,  logging_quantiles, args = None, left_args = None, right_args = None):

        self.args = args
        self.left_args = left_args
        self.right_args = right_args

        self.log_dir = os.path.join(
            log_dir,
            self.args.run_name + '_' + str(self.args.seed) + '_' + self.args.algorithm + datetime.datetime.now().strftime('_%d%m_%H%M%S')
        )

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.network_dir = self.log_dir + '/actor_critic.pt'

        print('logging under', self.log_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logging_quantiles = logging_quantiles
        self.result_log_headers = [
            'training_task', 
            'evaluation_task', 
            'num_successes', 
            'num_episodes', 
            'reward_mean', 
            *['rq_' + str(q) for q in self.logging_quantiles], # for rewards quantiles
            *['gq_' + str(q) for q in self.logging_quantiles], # for gating quantiles
            'frame'
        ]

        ## create csvs with headers
        self.train_csv_dir = self.log_dir + '/train_results.csv'
        self.init_result_csv(self.train_csv_dir)
        self.test_csv_dir = self.log_dir + '/test_results.csv'
        self.init_result_csv(self.test_csv_dir)

        # if logging bicameral agent, 
        if (self.args is not None) and (self.args.algorithm=='bicameral'):
            self.left_csv_dir = self.log_dir + '/left_eval_results.csv'
            self.init_result_csv(self.left_csv_dir)


        ### save out args if supplied to continual learner - otherwise ignore
        if self.args is not None:

            self.args.log_dir = self.log_dir
            self.log_args(self.args, os.path.join(self.log_dir, 'config.json'))
        else:
            print('no args supplied...')

        ### save out network args if supplied to continual learner - otherwise ignore
        if self.left_args is not None:

            self.log_args(self.left_args, os.path.join(self.log_dir,'left_config.json'))
        elif self.right_args is not None:
            self.log_args(self.right_args, os.path.join(self.log_dir,'right_config.json'))
        else:
            print('no base network args supplied...')

    def add_tensorboard(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)

    def add_multiple_tensorboard(self, name, value_dict, x_pos):
        self.writer.add_scalars(name, value_dict, x_pos)

    def add_csv(self, row, csv_to_do):
        if csv_to_do == 'train':
            csv_dir = self.train_csv_dir
        elif csv_to_do == 'left':
            csv_dir = self.left_csv_dir
        elif csv_to_do == 'test':
            csv_dir = self.test_csv_dir
        else:
            raise ValueError(f"No csv for {csv_to_do}")

        assert len(row) == len(self.result_log_headers), 'You have not passed sufficient values to logger'
        with open(csv_dir, 'a') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                row
            )

    def init_result_csv(self, csv_dir):
        with open(csv_dir, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                self.result_log_headers
            )
        
    def save_network(self, network):
        torch.save(network, self.network_dir)

    def log_args(self, args, path):
        
        with open(os.path.join(path), 'w') as f:
            try:
                config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            except:
                config = args
            config.update(device=device.type)
            json.dump(config, f, indent=2)

