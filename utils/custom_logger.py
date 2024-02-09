import datetime
import json
import os
import csv

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomLogger:
    def __init__(self, log_dir,  logging_quantiles, args = None, base_network_args = None):

        self.args = args
        self.base_network_args = base_network_args

        self.log_dir = os.path.join(
            log_dir,
            'CL' + '_' + self.args.algorithm + datetime.datetime.now().strftime('_%d%m_%H%M%S')
        )

        self.train_csv_dir = self.log_dir + '/train_results.csv'
        self.test_csv_dir = self.log_dir + '/test_results.csv'
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

        ## create train / test csvs
        with open(self.train_csv_dir, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                self.result_log_headers
            )
        with open(self.test_csv_dir, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                self.result_log_headers
            )

        ### save out args if supplied to continual learner - otherwise ignore
        if self.args is not None:

            self.args.log_dir = self.log_dir
            self.log_args(self.args, os.path.join(self.log_dir, 'config.json'))
        else:
            print('no args supplied...')

        ### save out network args if supplied to continual learner - otherwise ignore
        if self.base_network_args is not None:

            self.log_args(self.base_network_args, os.path.join(self.log_dir,'base_network_config.json'))
        else:
            print('no base network args supplied...')

    def add_tensorboard(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)

    def add_multiple_tensorboard(self, name, value_dict, x_pos):
        self.writer.add_scalars(name, value_dict, x_pos)

    def add_csv(self, row, train = True):
        if train:
            csv_dir = self.train_csv_dir
        else:
            csv_dir = self.test_csv_dir
        assert len(row) == len(self.result_log_headers), 'You have not passed sufficient values to logger'
        with open(csv_dir, 'a') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                row
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

