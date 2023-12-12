import datetime
import json
import os
import csv

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomLogger:
    def __init__(self, log_dir, scenario_name, logging_quantiles):

        self.log_dir = os.path.join(
            log_dir,
            scenario_name + '_' + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S')
        )

        self.csv_dir = self.log_dir + '/results.csv'
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.network_dir = self.log_dir + '/actor_critic.pt'

        print('logging under', self.log_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logging_quantiles = logging_quantiles
        self.result_log_headers = ['training_task', 'evaluation_task', 'successes', 'result_mean', *['q_' + str(q) for q in self.logging_quantiles], 'episode']
        with open(self.csv_dir, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                self.result_log_headers
            )

        ### TODO: when you create a runner script, make sure you can save out the args in a json
        # with open(os.path.join(self.full_output_folder, 'config.json'), 'w') as f:
        #     try:
        #         config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        #     except:
        #         config = args
        #     config.update(device=device.type)
        #     json.dump(config, f, indent=2)

    def add_tensorboard(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)

    def add_multiple_tensorboard(self, name, value_dict, x_pos):
        self.writer.add_scalars(name, value_dict, x_pos)

    def add_csv(self, row):
        assert len(row) == len(self.result_log_headers), 'You have not passed sufficient values to logger'
        with open(self.csv_dir, 'a') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(
                row
            )
    def save_network(self, network):
        torch.save(network, self.network_dir)

