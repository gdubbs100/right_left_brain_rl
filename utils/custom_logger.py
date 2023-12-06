import datetime
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomLogger:
    def __init__(self, log_dir, scenario_name):

        self.log_dir = os.path.join(
            log_dir,
            scenario_name + '_' + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S')
        )

        self.writer = SummaryWriter(log_dir=self.log_dir)

        print('logging under', self.log_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        ### TODO: when you create a runner script, make sure you can save out the args in a json
        # with open(os.path.join(self.full_output_folder, 'config.json'), 'w') as f:
        #     try:
        #         config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        #     except:
        #         config = args
        #     config.update(device=device.type)
        #     json.dump(config, f, indent=2)

    def add(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)
