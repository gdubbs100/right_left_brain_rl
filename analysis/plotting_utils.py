import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

## constants to select all reward / gating quantile columns
REWARD_QUANTILES = [f"rq_{i/10}" for i in range(1, 10)]
GATING_QUANTILES = [f"gq_{i/10}" for i in range(1, 10)]

## color palette
algs =[
    'bicameral_w_gating_encoder',
    'bicameral_w_gating_encoder+penalty',
    'bicameral_w_gating_schedule',
    'left_only_double_params',
    'random',
    'right_only_double_params',
    'right_only',

]

PALETTE = dict()
for col, alg in zip(sns.color_palette('tab10'), algs):
    PALETTE[alg] = col

### classes to handle reading of data
class resultsManager:

    def __init__(self, experiment_log_loc, environment, setting='random'):
        self.experiments = (
            pd.read_csv(experiment_log_loc)
            .query(f'(environment=="{environment}") & (latest=="Y") & (setting=="{setting}") & (~file_location.isna())')
            .loc[:,['file_location', 'algorithm']]
            .set_index('algorithm')
            .to_dict()['file_location']
        )
        self.environment = environment
        self.setting = setting
        self.data = {name: resultSet(root = os.path.join('../logs', file), name = name) \
            for name, file in self.experiments.items()}

        self.random_baseline = self.calculate_baseline('random')
        self.right_only_single_baseline = self.calculate_baseline('right_only')
        # self.right_only_double_baseline = None ## need to create this still

    def calculate_baseline(self, name):
        # assumes right_only / random have only train data
        reward_mean = (
            self.data[name]
            .data
            .query('result_group=="train_results"')
            .loc[:, 'reward_mean']
            .mean()
        )
        success_rate = (
            self.data[name]
            .data
            .query('result_group=="train_results"')
            .loc[:, 'num_successes']
            .mean()
        )
        return (reward_mean, success_rate)

    def get_result_group_data(self, result_group):
        return (
            pd.concat(
                [self.data[k].data.query(f'result_group=="{result_group}"') for k in self.data.keys()]
                )
            .drop(['num_episodes', 'evaluation_task'], axis =1) 
        )
            
class resultSet:

    def __init__(self, root, name):
        self.root = root
        self.name = name
        self.contents = os.listdir(self.root)
        self.data = self.collect_results()

    def collect_results(self):
        train_results = self.read_if_exists('train_results.csv')
        test_results = self.read_if_exists('test_results.csv')
        left_eval_results = self.read_if_exists('left_eval_results.csv')

        combined_results = (
            pd.concat([train_results, test_results, left_eval_results])
            .assign(run_name = self.name)
        )

        return combined_results

    def read_if_exists(self, path):
        
        if path in self.contents:
            result = (
                pd.read_csv(os.path.join(self.root, path))
                .assign(result_group = path.replace('.csv', ''))
            )
        else:
            result = None
        
        return result


## Plotting functions
def create_lineplot(data, title, ax):

    sns.lineplot(
        data = data,
        x = 'frame',
        y = 'value',
        hue = 'run_name',
        ax = ax,
        palette=PALETTE
    )
    
    ax.set_xlabel('Environment Steps')
    ax.set_title(title)

def plot_rew_and_succ(data, alpha = 1, figsize=(20,7)):

    plot_df = (
        data
        .melt(id_vars=['training_task', 'frame', 'result_group', 'run_name'])
        .query('variable.isin(["reward_mean", "num_successes"])')
        .set_index('frame')
        .groupby(['training_task', 'result_group', 'run_name', 'variable'])
        .apply(lambda x: x[['value']].ewm(alpha=alpha).mean())
        .reset_index()
    )

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    create_lineplot(
        data = plot_df.query('variable=="reward_mean"'),
        title = "Mean Rewards",
        ax=ax[0]
    )

    create_lineplot(
        data = plot_df.query('variable=="num_successes"'),
        title = 'Success Rate',
        ax = ax[1]
    )

    plt.tight_layout()
    plt.show();

def plot_rew_quantiles(data, alpha=1, figsize = (20, 7)):

    plot_df = (
        data
        .melt(id_vars=['training_task', 'frame', 'result_group', 'run_name'])
        .query('variable.isin(["rq_0.1", "rq_0.5", "rq_0.9"])')
        .set_index('frame')
        .groupby(['training_task', 'result_group', 'run_name', 'variable'])
        .apply(lambda x: x[['value']].ewm(alpha=alpha).mean())
        .reset_index()
    )

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    create_lineplot(
        plot_df.query('variable=="rq_0.1"'),
        title="0.1 Quantile Rewards",
        ax = ax[0]
    )
    create_lineplot(
        plot_df.query('variable=="rq_0.5"'),
        title="0.5 Quantile Rewards",
        ax = ax[1]
    )
    create_lineplot(
        plot_df.query('variable=="rq_0.9"'),
        title="0.9 Quantile Rewards",
        ax = ax[2]
    )

    plt.tight_layout()
    plt.show();

def plot_gating_values(data, title, alpha = 1, figsize = (20, 7)):

    plot_df = (
        data
        .melt(id_vars=['training_task', 'frame', 'result_group', 'run_name'])
        .query(f'variable.isin({GATING_QUANTILES})')
        .set_index('frame')
        .groupby(['training_task', 'result_group', 'run_name', 'variable'])
        .apply(lambda x: x[['value']].ewm(alpha=alpha).mean())
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)
    create_lineplot(
        data = plot_df,
        title=title,
        ax = ax
    )

    plt.show();

    