import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

## constants to select all reward / gating quantile columns
REWARD_QUANTILES = [f"rq_{i/10}" for i in range(1, 10)]
GATING_QUANTILES = [f"gq_{i/10}" for i in range(1, 10)]

## color palette

# for algorithms
algs =[
    'bicameral_w_gating_encoder',
    'bicameral_w_gating_encoder+penalty',
    'bicameral_w_gating_schedule',
    'left_only_double_params',
    'random',
    'right_only_double_params',
    'right_only'
]

PALETTE = dict()
for col, alg in zip(sns.color_palette('tab10'), algs):
    PALETTE[alg] = col

## for tasks
TASKS = [
    'reach-v2', 'push-v2', 'pick-place-v2',
    'reach-wall-v2', 'push-wall-v2', 'bin-picking-v2',
    'faucet-open-v2', 'door-open-v2', 'button-press-v2'
]

TASK_PALETTE=dict()
for col, task in zip(sns.color_palette('tab10'), TASKS):
    TASK_PALETTE[task] = col

## mapping of tasks to tiers
TIER_DICT = {
    'push-v2': 'Tier1',
    'reach-v2':'Tier1',
    'pick-place-v2': 'Tier1',
    'reach-wall-v2':'Tier2',
    'push-wall-v2':'Tier2',
    'bin-picking-v2':'Tier2',
    'faucet-open-v2':'Tier3',
    'door-open-v2':'Tier3',
    'button-press-v2':'Tier3'
}

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

### MAIN RESULTS
## DATA PROCESSING FUNCTIONS
def smooth_results_with_median(df, reward_col, window):
    return (
        df
        .loc[:,['frame','training_task', 'run_name', reward_col]]
        .set_index('frame')
        .groupby(['training_task', 'run_name'])
        .rolling(window=window)
        .median()
        .reset_index()
    )

def get_left_median(df, reward_col, min_over=1e6):
    return (
        df
        .loc[:, ['frame', 'training_task', 'run_name', reward_col]]
        .query('run_name=="left_only_double_params"')
        .query(f'frame<={min_over}')
        .groupby(['training_task', 'run_name'])
        .median()
        .reset_index()
        .drop(['run_name', 'frame'], axis=1)
        .rename(columns={reward_col:f"left_{reward_col}"})
    )

def get_left_min(df, reward_col):
    return (
        df
        .loc[:, ['frame', 'training_task', 'run_name', reward_col]]
        .query('run_name=="left_only_double_params"')
        .groupby(['training_task', 'run_name'])
        .min()
        .reset_index()
        .drop(['run_name', 'frame'], axis=1)
        .rename(columns={reward_col:f"left_{reward_col}"})
    )

def compare_to_left_min(df, left_min, reward_col):
    return (
        df
        .merge(
            left_min,
            on = 'training_task'
        )
        .assign(**{reward_col:lambda x: x[reward_col]/x[f"left_{reward_col}"]})
        .dropna()
    )

def get_median_vs_left_min(df, reward_col, window, min_over, use_left_min = False, use_left_median=False):

    ## smoothes results using rolling median
    smoothed_df = smooth_results_with_median(
        df=df,
        reward_col=reward_col,
        window=window
    )
    ## gets left aggregation
    if use_left_min and not use_left_median:

        ## minimum value achieved by left
        left_min = get_left_min(
            df=smoothed_df,
            reward_col=reward_col
        )
        return compare_to_left_min(df=smoothed_df, left_min=left_min, reward_col=reward_col)
    elif use_left_median and not use_left_min:

        ## median value achieved by left
        left_min = get_left_median(
            df=smoothed_df,
            reward_col=reward_col,
            min_over=min_over
        )
        return compare_to_left_min(df=smoothed_df, left_min=left_min, reward_col=reward_col)
    elif use_left_median and use_left_min:
        raise ValueError("At least one of use_left_min and use_left_median must be false")
    else:
        return smoothed_df

def get_left_only_vs_initial(
    train_df, 
    test_df, 
    left_eval_df, 
    initial_reward_col, 
    left_only_reward_col, 
    window, 
    min_over,
    use_left_median=False,
    use_left_min=True):
    """
    Gets initial results for bicameral model and joins them with later results for the left hemisphere
    comparisons are made against a left-only baseline
    """
    ## get initial comparison of performance vs left baseline
    initial_comparison = (
        get_median_vs_left_min(
            df = train_df,
            reward_col = initial_reward_col,
            window = window,
            min_over = min_over,
            use_left_median=use_left_median,
            use_left_min=use_left_min
        )
        .loc[:,['training_task', 'run_name', initial_reward_col]]
        .rename(columns={initial_reward_col:f'initial_{initial_reward_col}'})
        .groupby(['training_task', 'run_name'])
        .min()
        .reset_index()
    )

    ## compare left hemisphere vs left baseline performance
    hem_vs_baseline = (
        pd.merge(
            left_eval_df,
            (
                test_df
                .query('run_name == "left_only_double_params"')
                .loc[:,['frame','training_task','reward_mean']]
                .rename(columns={left_only_reward_col:f'left_{left_only_reward_col}'})
            ),
            on=['frame','training_task']
        )
        .query('frame > 4e6') # parametrize?
        .loc[:, ['training_task', 'run_name', left_only_reward_col,f'left_{left_only_reward_col}']]
        .groupby(['training_task', 'run_name'])
        .median()
        .assign(**{f"left_hemisphere_{left_only_reward_col}":lambda x: x[left_only_reward_col] / x[f'left_{left_only_reward_col}']})
        .reset_index()
    )

    return (
        pd.merge(
            hem_vs_baseline,
            initial_comparison,
            on=['training_task', 'run_name']
        )
    )

def agg_left_vs_initial_results(train_df, test_df, left_eval_df, agg_col, initial_reward_col, left_only_reward_col, window, min_over):
    left_vs_initial = get_left_only_vs_initial(
        train_df,
        test_df,
        left_eval_df,
        initial_reward_col=initial_reward_col, 
        left_only_reward_col=left_only_reward_col, 
        window=100, 
        min_over=1e6
    ).assign(tier=lambda x: x.training_task.apply(lambda y: TIER_DICT.get(y)))

    return (
        left_vs_initial
        .loc[:,[agg_col, f"left_hemisphere_{left_only_reward_col}", f"initial_{initial_reward_col}"]]
        .groupby(agg_col)
        .median()
        .reset_index()
    )

### MAIN PLOTTING FUNCTIONS
def plot_reward_trajectory(
    df, reward_col, window, min_over, tasks, to_remove,
    title, ylabel, figsize=(21, 14), use_left_median=False, use_left_min=False):
    """
    Plots a specified reward col with specified smoothing
    Option to present as ratio to left-only baseline results
    """
    to_plot = get_median_vs_left_min(
        df = df.query(f"~run_name.isin({to_remove})"),
        reward_col = reward_col,
        window = window,
        min_over = min_over,
        use_left_median=use_left_median,
        use_left_min=use_left_min
    )

    ## set out plots by task - assume tasks are 3x3 for now
    fig, ax = plt.subplots(3,3, figsize=figsize, sharex=True)
    ax = ax.flatten()
    fig.suptitle(title)

    for i, task in enumerate(tasks):

        sns.lineplot(
            data = to_plot.query(f"training_task=='{task}'"),
            x='frame',
            y=reward_col,
            hue='run_name',
            ax = ax[i],
            palette=PALETTE,
            alpha = .75,
            linewidth=2
        )

        ax[i].set(title=task, xlabel='Environment Steps')
        if i % 3 == 0:
            ax[i].set_ylabel(ylabel)
        else:
            ax[i].set_ylabel('')
        ax[i].set_ylim(0)

        ## show baselines
        # random
        random_mean = (
            df
            .query('run_name=="random"')
            .query(f'training_task=="{task}"')
            .loc[:,reward_col].mean()
        )
        ax[i].axhline(random_mean, c=PALETTE['random'], linewidth=2, alpha=.75)

        # right only double
        random_mean = (
            df
            .query('run_name=="right_only_double_params"')
            .query(f'training_task=="{task}"')
            .loc[:,reward_col].mean()
        )
        ax[i].axhline(random_mean, c=PALETTE['right_only_double_params'], linewidth=2, alpha=.75)
        
        # show left baseline value at 1 if adjust_left
        if use_left_median or use_left_min:
            ax[i].axhline(1, c='black', alpha=.7)
        # if i > 0:
        ax[i].get_legend().remove()

    plt.tight_layout()
    plt.show();

    if use_left_median or use_left_min:
        return to_plot

def plot_inital_reward_vs_final_left(
    train_df, 
    test_df, 
    left_eval_df, 
    initial_reward_col, 
    left_only_reward_col, 
    window, 
    min_over,
    title,
    figsize=(21, 14),
    xlabel=None,
    ylabel=None,
    plot_tiers = False,
    use_left_median=False,
    use_left_min=True
    ):
    """
    Scatter plot which compares initial bicameral results against final left hemisphere results
    """
    left_vs_initial = get_left_only_vs_initial(
        train_df,
        test_df,
        left_eval_df,
        initial_reward_col=initial_reward_col, 
        left_only_reward_col=left_only_reward_col, 
        window=window, 
        min_over=min_over,
        use_left_median=use_left_median,
        use_left_min=use_left_min
    ).assign(tier=lambda x: x.training_task.apply(lambda y: TIER_DICT.get(y)))
    
    
    algorithms = np.unique(left_vs_initial.run_name)
    # hard code grid (1,3) depends on num algorithms
    fig, ax = plt.subplots(1,3,figsize=figsize, sharex=True, sharey=True)
    ax = ax.flatten()

    plt.suptitle(title)

    for i, alg in enumerate(algorithms):
        if plot_tiers:
            sns.scatterplot(
                data=left_vs_initial.query(f'run_name=="{alg}"'),
                x=f'left_hemisphere_{left_only_reward_col}',
                y=f'initial_{initial_reward_col}',
                hue='training_task',
                style='tier',
                palette=TASK_PALETTE,
                ax = ax[i],
                s=200
            )
        
        else:
            sns.scatterplot(
                data=left_vs_initial.query(f'run_name=="{alg}"'),
                x=f'left_hemisphere_{left_only_reward_col}',
                y=f'initial_{initial_reward_col}',
                hue='training_task',
                ax = ax[i],
                s=200
            )
        if i > 0:
            ax[i].get_legend().remove()

        ax[i].set_title(alg)
        if xlabel is not None:
            ax[i].set_xlabel(xlabel)
        if ylabel is not None:
            ax[i].set_ylabel(ylabel)
        ax[i].axhline(1, c='black')
        ax[i].axvline(1, c='black')
        ax[i].axvline(.5, c='black', alpha=0.5)

    plt.tight_layout()
    plt.show();

def plot_agg_left_vs_initial(
    train_df, 
    test_df, 
    left_eval_df, 
    agg_col, 
    initial_reward_col, 
    left_only_reward_col, 
    window, 
    min_over,
    title,
    figsize=(21, 14),
    xlabel=None,
    ylabel=None):
    """
    Aggregate scatter plot with initial vs left
    """
    left_vs_initial = agg_left_vs_initial_results(
        train_df, test_df, left_eval_df, 
        agg_col, initial_reward_col, left_only_reward_col,
        window, min_over)
    fig, ax = plt.subplots(figsize=(21, 10))

    plt.suptitle(title)

    sns.scatterplot(
        data=left_vs_initial,
        x=f'left_hemisphere_{left_only_reward_col}',
        y=f'initial_{initial_reward_col}',
        hue=agg_col,
        ax = ax,
        s=200
    )
        

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.axhline(1, c='black')
    ax.axvline(1, c='black')
    ax.axvline(.5, c='black', alpha=0.5)

    plt.tight_layout()
    plt.show();


    