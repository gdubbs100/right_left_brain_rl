import gym
import metaworld
import random

from environments.custom_metaworld_benchmark import ML3

class ML3SingleEnv(gym.Env):

    def __init__(self, task_name, train=False):
        self.benchmark= ML3()
        self.train = train
        self.env_name = task_name
        if train:
            self.tasks = [task for task in self.benchmark.train_tasks if task.env_name==self.env_name]
            self.env_cls = self.benchmark.train_classes[self.env_name]
        else:
            self.tasks = [task for task in self.benchmark.test_tasks if task.env_name==self.env_name]
            self.env_cls = self.benchmark.test_classes[self.env_name]

        ## set the task
        self.env = self.env_cls()
        self.task = random.choice(self.tasks)
        self.env.set_task(self.task)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # metaworld max steps - hardcoded
        self._max_episode_steps = 500

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['task'] = self.task
        return obs, reward, done, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def get_task(self):
        return self.task
    
    ## reset_task is automatically created in make_env using set_task
    def set_task(self, task = None):
        if task is None:
            if self.train:
                task = random.choice(
                    [task for task in self.benchmark.train_tasks if task.env_name==self.env_name]
                    )
            else: 
                task = random.choice(
                    [task for task in self.benchmark.test_tasks if task.env_name==self.env_name]
                    )

        self.task = task
        self.env.set_task(self.task)

    # duplicated for varibad temporarily
    def reset_task(self, task = None):
        if task is None:
            if self.train:
                task = random.choice(
                    [task for task in self.benchmark.train_tasks if task.env_name==self.env_name]
                    )
            else: 
                task = random.choice(
                    [task for task in self.benchmark.test_tasks if task.env_name==self.env_name]
                    )

        self.task = task
        self.env.set_task(self.task)

class ML3Env(gym.Env):

    def __init__(self):
        # initialise blank env
        self.benchmark = ML3()
        self.task_names = list(self.benchmark.train_classes.keys())
        self.num_tasks = len(self.task_names)

        # set a dummy task from the benchmark for init purposes
        self.set_benchmark_task(0)

        # metaworld max steps - hardcoded
        self._max_episode_steps = 500

    def set_benchmark_task(self, _task_id):
        self.task_id = _task_id % self.num_tasks
        self.env_name = self.task_names[self.task_id]
        self.env_cls = self.benchmark.train_classes[self.env_name]
        self.env = self.env_cls()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['task'] = self.task
        return obs, reward, done, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def get_task(self):
        return self.task_id
    
    # def get_task(self):
    #     return self.env_name, self.env_cls
    
    ## reset_task is automatically created in make_env using set_task
    def set_task(self, task = None):
        if task is None:
            task = random.choice(
                [task for task in self.benchmark.train_tasks if task.env_name==self.env_name]
                )

        self.task = task
        self.env.set_task(self.task)

    # duplicated for varibad temporarily
    def reset_task(self, task = None):
        if task is None:
            task = random.choice(
                [task for task in self.benchmark.train_tasks if task.env_name==self.env_name]
                )

        self.task = task
        self.env.set_task(self.task)


class ML3TestEnv(gym.Env):

    def __init__(self):
        # initialise blank env
        self.benchmark = ML3()
        self.task_names = list(self.benchmark.test_classes.keys())
        self.num_tasks = len(self.task_names)

        # set a dummy task from the benchmark for init purposes
        self.set_benchmark_task(0)

        # metaworld max steps - hardcoded
        self._max_episode_steps = 500

    def set_benchmark_task(self, _task_id):
        task_id = _task_id % self.num_tasks
        self.env_name = self.task_names[task_id]
        self.env_cls = self.benchmark.test_classes[self.env_name]
        self.env = self.env_cls()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['task'] = self.task
        return obs, reward, done, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def get_task(self):
        return self.env_name, self.env_cls
    
    ## reset_task is automatically created in make_env using set_task
    def set_task(self, task = None):
        if task is None:
            task = random.choice(
                [task for task in self.benchmark.test_tasks if task.env_name==self.env_name]
                )

        self.task = task
        self.env.set_task(self.task)

    # duplicated for varibad temporarily
    def reset_task(self, task = None):
        if task is None:
            task = random.choice(
                [task for task in self.benchmark.test_tasks if task.env_name==self.env_name]
                )

        self.task = task
        self.env.set_task(self.task)
