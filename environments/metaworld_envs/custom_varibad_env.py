import gym
import metaworld
import random

from environments.custom_metaworld_benchmark import CustomML10

class CustomML10Env(gym.Env):

    def __init__(self):
        # initialise blank env
        self.benchmark = CustomML10()
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


class CustomML10TestEnv(gym.Env):

    def __init__(self):
        # initialise blank env
        self.benchmark = CustomML10()
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

class ML10Env(gym.Env):

    def __init__(self):
        # initialise blank env
        self.benchmark = metaworld.ML10()
        self.task_names = list(self.benchmark.train_classes.keys())
        self.num_tasks = len(self.task_names)

        # set a dummy task from the benchmark for init purposes
        self.set_benchmark_task(0)

        # metaworld max steps - hardcoded
        self._max_episode_steps = 500

    def set_benchmark_task(self, _task_id):
        task_id = _task_id % self.num_tasks
        self.env_name = self.task_names[task_id]
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
        return self.env_name, self.env_cls
    
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


class ML10TestEnv(gym.Env):

    def __init__(self):
        # initialise blank env
        self.benchmark = metaworld.ML10()
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