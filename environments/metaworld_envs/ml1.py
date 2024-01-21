import gym
import metaworld
import random


class ML1Env(gym.Env):

    def __init__(self):
        ## hardcode this - easy to learn env base on https://jmlr.org/papers/volume22/21-0657/21-0657.pdf
        self.env_name = 'push-v2'
        # initialise blank env
        self.benchmark = metaworld.ML1(self.env_name)
        self.task_names = list(self.benchmark.train_classes.keys())
        self.num_tasks = len(self.task_names)

        self.env_cls = self.benchmark.train_classes[self.env_name]
        self.env = self.env_cls()

        # set a dummy task from the benchmark for init purposes
        self.set_task()
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

class ML1TestEnv(gym.Env):

    def __init__(self):
        ## hardcode this - easy to learn env base on https://jmlr.org/papers/volume22/21-0657/21-0657.pdf
        self.env_name = 'push-v2'
        # initialise blank env
        self.benchmark = metaworld.ML1(self.env_name)
        self.task_names = list(self.benchmark.test_classes.keys())
        self.num_tasks = len(self.task_names)

        self.env_cls = self.benchmark.test_classes[self.env_name]
        self.env = self.env_cls()

        # set a dummy task from the benchmark for init purposes
        self.set_task()
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