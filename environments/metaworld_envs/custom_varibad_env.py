import gym
import metaworld
import random

class ML10Env(gym.Env):

    def __init__(self):
        # initialise blank env
        
        self.benchmark = metaworld.ML10()
        # set a random task from the benchmark
        self.set_task()

        # requires self.set_task()
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
        return self.benchmark.train_classes
    
    ## reset_task is automatically created in make_env using set_task
    def set_task(self, task = None):
        if task is None:
            env_name, env_cls = random.choice(list(self.benchmark.train_classes.items()))
            self.env = env_cls()
            task = random.choice([task for task in self.benchmark.train_tasks if task.env_name==env_name])

        self.task = task
        self.env.set_task(self.task)

    # duplicated for varibad temporarily
    def reset_task(self, task = None):
        if task is None:
            env_name, env_cls = random.choice(list(self.benchmark.train_classes.items()))
            self.env = env_cls()
            task = random.choice([task for task in self.benchmark.train_tasks if task.env_name==env_name])

        self.task = task
        self.env.set_task(self.task)

