import gym
import numpy as np

from copy import deepcopy
from typing import Any, Dict, List, Tuple

class ContinualEnv(gym.Env):
    """
    Based on continual world env design:
    https://github.com/awarelab/continual_world/blob/main/continualworld/envs.py
    """
    def __init__(self, envs: List[gym.Env], steps_per_env: int):

        ## good check to do
        for i in range(len(envs)):
            assert envs[0].action_space == envs[i].action_space

        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)

        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _get_envs(self):
        return self.envs
    
    def _get_env_ids(self):
        return [i for i in len(self.envs)]

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:

        # step
        obs, reward, terminated, truncated, info = self.envs[self.cur_seq_idx].step(action)
        done = terminated or truncated
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            done = True
            info["TimeLimit.truncated"] = True

            self.cur_seq_idx += 1

        ## add done flag
        to_append = 1 if done else 0
        obs = np.append(obs, to_append).reshape(1, -1)

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        obs, _ = self.envs[self.cur_seq_idx].reset()
        # add done flag
        obs = np.append(obs, 0).reshape(1, -1)
        return obs
    
    # def update_cur_step(self, value: int):
    #     """
    #     Manually update cur_step
    #     Can help manage vec envs        
    #     """
    #     self.cur_step = value

    # def update_cur_seq_idx(self, value: int):
    #     """
    #     Manually update cur_seq_idx
    #     Can help manage vec envs
    #     """
    #     self.cur_seq_idx = value
