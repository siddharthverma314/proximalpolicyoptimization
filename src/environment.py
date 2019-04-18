"""Module for handling gym.Env abstraction"""

import gym
import torch


class Environment:
    """Abstraction class for gym.Env"""

    def __init__(self, env_name, device):
        self.env = gym.make(env_name)
        self.device = device

    @staticmethod
    def _process_obs(obs):
        """Process an observation. Must be overrided for special behavior."""
        return obs

    @staticmethod
    def _process_action(action):
        """Process an action. Must be overrided for special behavior."""
        return action.squeeze(0)

    def step(self, action):
        """
        Takes a step in the environment.

        Inputs:
        action - Torch tensor, (1, action_size)

        Returns:
        The tuple observation, reward, finished. Observation and rewards are
        Pytorch tensors on self.device, (1, size).

        """
        action = self._process_action(action).cpu().detach().numpy()

        obs, reward, done, _ = self.env.step(action)
        obs = self._process_obs(obs)

        obs = torch.tensor(obs).float().to(self.device)[None]
        reward = torch.tensor(reward).float().to(self.device)[None, None]

        return obs, reward, done

    def reset(self):
        """
        Reset environment.

        Returns:
        observation - Pytorch tensor on self.device

        """
        obs = self.env.reset()
        obs = self._process_obs(obs)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)[None]
        return obs
