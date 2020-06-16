"""Handles collecting data from a policy"""

from __future__ import annotations
from typing import Dict, List, Union
import torch
from torch import tensor, nn
import numpy as np
from logger import Logger
from environment import Environment


class Arc:
    """Stores data from one iteration of an environment"""

    def __init__(self, gamma: float):
        self.data: Dict[str, Union[List[tensor], tensor]] = dict()
        for var in "sarpRA":
            self.data[var] = []
        self.gamma = gamma

    def add_obs(self, state, action, reward, prob):
        """
        Add an observation to the arc

        :param state: state tensor
        :param action: action tensor
        :param reward: reward tensor
        :param prob: probability of taking action tensor
        """
        for var, val in ('s', state), ('a', action), ('r', reward), ('p', prob):
            self.data[var].append(val)

    def compute(self, value_fn):
        """
        Computes rewards to go and advantages. Also converts data to tensors

        :param value_fn: subclass of nn.Module
        """

        # stack tensors
        for var in "sarp":
            self.data[var] = torch.cat(self.data[var])

        # compute rewards to go
        self.data['R'] = torch.clone(self.data['r'])
        i = self.data['R'].shape[0] - 2
        while i >= 0:
            self.data['R'][i, 0] += self.gamma * self.data['R'][i+1, 0]
            i -= 1

        #TODO: Work on new advantage function
        self.data['A'] = self.data['R']# - value_fn(self.data['s'])

        # detach everything
        for var in "sarp":
            self.data[var] = self.data[var].detach()

    def concat(self, arc):
        for var in "sarpRA":
            self.data[var].append(arc.data[var])

    def torchify(self):
        for var in "sarpRA":
            self.data[var] = torch.cat(self.data[var]).detach()

    @property
    def states(self):
        return self.data['s']

    @property
    def actions(self):
        return self.data['a']

    @property
    def rewards(self):
        return self.data['r']

    @property
    def rewards_to_go(self):
        return self.data['R']

    @property
    def probs(self):
        return self.data['p']

    @property
    def advantages(self):
        return self.data['A']


class DataGenerator:
    """Generates data in the form of Arc classes by calling .generate() method"""

    def __init__(self,
                 env: Environment,
                 data_iterations,
                 max_timesteps,
                 gamma,
                 logger: Logger):
        """
        Initialize a new Data Generator.

        :param env: the environment to run the data collection in. Subclass of
        environment.Environment
        :param data_iterations: the number of runs to do
        :param max_timesteps: timestep limit
        :param gamma: decay rate
        :param logger: instance of Logger class
        """
        self.env = env
        self.data_iterations = data_iterations
        self.max_timesteps = max_timesteps
        self.gamma = gamma
        self.logger = logger

    def _generate_arc(self, policy, value_fn):
        obs = self.env.reset()
        arc = Arc(self.gamma)
        total_r = 0

        for _ in range(self.max_timesteps):
            probs = policy(obs)
            action = probs.choice()
            prob = probs.prob(action)

            obs_new, reward, done = self.env.step(action)
            total_r += reward

            arc.add_obs(obs, action, reward, prob)
            obs = obs_new

            if done:
                break

        arc.compute(value_fn)
        return arc, total_r

    def generate(self, policy, value_fn):
        """
        Generates a bunch of data and compiles it

        :param policy: wrapped policy function neural network
        :param value_fn: value function neural network
        """
        rewards = []
        arcs = Arc(self.gamma)
        with torch.no_grad():
            for _ in range(self.data_iterations):
                arc, total_r = self._generate_arc(policy, value_fn)
                arcs.concat(arc)

                # logging
                rewards.append(total_r)
                self.logger.log(f"Collected reward: {total_r}")

            arcs.torchify()

        # logging
        rewards = np.array(rewards)
        data = dict()
        data['data_reward_avg'] = rewards.mean()
        data['data_reward_std'] = rewards.std()
        data['data_reward_min'] = rewards.min()
        data['data_reward_max'] = rewards.max()
        self.logger.update(data)

        return arcs
