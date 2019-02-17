import torch
import torch.nn as nn


class Policy(nn.Module):
    """Abstract class for a policy"""
    
    @staticmethod
    def log_prob(probs, action):
        """Return the log probability/density of $$\pi(s|a)$$"""
        raise NotImplementedError

    @staticmethod
    def choice(probs):
        """Return a random action according to the density $$\pi(s|a)$$"""
        raise NotImplementedError


class DiscretePolicy(Policy):
    """Represents a discrete policy"""

    def __init__(self, policy):
        self.__policy = policy

    def forward(self, state):
        return self.__policy(state)

    @staticmethod
    def log_prob(probs, action):
        return torch.log(probs[action])

    @staticmethod
    def choice(probs):
        dist = torch.distributions.Categorical(probs)
        return dist.sample()


class ContinuousPolicy(Policy):
    """Represents a continuous policy"""

    def __init__(self, mean_nn, var_nn):
        """policy should return (mean, variance) tuple"""
        self.__mean_nn = mean_nn
        self.__var_nn = var_nn

    def forward(self, state):
        return self.__policy(state)

    @staticmethod
    def log_prob(probs, action):
        dist = torch.distributions.Normal(*probs)
        return dist.log_prob(action)

    @staticmethod
    def choice(probs):
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

