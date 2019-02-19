import torch
import torch.nn as nn


class Policy(nn.Module):
    """Abstract class for a policy"""

    def __init__(self):
        nn.Module.__init__(self)
    
    @staticmethod
    def prob(probs, action):
        """Return the probability/density of $$\pi(s|a)$$"""
        raise NotImplementedError

    @staticmethod
    def prob(probs, action):
        """Return the log probability/density of $$\pi(s|a)$$"""
        raise NotImplementedError

    @staticmethod
    def choice(probs):
        """Return a random action according to the density $$\pi(s|a)$$"""
        raise NotImplementedError


class DiscretePolicy(Policy):
    """Represents a discrete policy"""

    def __init__(self, policy):
        Policy.__init__(self)
        self.__policy = policy

    def forward(self, state):
        return self.__policy(state)

    @staticmethod
    def prob(probs, action):
        return probs[action]

    @staticmethod
    def log_prob(probs, action):
        return torch.log(DiscretePolicy.prob(probs, action))

    @staticmethod
    def choice(probs):
        dist = torch.distributions.Categorical(probs)
        return dist.sample()


class ContinuousPolicy(Policy):
    """Represents a continuous policy."""

    def __init__(self, shared_nn, mean_nn, var_nn):
        """
        Defines a Continuous policy. The forward pass consists of three
        modules, shared_nn, mean_nn and var_nn. The computational diagram is as
        follows:

        state ---> shared_nn --+--> mean_nn
                               |
                               |--> var_nn
        """
        Policy.__init__(self)
        self.__shared_nn = shared_nn
        self.__mean_nn = mean_nn
        self.__var_nn = var_nn

    def forward(self, state):
        """Returns (mean, var) tuple"""
        shared = self.shared_nn(state)
        return self.__mean_nn(shared), self.__var_nn(shared)

    @staticmethod
    def log_prob(probs, action):
        dist = torch.distributions.Normal(*probs)
        return dist.log_prob(action)

    @staticmethod
    def prob(probs, action):
        return torch.exp(ContinuousPolicy.log_prob(probs, action))

    @staticmethod
    def choice(probs):
        dist = torch.distributions.Normal(*probs)
        return dist.sample()

