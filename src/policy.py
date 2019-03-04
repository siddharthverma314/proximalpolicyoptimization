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
    def log_prob(probs, action):
        """Return the log probability/density of $$\pi(s|a)$$"""
        raise NotImplementedError

    @staticmethod
    def choice(probs):
        """Return a random action according to the density $$\pi(s|a)$$"""
        raise NotImplementedError


class DiscretePolicy(Policy):
    """Represents a discrete policy"""
    EPSILON = 1e-20

    def __init__(self, policy):
        Policy.__init__(self)
        self.__policy = policy

    def forward(self, state):
        return self.__policy(state)

    @staticmethod
    def prob(probs, action):
        if action.dim() == 0:
            return probs[action]
        elif action.dim() == 1:
            return probs[torch.arange(len(action)), action]

    @staticmethod
    def log_prob(probs, action):
        return torch.log(DiscretePolicy.EPSILON + DiscretePolicy.prob(probs, action))

    @staticmethod
    def choice(probs):
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()


class ContinuousPolicy(Policy):
    """Represents a continuous policy."""

    def __init__(self, policy):
        """
        Defines a Continuous policy. The forward pass consists of three
        modules, shared_nn, mean_nn and var_nn. The computational diagram is as
        follows:

        state ---> shared_nn --+--> mean_nn
                               |
                               |--> var_nn
        """
        Policy.__init__(self)
        self.__policy = policy

    def forward(self, state):
        """Returns (mean, var) tuple"""
        policy = self.__policy(state)
        ng = policy.shape[-1] // 2
        if policy.dim() == 1:
            mean = policy[:ng]
            std = policy[ng:]
        else:
            mean = policy[:, :ng]
            std = policy[:, ng:]
        return mean, torch.abs(std)

    @staticmethod
    def log_prob(params, action):
        dist = torch.distributions.Normal(params[0], params[1])
        probs = dist.log_prob(action)
        if probs.dim() == 2:
            probs = probs.sum(dim=1)
        return probs

    @staticmethod
    def prob(params, action):
        return torch.exp(ContinuousPolicy.log_prob(params, action))

    @staticmethod
    def choice(params):
        dist = torch.distributions.Normal(params[0], params[1])
        return dist.sample()

