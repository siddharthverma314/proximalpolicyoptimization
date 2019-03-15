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

    @staticmethod
    def kl_divergence(probs1, probs2):
        """Return a state of the current policy"""
        raise NotImplementedError


class DiscretePolicy(Policy):
    """Represents a discrete policy"""
    EPSILON = 1e-12

    def __init__(self, policy):
        Policy.__init__(self)
        self.__policy = policy

    def forward(self, state):
        return self.__policy(state)

    @staticmethod
    def prob(probs, action):
        action = action.squeeze(1)
        indices = torch.arange(len(action))
        return probs[indices, action][..., None]

    @staticmethod
    def log_prob(probs, action):
        return torch.log(DiscretePolicy.EPSILON + DiscretePolicy.prob(probs, action))

    @staticmethod
    def choice(probs):
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()[..., None]

    @staticmethod
    def kl_divergence(probs1, probs2):
        return (probs1 * torch.log(probs1 / probs2)).sum(1).mean()


class ContinuousPolicy(Policy):
    """Represents a continuous policy."""

    def __init__(self, mean, std):
        """
        Defines a Continuous policy. The neural networks are used as follows
        """

        Policy.__init__(self)
        self.__mean = mean
        self.__std = std

    def forward(self, state):
        """Returns (mean, var) tuple"""
        mean = self.__mean(state)
        std = self.__std(state)
        return mean, std * std

    @staticmethod
    def log_prob(params, action):
        dist = torch.distributions.Normal(params[0], params[1])
        probs = dist.log_prob(action)
        if probs.dim() == 2:
            probs = probs.sum(1)[..., None]
        return probs

    @staticmethod
    def prob(params, action):
        return torch.exp(ContinuousPolicy.log_prob(params, action))

    @staticmethod
    def kl_divergence(params1, params2):
        m1, s1 = params1
        m2, s2 = params2
        kld = torch.log(s2/s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5
        return kld.mean()

    @staticmethod
    def choice(params):
        dist = torch.distributions.Normal(params[0], params[1])
        return dist.sample()

