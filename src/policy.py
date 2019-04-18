import abc
import torch
import torch.nn as nn

class Policy(abc.ABC):
    """Abstract class for a policy"""

    @abc.abstractstaticmethod
    def wrap_model(*args):
        r"""Return a function that returns a policy class given an input by
        applying the model on it and calling the base class"""

    @abc.abstractmethod
    def prob(self, action):
        r"""Return the probability/density of $$\pi(s|a)$$"""

    @abc.abstractmethod
    def log_prob(self, action):
        r"""Return the log probability/density of $$\pi(s|a)$$"""

    @abc.abstractmethod
    def choice(self):
        r"""Return a random action according to $$\pi(s|a)$$"""

    @abc.abstractmethod
    def kl_divergence(self, policy):
        r"""Return the KL divergence between the current and another policy"""


class DiscretePolicy(Policy):
    """Represents a discrete policy"""
    EPSILON = 1e-12

    def __init__(self, probs):
        Policy.__init__(self)
        self.probs = probs
        
    @staticmethod
    def wrap_model(model):
        return lambda x: DiscretePolicy(model(x))

    def prob(self, action):
        action = action.squeeze(1)
        indices = torch.arange(len(action))
        return self.probs[indices, action][..., None]

    def log_prob(self, action):
        return torch.log(self.EPSILON + self.prob(action))

    def choice(self):
        dist = torch.distributions.Categorical(probs=self.probs)
        return dist.sample()[..., None]

    def kl_divergence(self, policy):
        return (self.probs * (torch.log(self.probs) - torch.log(policy.probs))).sum(1).mean()


class ContinuousPolicy(Policy):
    """Represents a gaussian continuous policy."""

    def __init__(self, mean, std):
        Policy.__init__(self)
        self.mean = mean
        self.std = torch.abs(std)

    @staticmethod
    def wrap_model(mean_model, std_model):
        return lambda x: ContinuousPolicy(mean_model(x), std_model(x))

    def log_prob(self, action):
        dist = torch.distributions.Normal(self.mean, self.std)
        probs = dist.log_prob(action)
        # TODO: account for multiple means and variances
        # if probs.dim() == 2:
        #     probs = probs.sum(1)[..., None]
        return probs

    def prob(self, action):
        return torch.exp(self.log_prob(action))

    def choice(self):
        dist = torch.distributions.Normal(self.mean, self.std)
        return dist.sample()

    def kl_divergence(self, policy):
        m1, s1 = self.mean, self.std
        m2, s2 = policy.mean, policy.std
        kld = torch.log(s2/s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5
        return kld.mean()
