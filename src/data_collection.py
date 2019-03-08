import logging as log
import torch


class Arc:
    """Stores the probability and reward of an arc"""
    def __init__(self, gamma):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []

        self.rewards_to_go = []
        self.advantages = []
        self.gamma = gamma

    def add_obs(self, state, action, reward, prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob.detach())

    def compute(self, value_fn):
        """Computes rewards to go and advantages and converts data to tensors"""
        # stack tensors
        self.states = torch.cat(self.states).detach()
        self.actions = torch.cat(self.actions).detach()
        self.rewards = torch.cat(self.rewards).detach()
        self.probs = torch.cat(self.probs).detach()

        # compute rewards to go
        self.rewards_to_go = torch.clone(self.rewards)
        i = self.rewards_to_go.shape[0] - 2
        while i >= 0:
            self.rewards_to_go[i, 0] += self.gamma * self.rewards_to_go[i+1, 0]
            i -= 1

        #TODO: Work on new advantage function
        self.advantages = self.rewards_to_go - value_fn(self.states)

    def concat(self, arc):
        self.states.append(arc.states)
        self.actions.append(arc.actions)
        self.rewards.append(arc.rewards)
        self.probs.append(arc.probs)
        self.rewards_to_go.append(arc.rewards_to_go)
        self.advantages.append(arc.advantages)

    def torchify(self):
        self.states = torch.cat(self.states).detach()
        self.actions = torch.cat(self.actions).detach()
        self.rewards = torch.cat(self.rewards).detach()
        self.probs = torch.cat(self.probs).detach()
        self.rewards_to_go = torch.cat(self.rewards_to_go).detach()
        self.advantages = torch.cat(self.advantages).detach()


def generate_arc(env, policy, value_fn, max_timesteps, gamma) -> Arc:
    """
    Collect data.

    Inputs:
    env - environment to run in
    policy - instance of Policy class
    max_timesteps - number of timesteps to truncate after

    Returns: An arc object
    """

    obs = env.reset()
    arc = Arc(gamma)
    total_r = 0

    for _ in range(max_timesteps):
        probs = policy(obs)

        action = policy.choice(probs)
        prob = policy.prob(probs, action)

        obs_new, reward, done = env.step(action)
        total_r += reward

        arc.add_obs(obs, action, reward, prob)
        obs = obs_new

        if done:
            break

    log.debug(f"Collected reward: {total_r.item()}")
    arc.compute(value_fn)
    return arc


def generate_data(env, policy, value_fn, data_iterations, max_timesteps, gamma):
    """
    Generates a bunch of data and compiles it.
    
    Inputs:
    env - environment to run in
    policy - nn.Module
    data_iterations - number of experiments
    max_timesteps - timestep cutoff

    Returns: Arc with all the compiled data in it
    """

    arc = Arc(gamma)
    with torch.no_grad():
        for _ in range(data_iterations):
            res = generate_arc(env, policy, value_fn, max_timesteps, gamma)
            arc.concat(res)

        arc.torchify()
    return arc

