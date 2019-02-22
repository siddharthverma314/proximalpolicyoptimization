import logging as log
import torch

class Arc:
    """Stores the probability and reward of an arc"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []

        self.rewards_to_go = []
        self.advantages = []

    def add_obs(self, state, action, reward, prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob.detach())

    def compute(self, value_fn):
        """Computes rewards to go and advantages and converts data to tensors"""
        # stack tensors
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.rewards = torch.stack(self.rewards)
        self.probs = torch.stack(self.probs)

        # compute rewards to go
        self.rewards_to_go = torch.clone(self.rewards)
        i = len(self.rewards_to_go) - 2
        while i >= 0:
            self.rewards_to_go[i] += self.rewards_to_go[i+1]
            i -= 1

        self.advantages = (self.rewards_to_go - value_fn(self.states).squeeze()).detach()

    def __repr__(self):
        return f"Arc({self.rewards})"


def generate_arc(env, policy, value_fn, max_timesteps) -> Arc:
    """
    Collect data.

    Inputs:
    env - environment to run in
    policy - instance of Policy class
    max_timesteps - number of timesteps to truncate after

    Returns: An arc object
    """

    obs = env.reset()
    arc = Arc()
    total_r = 0

    for _ in range(max_timesteps):
        probs = policy(obs)
        action = policy.choice(probs)
        prob = policy.prob(probs, action)

        obs, r, done = env.step(action.detach().cpu().numpy())
        total_r += r

        arc.add_obs(obs, action, r, prob)

        if done:
            break

    log.debug(f"Collected reward: {total_r}")
    arc.compute(value_fn)
    return arc


def generate_data(env, policy, value_fn, data_iterations, max_timesteps):
    """
    Generates a bunch of data and compiles it.
    
    Inputs:
    env - environment to run in
    policy - nn.Module
    data_iterations - number of experiments
    max_timesteps - timestep cutoff

    Returns: Arc with all the compiled data in it
    """

    result = Arc()

    for _ in range(data_iterations):
        a = generate_arc(env, policy, value_fn, max_timesteps)
        result.states.append(a.states)
        result.actions.append(a.actions)
        result.rewards.append(a.rewards)
        result.probs.append(a.probs)
        result.rewards_to_go.append(a.rewards_to_go)
        result.advantages.append(a.advantages)

    result.states = torch.cat(result.states)
    result.actions = torch.cat(result.actions)
    result.rewards = torch.cat(result.rewards)
    result.probs = torch.cat(result.probs)
    result.rewards_to_go = torch.cat(result.rewards_to_go)
    result.advantages = torch.cat(result.advantages)

    return result

