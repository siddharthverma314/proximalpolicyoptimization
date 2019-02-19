import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
import environment
import policy


##########
# PARSER #
##########

parser = argparse.ArgumentParser()

environ = parser.add_mutually_exclusive_group(required=True)
environ.add_argument('--fpp', help="FetchPickAndPlace-v1", action='store_true')
environ.add_argument('--cp', help="CartPole-v1", action='store_true')

parser.add_argument("--show-gui", action='store_true')
parser.add_argument("--no-gpu", action='store_true')

parser.add_argument('--timesteps', help="Maximum timesteps for each actor", type=int, default=1000)
parser.add_argument('--iterations', help="Maximum timesteps for each actor", type=int, default=100)

args = parser.parse_args()


###########
# GLOBALS #
###########

if args.no_gpu and torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda')

# HYPERPARAMETERS
MAX_TIMESTEPS = args.timesteps
ITERATIONS = args.iterations

# environment
if args.fpp:
    env = environment.FPPEnvironment()

    class Policy(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.__policy = nn.Sequential(
                nn.Linear(31, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 4),
            )

        def forward(self, state):
            actions = self.__policy(state)
            dist = torch.distributions.Categorical(probs=actions)
            chosen = dist.sample()
            return chosen, actions[chosen]

    policy = Policy().to(device=DEVICE)
    value = nn.Sequential(
        nn.Linear(31, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 4),
    ).to(device=DEVICE)


elif args.cp:
    env = environment.CPEnvironment()

    policy = policy.DiscretePolicy(nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 2), nn.Softmax(dim=0)))

    policy = Policy().to(device=DEVICE)
    value = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.ReLU()
    ).to(device=DEVICE)


###################
# DATA COLLECTION #
###################


class Arc:
    """Stores the probability and reward of an arc"""
    def __init__(self):
        self.values = []
        self.probs = []
        self.rewards = []

    def __repr__(self):
        return f"Arc({self.probs}, {self.rewards})"


def generate_arc(env, policy, max_timesteps=None):
    """
    Collect data.

    Inputs:
    policy - policy to use that returns (action, prob) tuple
    max_timesteps - number of timesteps to truncate after

    Returns: A list of Arc objects.
    """

    if not max_timesteps:
        max_timesteps = MAX_TIMESTEPS

    obs = env.reset()
    arc = Arc()

    for _ in range(max_timesteps):
        obs = torch.tensor(obs).float().to(DEVICE)
        action, prob = policy(obs)
        v = value(obs)

        obs, r, done = env.step(action.detach().cpu().numpy())

        arc.rewards.append(r)
        arc.probs.append(prob)
        arc.values.append(v.detach())

        if done:
            break

    process_arc(arc)
    return arc


def process_arc(arc: Arc):
    """Mutates an arc such that arc.rewards is advantages"""
    # compute rewards to go
    rtg = []
    total = 0
    for i, v in zip(arc.rewards[::-1], arc.values[::-1]):
        total += i
        rtg.append(total - v)
    arc.rewards = rtg[::-1]


def generate_data(env, policy, iterations=None, max_timesteps=None):
    if not iterations:
        iterations = ITERATIONS

    data = []
    for _ in range(iterations):
        data.append(generate_arc(env, policy, max_timesteps))
    return data


####################
# LOSS CALCULATION #
####################

def ppo_clip_loss(policy, env):
    policy(env.

print(generate_data(env, policy, 1, 10))
