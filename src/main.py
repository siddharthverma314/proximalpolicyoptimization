import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import environment
from policy import Policy, DiscretePolicy, ContinuousPolicy


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
parser.add_argument('--data-iterations', help="Number of iterations in data collection", type=int, default=100)
parser.add_argument('--policy-iterations', help="Number of iterations in policy optimization", type=int, default=1000)
parser.add_argument('--policy-batch-size', help="Policy batch size", type=int, default=500)
parser.add_argument('--value-iterations', help="Number of iterations in value optimization", type=int, default=100)
parser.add_argument('--epsilon', help="Epsilon in loss", type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=100)

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
DATA_ITERATIONS = args.data_iterations
POLICY_ITERATIONS = args.policy_iterations
POLICY_BATCH_SIZE = args.policy_batch_size
VALUE_ITERATIONS = args.value_iterations
EPSILON = args.epsilon
EPOCHS = args.epochs

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

    policy = DiscretePolicy(nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        nn.Softmax(dim=0))).to(device=DEVICE)

    value = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.ReLU()
    ).to(device=DEVICE)

policy_optim = torch.optim.SGD(policy.parameters(), 0.04)
value_optim = torch.optim.SGD(value.parameters(), 0.01)

###################
# DATA COLLECTION #
###################


class Arc:
    """Stores the probability and reward of an arc"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []

        self.rewards_to_go = []
        self.advantages = []

    def add_obs(self, state, action, reward, value, prob):
        self.states.append(torch.from_numpy(state).float().to(device=DEVICE))
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.probs.append(prob.detach())

    def compute(self):
        """Computes rewards to go and advantages and converts data to tensors"""
        total = 0
        for i, v in zip(self.rewards[::-1], self.values[::-1]):
            total += i
            self.rewards_to_go.append(total)
            self.advantages.append(total - v)
        self.rewards_to_go.reverse()
        self.advantages.reverse()

        # convert to tensors on DEVICE
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.rewards_to_go = torch.tensor(self.rewards_to_go).float().to(device=DEVICE)
        self.advantages = torch.tensor(self.advantages).float().to(device=DEVICE)
        self.probs = torch.stack(self.probs)

    def __repr__(self):
        return f"Arc({self.rewards})"


def generate_arc(env: environment.Environment, policy) -> Arc:
    """
    Collect data.

    Inputs:
    policy - instance of Policy class
    max_timesteps - number of timesteps to truncate after

    Returns: A list of Arc objects.
    """

    obs = env.reset()
    arc = Arc()

    for _ in range(MAX_TIMESTEPS):
        obs = torch.tensor(obs).float().to(DEVICE)
        probs = policy(obs)
        action = policy.choice(probs)
        prob = policy.prob(probs, action)
        v = value(obs)

        obs, r, done = env.step(action.detach().cpu().numpy())

        arc.add_obs(obs, action, r, v, prob)

        if done:
            break

    arc.compute()
    # DEBUGGING
    print(arc.rewards_to_go[0])
    return arc


def generate_data(env, policy):
    data = []
    for _ in range(DATA_ITERATIONS):
        data.append(generate_arc(env, policy))
    return data


####################
# LOSS CALCULATION #
####################

def L(policy: Policy, arc: Arc):
    p_new = policy.prob(policy(arc.states), arc.actions)
    policy_factor = p_new / arc.probs * arc.advantages

    # compute G
    g = ((arc.advantages >= 0) * 2  - 1).float() * EPSILON
    g = (g + 1) * arc.advantages

    return torch.min(policy_factor, g).mean()


def ppo_clip_loss(policy, arcs):
    # simplification taken from OpenAI spinning up
    loss = 0
    for arc in arcs:
        loss += L(policy, arc)
    loss /= len(arcs)
    return -loss


def optimize_policy(policy, arcs):
    for _ in range(POLICY_ITERATIONS):
        policy_optim.zero_grad()
        value_optim.zero_grad()

        arc_batch = [arcs[random.randrange(len(arcs))] for i in range(POLICY_BATCH_SIZE)]

        loss = ppo_clip_loss(policy, arc_batch)
        loss.backward()

        policy_optim.step()


def value_loss(value, arcs):
    loss = 0
    for arc in arcs:
        v = value(arc.states)
        dot = arc.rewards_to_go - v
        loss += (dot * dot).mean()
    loss /= len(arcs)
    return loss


def optimize_value(value, arcs):
    for _ in range(POLICY_ITERATIONS):
        policy_optim.zero_grad()
        value_optim.zero_grad()

        loss = value_loss(value, arcs)
        loss.backward()

        value_optim.step()


for _ in range(EPOCHS):
    arcs = generate_data(env, policy)
    print(ppo_clip_loss(policy, arcs))
    print(value_loss(value, arcs))
    optimize_policy(policy, arcs)
    optimize_value(value, arcs)
    print(ppo_clip_loss(policy, arcs))
    print(value_loss(value, arcs))
