import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
import environment
import data_collection
from policy import Policy, DiscretePolicy, ContinuousPolicy
import logging as log
import datetime
import os

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
parser.add_argument('--value-iterations', help="Number of iterations in value optimization", type=int, default=100)
parser.add_argument('--epsilon', help="Epsilon in loss", type=float, default=1e-2)
parser.add_argument('--discount', help="Discount factor", type=float, default=0.8)
parser.add_argument('--log', help="Whether to store a log", default='store_true')
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()


###########
# GLOBALS #
###########

if args.log:
    name = datetime.datetime.now().strftime("%Y%m%d%H%m%S")
    os.mkdir(f"log/{name}")
    log.basicConfig(filename=f"log/{name}/log", level=log.DEBUG)
else:
    log.basicConfig(level=log.DEBUG)

log.info("Started program at {str(datetime.datetime.now())}")
log.info(f"Args: {str(args)}")

if args.no_gpu and torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda')

# HYPERPARAMETERS
MAX_TIMESTEPS = args.timesteps
DATA_ITERATIONS = args.data_iterations
POLICY_ITERATIONS = args.policy_iterations
VALUE_ITERATIONS = args.value_iterations
EPSILON = args.epsilon
EPOCHS = args.epochs
DISCOUNT = args.discount

# environment
if args.fpp:
    # TODO: Edit this
    env = environment.FPPEnvironment(DEVICE)

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

    policy = Policy().to(DEVICE)
    value = nn.Sequential(
        nn.Linear(31, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 4),
    ).to(DEVICE)


elif args.cp:
    env = environment.CPEnvironment(DEVICE)

    policy = DiscretePolicy(nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        nn.Softmax(dim=0))).to(DEVICE)

    value = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.ReLU()
    ).to(DEVICE)

policy_optim = torch.optim.SGD(policy.parameters(), 0.04)
value_optim = torch.optim.SGD(value.parameters(), 0.01)

####################
# LOSS CALCULATION #
####################

def ppo_clip_loss(policy, arc):
    # simplification taken from OpenAI spinning up
    p_new = policy.prob(policy(arc.states), arc.actions)
    policy_factor = p_new / arc.probs * arc.advantages

    # compute g
    g = ((arc.advantages >= 0) * 2  - 1).float() * EPSILON
    g = (g + 1) * arc.advantages

    loss = torch.min(policy_factor, g).mean()
    log.debug(f"Policy Loss: {loss.item()}")
    return -loss


def optimize_policy(policy, arc):
    for _ in range(POLICY_ITERATIONS):
        loss = ppo_clip_loss(policy, arc)
        loss.backward()
        policy_optim.step()


def value_loss(value, arc):
    v = value(arc.states)
    dot = (arc.rewards_to_go - v)**2
    loss = dot.mean()
    log.debug(f"Value Loss: {loss.item()}")
    return loss


def optimize_value(value, arc):
    for _ in range(POLICY_ITERATIONS):
        loss = value_loss(value, arc)
        loss.backward()
        value_optim.step()


for i in range(EPOCHS):
    log.info(f"Epoch {i}")
    log.info("Collecting new data")
    arc = data_collection.generate_data(env, policy, value, DATA_ITERATIONS, MAX_TIMESTEPS)
    log.info("Optimizing policy")
    optimize_policy(policy, arc)
    log.info("Optimizing value function")
    optimize_value(value, arc)
