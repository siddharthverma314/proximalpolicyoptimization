import argparse
import datetime
import os
import logging as log
import torch
import torch.nn as nn
import data_collection
import environment
from policy import DiscretePolicy, ContinuousPolicy


##########
# PARSER #
##########

parser = argparse.ArgumentParser()

environ = parser.add_mutually_exclusive_group(required=True)
environ.add_argument('--fpp', help="FetchPickAndPlace-v1", action='store_true')
environ.add_argument('--cp', help="CartPole-v1", action='store_true')
environ.add_argument('--ip', help="Pendulum-v0", action='store_true')

parser.add_argument("--show-gui", action='store_true')
parser.add_argument("--no-gpu", action='store_true')

parser.add_argument('--timesteps', help="Maximum timesteps for each actor", type=int, default=1000)
parser.add_argument('--data-iterations', help="Number of iterations in data collection", type=int, default=100)
parser.add_argument('--policy-iterations', help="Number of iterations in policy optimization", type=int, default=100)
parser.add_argument('--value-iterations', help="Number of iterations in value optimization", type=int, default=100)
parser.add_argument('--max-kl', help="Maximum KL Divergence in policy training", type=float, default=20.)
parser.add_argument('--epsilon', help="Epsilon in loss", type=float, default=0.2)
parser.add_argument('--discount', help="Discount factor", type=float, default=0.8)
parser.add_argument('--log', help="Whether to store a log", action='store_true')
parser.add_argument('--epochs', type=int, default=50)

args = parser.parse_args()


###########
# GLOBALS #
###########

if args.log:
    name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.mkdir(f"../log/{name}")
    log.basicConfig(filename=f"../log/{name}/log", level=log.DEBUG)
else:
    log.basicConfig(level=log.DEBUG)

log.info(f"Started program at {str(datetime.datetime.now())}")
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
MAX_KL_DIVERGENCE = args.max_kl

# environment
if args.fpp:
    # TODO: Edit this
    env = environment.FPPEnvironment(DEVICE)

    policy = ContinuousPolicy(
            nn.Sequential(
                nn.Linear(31, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 2)
            )).to(DEVICE)

    value = nn.Sequential(
        nn.Linear(31, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.ReLU()).to(DEVICE)


elif args.cp:
    env = environment.CPEnvironment(DEVICE)

    policy = DiscretePolicy(nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.Softmax(dim=-1))).to(DEVICE)

    value = nn.Sequential(
        nn.Linear(4, 5),
        nn.Sigmoid(),
        nn.Linear(5, 1),
        nn.ReLU()
    ).to(DEVICE)


elif args.ip:
    env = environment.IPEnvironment(DEVICE)

    mean = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))
    std = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))

    policy = ContinuousPolicy(mean, std).to(DEVICE)

    value = nn.Sequential(
        nn.Linear(3, 5),
        nn.Sigmoid(),
        nn.Linear(5, 1),
        nn.ReLU()
    ).to(DEVICE)

policy_optim = torch.optim.Adam(policy.parameters())
value_optim = torch.optim.Adam(value.parameters())


####################
# LOSS CALCULATION #
####################

def ppo_clip_loss(policy, arc):
    # simplification taken from OpenAI spinning up
    r = policy.prob(policy(arc.states), arc.actions) / arc.probs
    policy_factor = r * arc.advantages

    # compute g
    g = ((arc.advantages >= 0).float() * 2  - 1) * EPSILON + 1
    g = (g * arc.advantages).detach()

    return torch.min(policy_factor, g).mean()

def vpg_loss(policy, arc):
    probs = policy.log_prob(policy(arc.states), arc.actions)
    return (probs * arc.advantages).mean()

def optimize_policy(policy, arc):
    previous_policy = policy(arc.states)
    for _ in range(POLICY_ITERATIONS):
        loss = ppo_clip_loss(policy, arc)
        #loss = vpg_loss(policy, arc)
        log.debug(f"Policy Loss: {loss.item()}")
        policy_optim.zero_grad()
        (-loss).backward()
        policy_optim.step()
        kld = policy.kl_divergence(previous_policy, policy(arc.states))
        log.info(f"KL Divergence: {kld}")
        if kld > MAX_KL_DIVERGENCE:
            break

def value_loss(value, arc):
    v = value(arc.states).squeeze()
    dot = (arc.rewards_to_go - v)**2
    loss = dot.mean()
    return loss


def optimize_value(value, arc):
    for _ in range(VALUE_ITERATIONS):
        loss = value_loss(value, arc)
        log.debug(f"Value Loss: {loss.item()}")
        value_optim.zero_grad()
        loss.backward()
        value_optim.step()


for i in range(EPOCHS):
    log.info(f"Epoch {i}")
    log.info("Collecting new data")
    arc = data_collection.generate_data(env, policy, value, DATA_ITERATIONS, MAX_TIMESTEPS, DISCOUNT)
    log.info("Optimizing policy")
    optimize_policy(policy, arc)
    log.info("Optimizing value function")
    optimize_value(value, arc)
