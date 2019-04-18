import argparse
import datetime
import os
import logging as log
import torch
import torch.nn as nn
import data_collection
import environment
from policy import DiscretePolicy, ContinuousPolicy


if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda')

class IPEnvironment(Environment):
    ENVIRONMENT = 'Pendulum-v0'

log = logger.Logger("../../log")

# HYPERPARAMETERS
MAX_TIMESTEPS = 1000
DATA_ITERATIONS = 100
POLICY_ITERATIONS = 100
VALUE_ITERATIONS = 100
EPSILON = 0.1
EPOCHS = 100
DISCOUNT = 0.9
MAX_KL_DIVERGENCE = 10

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
