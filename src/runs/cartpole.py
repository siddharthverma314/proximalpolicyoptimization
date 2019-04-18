import argparse
import datetime
import os
import torch
import torch.nn as nn
import data_collection
import environment
import logger
import ppo
from policy import DiscretePolicy


if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

class CPEnvironment(environment.Environment):
    def __init__(self):
        super().__init__('CartPole-v1', DEVICE)

    @staticmethod
    def _process_action(action):
        return action.squeeze()

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

# models
env = CPEnvironment()
policy_fn = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.Softmax(dim=-1)
).to(DEVICE)
policy = DiscretePolicy.wrap_model(policy_fn)
value_fn = nn.Sequential(
    nn.Linear(4, 5),
    nn.Sigmoid(),
    nn.Linear(5, 1),
    nn.ReLU()
).to(DEVICE)

#optimizers
policy_optim = torch.optim.Adam(policy_fn.parameters())
value_optim = torch.optim.Adam(value_fn.parameters())

# data
data = data_collection.DataGenerator(env, DATA_ITERATIONS, 
                                     MAX_TIMESTEPS, DISCOUNT, log)

for i in range(EPOCHS):
    log.log(f"Epoch {i}")
    arc = data.generate(policy, value_fn)
    ppo.optimize_policy(policy, policy_optim, arc, log,
                        POLICY_ITERATIONS, MAX_KL_DIVERGENCE, EPSILON)
    ppo.optimize_value(value_fn, value_optim, arc, log, VALUE_ITERATIONS)
    log.step()
