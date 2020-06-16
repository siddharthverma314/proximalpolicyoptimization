import torch
from torch import nn
import logger
import data_collection
import ppo
from environment import Environment
from policy import ContinuousPolicy


if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda')

class IPEnvironment(Environment):
    def __init__(self, device):
        super().__init__('Pendulum-v0', device)


log = logger.Logger("../../log")

# HYPERPARAMETERS
MAX_TIMESTEPS = 100
DATA_ITERATIONS = 20
POLICY_ITERATIONS = 100
VALUE_ITERATIONS = 0 #10000
EPSILON = 0.2
EPOCHS = 100
DISCOUNT = 0.8
MAX_KL_DIVERGENCE = 3

env = IPEnvironment(DEVICE)
mean = nn.Sequential(
    nn.Linear(3, 10),
    nn.Tanh(),
    nn.Linear(10, 5),
    nn.Tanh(),
    nn.Linear(5, 1)).to(DEVICE)
std = nn.Sequential(
    nn.Linear(3, 10),
    nn.Tanh(),
    nn.Linear(10, 5),
    nn.Tanh(),
    nn.Linear(5, 1)).to(DEVICE)

policy = ContinuousPolicy.wrap_model(mean, std)

value_fn = nn.Sequential(
    nn.Linear(3, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
).to(DEVICE)

policy_optim = torch.optim.Adam(nn.ModuleList([mean, std]).parameters(), lr=1e-4)
value_optim = torch.optim.Adam(value_fn.parameters(), lr=1e-2)

# data
data = data_collection.DataGenerator(env, DATA_ITERATIONS, 
                                     MAX_TIMESTEPS, DISCOUNT, log)

for i in range(EPOCHS):
    log.log(f"Epoch {i}")
    arc = data.generate(policy, value_fn)
    ppo.optimize_policy(policy, policy_optim, arc, log,
                        POLICY_ITERATIONS, MAX_KL_DIVERGENCE, EPSILON, False)
    #ppo.optimize_value(value_fn, value_optim, arc, log, VALUE_ITERATIONS)
    log.step()
