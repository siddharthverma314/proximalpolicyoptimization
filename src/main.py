import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
import environment

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

    model = nn.Sequential(
        nn.Linear(31, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 4),
    ).to(device=DEVICE)

elif args.cp:
    env = environment.CPEnvironment()

    class Policy(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.__policy = nn.Sequential(
               nn.Linear(4, 10),
               nn.ReLU(),
               nn.Linear(10, 2),
               nn.Softmax(dim=0),
            )

        def forward(self, state):
            actions = self.__policy(state)
            dist = torch.distributions.Categorical(probs=actions)
            chosen = dist.sample()
            return chosen, actions[chosen]

    model = Policy().to(device=DEVICE)


###################
# DATA COLLECTION #
###################

class Arc:
    """Stores the probability and reward of an arc"""
    def __init__(self):
        self.probs = []
        self.rewards = []

    def __repr__(self):
        return f"Arc({self.probs}, {self.rewards})"


def collect_data():
    """
    Collect data.

    Inputs:
    model - model to use
    step - function that takes an action and returns (observation, reward, finished)
    
    Returns: A list of Arc objects.
    """
    data = []

    for i in range(ITERATIONS):
        obs = env.reset()
        curr_arc = Arc()
        for _ in range(MAX_TIMESTEPS):
            action, prob = model(torch.tensor(obs).float().to(DEVICE))
            obs, r, done = env.step(action.detach().cpu().numpy())

            curr_arc.rewards.append(r)
            curr_arc.probs.append(prob)

            if done:
                break

        data.append(curr_arc)
    return data 


####################
# LOSS CALCULATION #
####################


data = collect_data()
print(data)
