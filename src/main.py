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

environment = parser.add_mutually_exclusive_group(required=True)
environment.add_argument('--fpp', help="FetchPickAndPlace-v1", action='store_true')
environment.add_argument('--cp', help="CartPole-v1", action='store_true')

parser.add_argument("--show-gui", action='store_true')

parser.add_argument('--timesteps', help="Maximum timesteps for each actor", type=int, default=1000)
parser.add_argument('--iterations', help="Maximum timesteps for each actor", type=int, default=10)

args = parser.parse_args()

###########
# GLOBALS #
###########

# HYPERPARAMETERS
MAX_TIMESTEPS = parser.timesteps
ITERATIONS = parser.iterations

# environment
if args.fpp:
    env = environment.FPPEnvironment()

    model = nn.Sequential(
        nn.Linear(31, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 4),
    )

elif args.cp:
    env: gym.Env = gym.make('CartPole-v1')

    def step(self, action):
        obs, r, done, _ = env.step(action)
        obs = torch.from_numpy(obs).float().cuda()
        return obs, r, done

    class Policy(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.__policy = nn.Sequential(
               nn.Linear(4, 10),
               nn.Softmax(),
               nn.Linear(10, 2),
               nn.Softmax(),
            )

        def forward(self, state):
            state = torch.from_numpy(state).float()
            actions = self.__policy(state)
            dist = torch.distributions.Categorical(probs=actions)
            chosen = dist.sample()
            return chosen, actions[chosen]

    model = Policy()


###################
# DATA COLLECTION #
###################

class Arc:
    """Stores the probability and reward of an arc"""
    def __init__(self):
        self.actions = []
        self.total_r = 0

    def __repr__(self):
        return f"Arc({self.actions}, {self.total_r})"

def collect_data(model, step):
    """
    Collect data.

    Inputs:
    model - model to use
    step - function that takes an action and returns (observation, reward, finished)
    
    Returns: A list of Arc objects.
    """
    data = []
    for i in range(ITERATIONS):
        print('iteration')
        obs = env.reset()
        obs = _process_obs(obs)
        curr_arc = Arc()
        for _ in range(MAX_TIMESTEPS):
            action = model(obs)
            obs, r, done = step(action.detach().cpu().numpy())

            curr_arc.total_r += r
            curr_arc.actions.append(action)

        data.append(curr_arc)
    return data 


if __name__=="__main__":
    model = nn.Sequential(
        nn.Linear(31, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 4),
    ).cuda()

    dc = DataCollection(model)
