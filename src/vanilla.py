import torch
import torch.nn as nn
import gym
import time


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


if __name__ == "__main__":
    TIMESTEP = 5000
    BATCH_SIZE = 5
    ITERATIONS = 1000

    env: gym.Env = gym.make('CartPole-v0')
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    def train_iteration(render=False):

        opt.zero_grad()
        loss = 0
        for _ in range(BATCH_SIZE):
            obs = env.reset()
            cost = 0
            total_r = 0
            for _ in range(TIMESTEP):
                if render:
                    env.render()
                action, prob = policy(obs)
                obs, r, done, _ = env.step(action.item())

                total_r += r
                cost += prob.log()

                if done:
                    break

            loss += -cost * total_r
            print(total_r)

        loss /= BATCH_SIZE
        loss.backward()
        opt.step()

    for _ in range(ITERATIONS):
        for _ in range(BATCH_SIZE):
            train_iteration(False)
    env.close()
