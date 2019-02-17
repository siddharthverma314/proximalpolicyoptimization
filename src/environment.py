import gym

class Environment:
    def __init__(self, name):
        self.env: gym.Env = gym.make(name)

    @staticmethod
    def _process_obs(obs):
        """Process an observation. Must be overrided."""
        raise NotImplementedError

    def step(self, action):
        """
        Takes a step in the environment.

        Inputs:
        action - A numpy array

        Returns:
        The tuple observation, reward, finished. Observation is a numpy array
        
        """
        obs, r, done, _ = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, r, done

    def reset(self):
        """
        Reset environment.

        Returns:
        observation - numpy array

        """
        obs = self.env.reset()
        obs = self._process_obs(obs)
        return obs


class CPEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, 'CartPole-v1')

    @staticmethod
    def _process_obs(obs):
        return obs


class FPPEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, 'FetchPickAndPlace-v1')

    @staticmethod
    def _process_obs(obs):
        obs = np.r_[obs['observation'], obs['achieved_goal'], obs['desired_goal']]
        return obs

