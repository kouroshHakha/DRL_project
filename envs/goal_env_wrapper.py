import gym
import numpy as np
import random

from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv


class GoalEnvWrapper(gym.Wrapper):

    def __init__(self, env, seed=10, mobj=False):
        super(GoalEnvWrapper, self).__init__(env)
        self.multi_goal = mobj
        # assert isinstance(self.env, RobotEnv)
        # self.action_space = self.env.action_space

        # updating to new obs space by concating desired goal and observation
        env_obs_space = env.observation_space.spaces
        new_shape = (env_obs_space['desired_goal'].shape[0]+env_obs_space['observation'].shape[0]+self.action_space.shape[0],)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=new_shape, dtype='float32')

        # from reset with seed 10
        # a = np.array([-0.1526904 ,  0.29178822, -0.12482557,  0.78354603])
        # self.goals = [np.array([1.33726697e+00,  7.58034072e-01,  5.30846141e-01])]
        self.goals = []
        self._seed = None

    def reset(self):
        self.env.seed(self._seed)
        old_obs = self.env.reset()
        if self.multi_goal:
            index =  random.randint(0,len(self.goals)-1)
            self.env.goal = self.goals[index].copy()
            old_obs['desired_goal'] = self.goals[index]
        new_obs = np.concatenate([old_obs['observation'], old_obs['desired_goal'], -1*np.ones(self.action_space.shape[0])])
        return new_obs

    def step(self, action):
        old_obs, rew, done, info = self.env.step(action)
        if info['is_success']:
            done=True
        new_obs = np.concatenate([old_obs['observation'], old_obs['desired_goal'] , action])
        return new_obs, rew, done, info

    def seed(self, _seed=0):
        self._seed = _seed
        random.seed(_seed)
        np.random.seed(_seed)
        self.env.seed(_seed)
