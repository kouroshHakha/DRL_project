"""
Pointmass3d env with discrete actions and dense/sparse reward
"""
import gym
from gym.envs.registration import EnvSpec
import imageio
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import pickle
import IPython
import time
import scipy.ndimage
import random

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class PointMass3dd(object):
    def __init__(self, scale=200, multi_goal=False, sparse=False, threshold = 10):
        self.scale = int(scale)
        self.grid_size = self.scale ** 3
        self.multi_goal = multi_goal
        self.sparse = sparse
        self.action_meaning = [-10,-1,1,10]
        self.action_space = gym.spaces.Discrete(len(self.action_meaning)**3)
        self.observation_space = gym.spaces.Box(-np.inf,np.inf, shape=(9,))
        self.goals = [[scale-threshold]*3]
        self.spec = EnvSpec(id='PointMass3DDiscrete-v0', max_episode_steps=int(self.scale))
        self.threshold = threshold

    def reset(self):
        self.state = np.array([0,0,0])
        if self.multi_goal == False:
            self.goal = np.array(self.goals[0])
        else:
            index = random.randint(0,len(self.goals)-1)
            self.goal = np.array(self.goals[index])

        self.ob = np.concatenate([self.state, self.goal, np.zeros(3)])
        return self.ob

    def step(self, action):
        x = self.action_meaning[int(action % 4)]
        y = self.action_meaning[int((action // 4) % 4)]
        z = self.action_meaning[int((action // 4**2))]

        # next state
        self.state = self.state + np.array([x,y,z])
        self.state = np.clip(self.state, 0, self.scale-1)

        distance = goal_distance(self.state, self.goal)
        if distance < self.threshold:
            done = True
            reward = 10
        else:
            done = False
            if self.sparse:
                reward = -1
            else:
                reward = -distance

        self.ob = np.concatenate([self.state, self.goal, np.array([x,y,z])])
        return self.ob, reward, done, None

    def seed(self, seed):
        pass



