"""
Pointmass env with discrete actions and dense reward
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

class Env(object):
    def __init__(self):
        super(Env, self).__init__()

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def seed(self, seed):
        raise NotImplementedError

class PointMass(Env):
    def __init__(self, max_episode_steps_coeff=1, scale=200, goal_padding=2.0, multi_goal=False, sparse=False):
        super(PointMass, self).__init__()
        # define scale such that the each square in the grid is 1 x 1
        self.scale = int(scale)
        self.grid_size = self.scale * self.scale
        self.multi_goal = multi_goal
        self.sparse = sparse
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, -5.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,5.0, 5.0]))
        self.action_space = gym.spaces.Discrete(48)
        self.action_meaning = [-5,-3,-1,0,1,3,5]
        # self.action_meaning = [-10,-5,-1,0,1,5,10]
        self.boundaries = [[2,8,44,50], [2, 8, 1, 4], [30, 36, 35, 38], [62, 68, 1, 7], [72, 78, 21, 27], [142, 148, 51, 57], [192, 198, 192, 198], [5,10,20,25]] #  #[44, 47, 1, 7],
        self.fixed_goal_idx = 4
        self.spec = EnvSpec(id='PointMass-v4', max_episode_steps=int(max_episode_steps_coeff*self.scale))

    def get_boundary(self, center_x, center_y):
        bound = [center_x-3, center_x+3, center_y-3, center_y+3]
        bound = np.clip(bound, 0, self.scale-1)
        return bound

    def reset(self):
        plt.close()
        self.state = np.array([5, self.scale-10])
        if self.multi_goal == False:
            self.boundary = self.boundaries[self.fixed_goal_idx]
        else:
            self.changing_goal_idx = random.randint(0,len(self.boundaries)-1)
            self.boundary = self.boundaries[self.changing_goal_idx]
            # center = random.randint(0, self.grid_size-1)
            # center_x = center // self.scale
            # center_y = center % 7
            # boundary = [center_x-3, center_x+3, center_y-3, center_y+3]
            # self.boundary = np.clip(boundary, 0, self.scale-1)

        self.ob = np.concatenate([self.state, self.boundary, np.zeros(2)])
        self.x_goal = (self.boundary[0]+self.boundary[1])/2
        self.y_goal = (self.boundary[2]+self.boundary[3])/2
        return self.ob

    def step(self, action):
        x = self.action_meaning[int(action // 7)]
        y = self.action_meaning[int(action % 7)]
        # next state
        new_x = self.state[0]+x
        new_y = self.state[1]+y
        if new_x < 0:
            new_x = 0
        if new_x > self.scale-1:
            new_x = self.scale-1
        if new_y < 0:
            new_y = 0
        if new_y > self.scale-1:
            new_y = self.scale-1
        self.state = np.array([new_x, new_y])
        state = self.state/self.scale

        if (self.boundary[0] <= new_x) and (new_x <= self.boundary[1]) and (self.boundary[2] <= new_y) and (new_y <= self.boundary[3]):
            done = True
            reward = 10
        else:
            done = False
            if self.sparse:
                reward = -1
            else:
                reward = -abs((new_x-self.x_goal)/self.x_goal)-abs((new_y-self.y_goal)/self.y_goal)

        # done
        # done = False
        self.ob = np.concatenate([self.state, self.boundary, np.array([x,y])])
        return self.ob, reward, done, None

    def preprocess(self, state):
        scaled_state = self.scale * state
        x_floor, y_floor = np.floor(scaled_state)
        assert x_floor <= self.scale
        assert y_floor <= self.scale
        if x_floor == self.scale:
            x_floor -= 1
        if y_floor == self.scale:
            y_floor -= 1
        index = self.scale*x_floor + y_floor
        return index

    def unprocess(self, index):
        x_floor = index // self.scale
        y_floor = index % self.scale
        unscaled_state = np.array([x_floor, y_floor])/self.scale
        return unscaled_state

    def seed(self, seed):
        pass

    def render(self):
        # create a grid
        states = [self.state/self.scale]
        indices = np.array([int(self.preprocess(s)) for s in states])
        a = np.zeros(self.grid_size)
        for i in indices:
            a[i] += 1
        max_freq = np.max(a)
        a/=float(max_freq)  # normalize
        a = np.reshape(a, (self.scale, self.scale))
        a[int(self.gob[0])-1][int(self.gob[1])-1] = 1
        ax = sns.heatmap(a)
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    def close(self):
        # Nothing happens
        pass

    def visualize(self, itr, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            ob_array = data['ob']

        ob_array = np.reshape(ob_array, newshape=[-1,self.observation_space.shape[0]])
        first_ob_array = ob_array[:,:2].astype(np.int)
        boundary = ob_array[:, 2:6].astype(np.int)
        # ob_indices = np.array([int(self.preprocess(s)) for s in first_ob_array])

        images = []
        # for cnt, i in zip(range(len(ob_indices)), ob_indices):
        for i in range(int(len(ob_array)*0.2)):
            if not np.any(ob_array[i]):
                continue
            a = np.zeros(shape=[self.scale, self.scale])
            a[first_ob_array[i,0], first_ob_array[i,1]] = 1
            a[boundary[i,0]:(boundary[i,1]+1), boundary[i,2]:(boundary[i,3]+1)]= 0.5
            a = scipy.ndimage.zoom(a, 1, order=0)
            images.append(a)

        dirname = os.path.dirname(fname)
        imageio.mimsave(os.path.join(dirname, '{}.gif'.format(itr)), images)

        if itr.endswith('tg.dpkl'):
            b = np.zeros(shape=[self.scale, self.scale])
            for i in range(len(ob_array)):
                if not np.any(ob_array[i]):
                    continue
                b[boundary[i,0]:(boundary[i,1]+1), boundary[i,2]:(boundary[i,3]+1)]+=1
            b = b/np.max(b)
            imageio.imsave(os.path.join(dirname, '{}.png'.format(itr)), b)

    def create_gif(self, dirname, density=False):
        images = []
        if density:
            filenames = [x for x in os.listdir(dirname) if '_density.png' in x]
            sorted_fnames = sorted(filenames, key=lambda x: int(x.split('_density.png')[0]))
        else:
            filenames = [x for x in os.listdir(dirname) if ('.png' in x and 'density' not in x)]
            sorted_fnames = sorted(filenames, key=lambda x: int(x.split('.png')[0]))
        for f in sorted_fnames:
            images.append(imageio.imread(os.path.join(dirname, f)))
        imageio.mimsave(os.path.join(dirname, 'exploration.gif'), images)

    def create_visualization(self, dirname, density=False):
        for s in os.listdir(dirname):
            path = os.path.join(dirname, s)
            if os.path.isfile(path):
                continue
            iters = [s for s in os.listdir(path) if s.endswith(".dpkl")]
            for i in tqdm(range(len(iters))):
                if not os.path.exists(os.path.join(path, '{}.gif'.format(iters[i]))):
                    self.visualize(iters[i], os.path.join(dirname, s + '/'+ str(iters[i])))
                # self.create_gif(os.path.join(dirname, str(s)))

def debug():
    logdir = 'pm_debug'
    os.mkdir(logdir)
    num_episodes = 10
    num_steps_per_epoch = 20

    env = PointMass()
    for epoch in range(num_episodes):
        states = []
        state = env.reset()
        for i in range(num_steps_per_epoch):
            action = np.random.rand(2)
            state, reward, done, _ = env.step(action)
            states.append(state)
        env.visualize(np.array(states), epoch, logdir)
    env.create_gif(logdir)


if __name__ == "__main__":
    # debug()  # run this if you want to get a feel for how the PointMass environment works (make sure to comment out the code below)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str)
    parser.add_argument('--scale', type=int, default=200)
    args = parser.parse_args()
    env = PointMass(scale=args.scale)
    env.reset()
    env.create_visualization(args.dirname)




