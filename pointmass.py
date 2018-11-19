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
    def __init__(self, max_episode_steps_coeff=1, scale=20, goal_padding=2.0):
        super(PointMass, self).__init__()
        # define scale such that the each square in the grid is 1 x 1
        self.scale = int(scale)
        self.grid_size = self.scale * self.scale
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]))
        self.action_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]))
        self.goal_padding = goal_padding
        self.spec = EnvSpec(id='PointMass-v0', max_episode_steps=int(max_episode_steps_coeff*self.scale))

    def reset(self):
        plt.close()
        self.state = np.array([self.goal_padding, self.goal_padding])
        # self.state = np.random.randint(1,self.scale+1,size=[2,])
        # state = self.state/self.scale
        self.gob = np.array([1.0, 10.0])
        # self.gob = np.random.randint(1,self.scale+1,size=[2,])
        # self.gob = self.gob*1.0
        # self.gob = np.array([1-self.goal_padding/self.scale,1-self.goal_padding/self.scale])
        a0 = np.array([self.goal_padding, self.goal_padding])
        return self.state, self.gob, a0

    def step(self, action):
        x, y = action

        # next state
        new_x = self.state[0]+x
        new_y = self.state[1]+y
        if new_x < 0:
            new_x = 0
        if new_x > self.scale:
            new_x = self.scale
        if new_y < 0:
            new_y = 0
        if new_y > self.scale:
            new_y = self.scale
        self.state = np.array([new_x, new_y])
        state = self.state/self.scale

        if abs(new_x-self.gob[0]) <= 0 and abs(new_y-self.gob[1]) <= 0:
            reward = 10
        else:
            reg_term = -np.sum(abs(self.state - self.gob) / self.gob)
            reward = reg_term

        # done
        done = False

        return self.state, reward, done, None

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
            gob_array = data['golden_ob']
        ob_array = np.reshape(ob_array, newshape=[-1,2]) / self.scale
        gob_array = np.reshape(gob_array, newshape=[-1,2]) / self.scale

        ob_indices = np.array([int(self.preprocess(s)) for s in ob_array])
        gob_indices = np.array([int(self.preprocess(s)) for s in gob_array])

        images = []
        start = time.time()
        for cnt, i,j in zip(range(len(ob_indices)), ob_indices, gob_indices):
            a = np.zeros(int(self.grid_size))
            a[i] = 1
            a[j] = 0.5
            a = np.reshape(a, (self.scale, self.scale))
            a = scipy.ndimage.zoom(a, 200//self.scale, order=0)
            images.append(a)

        dirname = os.path.dirname(fname)
        imageio.mimsave(os.path.join(dirname, '{}.gif'.format(itr)), images)

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
                self.visualize(i, os.path.join(dirname, s + '/'+ str(iters[i])))
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
    args = parser.parse_args()
    env = PointMass()
    env.create_visualization(args.dirname)




