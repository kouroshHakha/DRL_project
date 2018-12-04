import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser('Rinokeras RL Example Script')
parser.add_argument('--normalize', action='store_true')
args = parser.parse_args()

files = glob.glob('*.npy')

for f in files:
    data = np.load(f)
    if args.normalize:
        data = (data - np.min(data))/ (np.max(data)-np.min(data))
    plt.plot(data)
plt.xlabel('Iteration')
plt.ylabel('Avg Episode Reward')
plt.legend([f.split('.')[0] for f in files])
plt.savefig('pointmass_mobj_effect.png')
plt.show()
