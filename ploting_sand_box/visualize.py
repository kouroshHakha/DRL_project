import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

files = glob.glob('*.npy')

for f in files:
    data = np.load(f)
    plt.plot(data)
plt.xlabel('Iteration')
plt.ylabel('Avg Episode Reward')
plt.legend([f.split('.')[0] for f in files])
plt.savefig('pointmass_results.png')
plt.show()
