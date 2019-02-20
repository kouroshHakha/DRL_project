import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import framework
import IPython
parser = argparse.ArgumentParser('Plotting Script Rinokeras')
parser.add_argument('--train_roll', type=int, help='number of rollouts during training')
parser.add_argument('--traj_len', type=int, help='trajectory length')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--expname', type=str, help='path to data you want plotted')
parser.add_argument('--num_objs_train', type=int, help='number of trained objectives')
parser.add_argument('--num_objs_test', type=int, help='number of untrained objectives')

args = parser.parse_args()

def plot_training_curve(ax,data):
    if args.normalize:
        data = (data - np.min(data))/ (np.max(data)-np.min(data))
    x_axis = np.array(range(0,len(data)))*args.train_roll*args.traj_len
    ax.set_xlabel('Number of samples', fontsize=8)
    ax.set_ylabel('Avg Episode Reward',fontsize=8)
    ax.set_title('Training Curve RL Agent',fontsize=10)
    ax.plot(x_axis,data)

def plot_trained_objs(ax,trajectory_with_reward_arr):
    #Trajectory array of arrays [[obj1],[obj2],...,[obj4]]
    #Each sub-array contains the rewards per step for each trajectory
    trained_objs = trajectory_with_reward_arr[0:args.num_objs_train,:]
    if len(trained_objs == 1):
        trained_objs = np.reshape(trained_objs,(trained_objs.shape[1],))
        ax.plot(trained_objs)
    else:
        for objs in trained_objs:
            objs = np.reshape(objs,(objs.shape[1],))
            ax.plot(objs)
    ax.set_xlabel('Number of Samples', fontsize=8)
    ax.set_ylabel('Reward per Step',fontsize=8)
    ax.set_title('Reward for Different Trained Objectives',fontsize=10)

def plot_untrained_objs(ax,trajectory_with_reward_arr):
    #Trajectory array of arrays [[obj1],[obj2],...,[obj4]]
    #Each sub-array contains the rewards per step for each trajectory
    untrained_objs = trajectory_with_reward_arr[args.num_objs_train:len(trajectory_with_reward_arr),:]
    if len(untrained_objs) == 1:
        untrained_objs = np.reshape(untrained_objs,(untrained_objs.shape[1],))
        ax.plot(untrained_objs)
    else:
        for objs in untrained_objs:
            ax.plot(objs)
    ax.set_xlabel('Number of Samples',fontsize=8)
    ax.set_ylabel('Reward per Step',fontsize=8)
    ax.set_title('Reward for Different Un-Trained Objectives',fontsize=10)

if __name__ == '__main__':
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.subplots_adjust(hspace=.5)
    framework_path = os.path.abspath(framework.__file__).split("__")
    path_to_data = framework_path[0].replace('framework/','')+args.expname+'/'
    files = glob.glob(path_to_data+'*.npy')
    files.sort()

    plot_training_curve(ax1,np.load(files[0]))
    plot_trained_objs(ax2,np.load(files[2]))
    plot_untrained_objs(ax3,np.load(files[2]))

    plt.show()
