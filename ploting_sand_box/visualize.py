import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import framework
import IPython
import pickle
import yaml
from envs.opamp_discrete import  TwoStageAmp 
from envs.opamp_full_discrete import TwoStageAmp as TwoStageAmpFull
parser = argparse.ArgumentParser('Plotting Script Rinokeras')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--expname', type=str, help='path to data you want plotted')
parser.add_argument('--num_objs_train', type=int, default=1, help='number of trained objectives')
parser.add_argument('--num_objs_test', type=int, default=52, help='number of untrained objectives')
parser.add_argument('--env', type=str, default='TwoStageAmp')
parser.add_argument('--ray', action='store_true') #use ray to plot
args = parser.parse_args()

def unlookup(norm_spec, goal_spec):
    spec = np.multiply(norm_spec, goal_spec) + goal_spec
    return spec

def get_specs(obs):
    #read yaml file
    #extract all the observations usong num param info
    if args.env == 'TwoStageAmp':
        env = TwoStageAmp()
    elif args.env == 'TwoStageFull':
        env = TwoStageAmpFull()
    global_spec = env.global_g
    spec_num = len(env.specs)
    obs = obs['ob']
    
    norm_des_specs = []
    for ob in obs:
        norm_des_spec = ob[0][spec_num:spec_num+spec_num]     
        norm_des_specs.append(norm_des_spec)
    
    #unnormalized specs
    unnorm_des_specs = unlookup(norm_des_specs,global_spec) 
    return unnorm_des_specs,spec_num

def plot_training_curve(ax,data):
    data_y = data[:,1]
    if args.normalize:
        data_y = (data_y - np.min(data_y))/ (np.max(data_y)-np.min(data_y))
    ax.set_xlabel('Number of samples', fontsize=8)
    ax.set_ylabel('Avg Episode Reward',fontsize=8)
    ax.set_title('Training Curve RL Agent',fontsize=10)
    ax.plot(data[:,0],data_y)

def plot_trained_objs(ax,trajectory_with_reward_arr):
    #Trajectory array of arrays [[obj1],[obj2],...,[obj4]]
    #Each sub-array contains the rewards per step for each trajectory
    trained_objs = trajectory_with_reward_arr[0:args.num_objs_train]
    if len(trained_objs) == 1:
        #trained_objs = np.reshape(trained_objs,(trained_objs.shape[1],))
        ax.plot(trained_objs[0])
    else:
        for objs in trained_objs:
            objs = np.reshape(objs,(objs.shape[1],))
            ax.plot(objs)
    ax.set_xlabel('Number of Samples', fontsize=8)
    ax.set_ylabel('Reward per Step',fontsize=8)
    ax.set_title('Reward for Different Trained Objectives',fontsize=10)

def plot_untrained_objs(ax,trajectory_with_reward_arr,unnorm_des_specs):
    #Trajectory array of arrays [[obj1],[obj2],...,[obj4]]
    #Each sub-array contains the rewards per step for each trajectory
    untrained_objs = trajectory_with_reward_arr[args.num_objs_train:len(trajectory_with_reward_arr)]
    if len(untrained_objs) == 1:
        untrained_objs = np.reshape(untrained_objs,(untrained_objs.shape[1],))
        ax.plot(untrained_objs)
    else:
        for objs in untrained_objs:
            ax.plot(objs)
    ax.set_xlabel('Number of Samples',fontsize=8)
    ax.set_ylabel('Reward per Step',fontsize=8)
    ax.set_title('Reward for Different Un-Trained Objectives',fontsize=10)
    #ax.legend(labels=list(unnorm_des_specs))

def prepare_trajectory(arr):
    non_zero_trajs = [] 

    for traj in arr:
        end_traj = np.argmax(traj>10.0)
        if end_traj != 0:
            new_traj = traj[0:end_traj+1]
        else:
            new_traj = traj
        non_zero_trajs.append(new_traj)
    return np.array(non_zero_trajs)

def print_char(untrained_obs, trajs, unnorm_des_specs, spec_num):
    #did or didn't meet spec
    unmet = []
    met = []
    obs = untrained_obs['ob']

    for i,traj in enumerate(trajs):
        if len(traj) < 60:
            met.append(list(unnorm_des_specs[i]))
        else:
            closest_spec = obs[i][len(obs[i])-1][0:spec_num]
            unnorm_spec = unlookup(closest_spec, unnorm_des_specs[i])
            unmet.append((list(unnorm_des_specs[i]),list(unnorm_spec))) 
    IPython.embed()
 
def plot_untrained_ray_obs(plt,valid_rollouts):
    #Trajectory array of arrays [[obj1],[obj2],...,[obj4]]
    #Each sub-array contains the rewards per step for each trajectory
    for objs in valid_rollouts:
        plt.plot(objs)
    plt.xlabel('Number of Samples',fontsize=8)
    plt.ylabel('Reward per Step',fontsize=8)
    plt.title('Reward for Different Un-Trained Objectives',fontsize=10)
    #plt.legend(labels=list(unnorm_des_specs))

if __name__ == '__main__':
    plt.close()
        
    if args.ray == True:
        with open('rollouts.pkl','rb') as f:
            valid_rollouts = pickle.load(f)
        plot_untrained_ray_obs(plt,valid_rollouts)
    else:
        fig, (ax1,ax2,ax3) = plt.subplots(3)
        fig.subplots_adjust(hspace=.5)
        framework_path = os.path.abspath(framework.__file__).split("__")
        project_path = framework_path[0].replace('framework/','')
        files = glob.glob(project_path+args.expname+'/'+'*.npy')
        files.sort()

        #Info to get unnormalized specs 
        dirs = os.listdir(project_path+'data/')
        for dir in dirs:
            if (args.expname in dir):
               obs_path = dir
        dpkl_dirs = os.listdir(project_path+'data/'+obs_path+'/0/')
        valid_dirs = [valid for valid in dpkl_dirs if not('tg' in valid)]
        newest_pkl = 0
        for pkl_file in valid_dirs:
            split = int(pkl_file.split('.')[0])
            if split > newest_pkl:
                newest_pkl = split
        with open(project_path+'data/'+obs_path+'/0/'+str(newest_pkl)+'.dpkl','rb') as f:
            untrained_obs = pickle.load(f)

        trajs = prepare_trajectory(np.load(files[2]))
        unnorm_des_specs, spec_num = get_specs(untrained_obs)
        plot_training_curve(ax1,np.load(files[0]))
        plot_trained_objs(ax2,trajs)
        plot_untrained_objs(ax3,trajs,unnorm_des_specs)
        print_char(untrained_obs, trajs, unnorm_des_specs, spec_num) 
    plt.show()
