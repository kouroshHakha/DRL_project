import itertools
import os
import pickle
import copy
import random
import gym
import numpy as np
from util import ExperienceBuffer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

from envs.pointmass3 import PointMass as PointMass_v3
from envs.pointmass4 import PointMass as PointMass_v4
from envs.pointmass5_seq_with_int_rewards import PointMass as PointMass_v5
from envs.ckt_env_discrete import CSAmp as CSAmpDiscrete
from envs.pointmass4_cont import PointMass4Cont
from envs.opamp_discrete import TwoStageAmp

from rinokeras.rl.env_runners import PGEnvironmentRunner, BatchRollout
from rinokeras.rl.policies import StandardPolicy, LSTMPolicy
from rinokeras.rl.trainers import PolicyGradient, PPO
from rinokeras.train import TrainGraph
import IPython
import argparse

#Inputs to RL algorithm
#Example of running file:  
#   python rl_rinokeras.py --expname descriptiveNameforExpHere --env ckt-v2 --policy standard --alg ppo

parser = argparse.ArgumentParser('Rinokeras RL Example Script')
parser.add_argument('--expname', type=str, help='Name of the experiment you are running')
parser.add_argument('--env', type=str, default='opamp', help='Which gym environment to run on')
parser.add_argument('--policy', type=str, choices=['standard', 'lstm'], default='standard',
                    help='Which type of policy to run')
parser.add_argument('--alg', type=str, choices=['vpg', 'ppo'], default='vpg',
                    help='Which algorithm to use to train the agent')
parser.add_argument('--logstd', type=float, default=0, help='initial_logstd') #how to initialize weights
parser.add_argument('--seed', '-s', type=int, default=10) #determines what random seed you run on
parser.add_argument('--mobj', action='store_true', help='multiple objectives')
parser.add_argument('--sparse', action='store_true', help='determines sparsity of the reward') #probably won't use
parser.add_argument('--mobj_gen', action='store_true', help='whether to run difference mobj during validation')

args = parser.parse_args()

# setup the log folder
if not(os.path.exists('data')):
    os.makedirs('data')
logdir = args.expname + 'rinokeras' + '_' + args.env + '_' + args.policy + '_' + args.alg + '_' + str(args.mobj) 
logdir = os.path.join('data', logdir + '/0')
if not(os.path.exists(logdir)):
    os.makedirs(logdir)

# instantiates gym environment  
if args.env == 'pm4':
    env = PointMass_v4(multi_goal=args.mobj, sparse=args.sparse)
    env_validation = PointMass_v4(sparse=args.sparse)
elif args.env == 'ckt-v2':
    env = CSAmpDiscrete(multi_goal=args.mobj, sparse=args.sparse)
    env_validation = CSAmpDiscrete(sparse=args.sparse, generalize_test=args.mobj_gen)
elif args.env == 'opamp':
    env = TwoStageAmp(multi_goal=args.mobj, sparse=args.sparse)
    env_validation = TwoStageAmp(sparse=args.sparse)

# initialize random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

policies = {
    'standard': StandardPolicy,
    'lstm': LSTMPolicy}
algorithms = {
    'vpg': PolicyGradient,
    'ppo': PPO}

# Parameters/Hyperparameters
discrete = not isinstance(env.action_space, gym.spaces.Box)
action_shape = (env.action_space.n,) if discrete else env.action_space.shape
model_dim = 64                                      #Num neurons for each layer
gamma = 0.95                                        #discount factor, dampen agent's choice of action
n_rollouts_per_batch_validation = 10                #Number of rollouts used for validation 
n_rollouts_per_batch_training = 20                  #Number of rollouts used for training
max_ep_steps= 60                                    #Maximum number of steps in each trajectory 
n_updates_per_batch = 1 if args.alg == 'vpg' else 3 #Efficient updates if you use PPO
embedding_model = Dense(model_dim)                  #Can experiment with number of layers by going to rinokers.rl.policies

# Placeholders
obs_ph = Input((None,) + env.observation_space.shape)
act_ph = Input((None,) + (() if discrete else env.action_space.shape), dtype=tf.int32 if discrete else tf.float32)
val_ph = Input((None,))
seqlen_ph = Input((), dtype=tf.int32)

# Setup policy, experiment, graph
policy = policies[args.policy](
    action_shape, 'discrete' if discrete else 'continuous', embedding_model, model_dim,
    initial_logstd=args.logstd, n_layers_logits=1, n_layers_value=1, take_greedy_actions=False)

#Set GPU settings
gpu_fraction = 0.1
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

experiment = algorithms[args.alg](policy, distribution_strategy=tf.contrib.distribute.OneDeviceStrategy('/cpu:0'),
                                  entcoeff=1)
graph = TrainGraph.from_experiment(experiment, (obs_ph, act_ph, val_ph, seqlen_ph))

#Initialize runners of the environment, Roshan's code stuff
runner = PGEnvironmentRunner(env, policy, gamma, max_episode_steps=max_ep_steps)
runner_validation = PGEnvironmentRunner(env_validation, policy, gamma, max_episode_steps=max_ep_steps)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

all_valid_rewards = []
all_rewards = []

# Do Training
for t in itertools.count():

    #Creates rollouts for validation 
    validation_rollouts = []
    for _ in range(n_rollouts_per_batch_validation):
        validation_rollouts.append(next(runner_validation))  # type: ignore
    validation_batch_rollout  = BatchRollout(validation_rollouts, variable_length=True, keep_as_separate_rollouts=True)

    #Rollouts for training
    rollouts = []
    for _ in range(n_rollouts_per_batch_training):
        rollouts.append(next(runner))  # type: ignore
    batch_rollout = BatchRollout(rollouts, variable_length=True, keep_as_separate_rollouts=True)

    #Gets mean reward/episode steps for validation 
    mean_episode_reward = np.mean(validation_batch_rollout.episode_rew)
    all_valid_rewards.append(mean_episode_reward)
    mean_episode_steps = np.mean(validation_batch_rollout.seqlens)
    current_episode_num = runner_validation.episode_num

    printstr = []
    printstr.append('TIME: {:>4}'.format(t))
    printstr.append('EPISODE: {:>7}'.format(current_episode_num))
    printstr.append('MEAN REWARD: {:>6.1f}'.format(mean_episode_reward))
    printstr.append('MEAN EPISODE STEPS: {:>5}'.format(mean_episode_steps))
    print(', '.join(printstr))

    #Gets mean reward/episode steps for training
    mean_episode_reward = np.mean(batch_rollout.episode_rew)
    all_rewards.append(mean_episode_reward)
    mean_episode_steps = np.mean(batch_rollout.seqlens)
    current_episode_num = runner.episode_num

    printstr = []
    printstr.append('TIME: {:>4}'.format(t))
    printstr.append('EPISODE: {:>7}'.format(current_episode_num))
    printstr.append('MEAN REWARD: {:>6.1f}'.format(mean_episode_reward))
    printstr.append('MEAN EPISODE STEPS: {:>5}'.format(mean_episode_steps))
    print(', '.join(printstr))

    #only when action space is continuous
    if policy.action_space != 'discrete':
        logstd = graph.run(policy.action_distribution.logstd)
        actions = batch_rollout.act
        print('STD_ACTIONS: {}'.format(np.exp(logstd)))

    #saves validation observations from rollouts in a pkl file 
    obs_log = {'ob': validation_batch_rollout.obs}
    with open(os.path.join(logdir, '{}.dpkl'.format(t)), 'wb') as f:
        pickle.dump(obs_log, f)

    #saves training observations
    obs_log_temp_goals = {'ob': batch_rollout.obs}
    with open(os.path.join(logdir, 'tg_{}.dpkl'.format(t)), 'wb') as f:
        pickle.dump(obs_log_temp_goals, f)

    #numpy array that saves the validation reward (use for plotting) 
    np.save('-'.join([args.expname, args.env, args.policy, args.alg, 'logstd=' + str(args.logstd) + 
        '-mobj=' + str(args.mobj) + '-sparse=' + str(args.sparse) ]) + '_' + str(n_rollouts_per_batch_validation) 
        + '_' + str(n_rollouts_per_batch_training) + '.npy', np.array(all_valid_rewards))

    #run the agent 
    if args.alg == 'ppo':
        experiment.update_old_model()
    for _ in range(n_updates_per_batch):
        loss = graph.run('update', (batch_rollout.obs, batch_rollout.act, batch_rollout.val, batch_rollout.seqlens))

    #exit training if it's taking too long
    if t > 1000:
        break
