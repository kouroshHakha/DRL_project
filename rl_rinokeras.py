import argparse
import itertools
import os
import pickle
import time
import copy

import gym
import numpy as np
import tensorflow as tf
from envs.pointmass3 import PointMass as PointMass_v3
from envs.pointmass4 import PointMass as PointMass_v4
from envs.pointmass5_seq_with_int_rewards import PointMass as PointMass_v5
from envs.ckt_env_discrete import CSAmp as CSAmpDiscrete
from tensorflow.keras.layers import Dense, Input

from rinokeras.rl.env_runners import PGEnvironmentRunner, BatchRollout
from rinokeras.rl.policies import StandardPolicy, LSTMPolicy
from rinokeras.rl.trainers import PolicyGradient, PPO
from rinokeras.train import TrainGraph
import IPython

# import seaborn as sns
def update_roll(obs, rew, rew_in, new_goal_norm, env):
    new_goal = env.unlookup(new_goal_norm, env.specs_ideal)
    for i in range(len(obs)):
        spec = env.unlookup(obs[i][:3], env.specs_ideal)
        obs[i][:3] = env.lookup(spec, new_goal)
        obs[i][5] = env.reward(spec, new_goal)
        rew[i] = obs[i][5]
        obs[i][6:9] = new_goal_norm
        if i > 0:
            rew_in[i] = rew[i-1]
    return obs, rew, rew_in

def updateHer(batch_rollout, env):
    # this function only works for pointmass 4 not circuits
    # TODO make it general to work with any sparse env
    if env.__class__.__name__ is not 'PointMass':
        raise NotImplementedError

    assert isinstance(batch_rollout, BatchRollout),\
        'updateHer expects BatchRollout, but received: {}'.format(type(batch_rollout))

    rollouts = copy.deepcopy(batch_rollout.rollouts)
    env.boundaries = []
    env.mobj = True
    def score_fn(state, boundary):
        assert len(state) == 2, 'cur_state does not have the right dimensions, len={} vs. 2'.format(len(state))
        assert len(boundary) == 4, 'boundbox does not have the right dimensions, len={} vs. 4'.format(len(boundary))
        x_goal = (boundary[0]+boundary[1])/2
        y_goal = (boundary[2]+boundary[3])/2
        score = -abs((state[0]-x_goal)/x_goal)-abs((state[1]-y_goal)/y_goal)
        return score

    best_score_list = []
    for roll in rollouts:
        score = [score_fn(state=ob[:2],boundary=ob[2:6]) for ob in roll.obs]
        best_ob = copy.deepcopy(roll.obs[np.argmax(score)])
        best_score_list.append(np.max(score))
        new_bound = env.get_boundary(best_ob[0], best_ob[1])
        env.boundaries.append(new_bound)

    IPython.embed()

parser = argparse.ArgumentParser('Rinokeras RL Example Script')
parser.add_argument('--env', type=str, default='pm', help='Which gym environment to run on')
parser.add_argument('--policy', type=str, choices=['standard', 'lstm'], default='standard',
                    help='Which type of policy to run')
parser.add_argument('--alg', type=str, choices=['vpg', 'ppo'], default='vpg',
                    help='Which algorithm to use to train the agent')
parser.add_argument('--logstd', type=float, default=0, help='initial_logstd')
parser.add_argument('--seed', '-s', type=int, default=10)
parser.add_argument('--mobj', action='store_true', help='multiple objectives')
parser.add_argument('--sparse', action='store_true', help='determines sparsity of the reward')
parser.add_argument('--her', action='store_true', help='Added Hindsight Experience Buffer')

args = parser.parse_args()

# setup the log folder
if not(os.path.exists('data')):
    os.makedirs('data')
logdir = 'rinokeras' + '_' + args.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join('data', logdir + '/0')
if not(os.path.exists(logdir)):
    os.makedirs(logdir)


if args.env == 'pm3':
    env = PointMass_v3(multi_goal=args.mobj)
elif args.env == 'pm4':
    env = PointMass_v4(multi_goal=args.mobj, sparse=args.sparse)
    env_validation = PointMass_v4(sparse=args.sparse)
elif args.env == 'pm5':
    env = PointMass_v5(multi_goal=args.mobj)
elif args.env == 'ckt-v2':
    env = CSAmpDiscrete(sparse=args.sparse)
else:
    env = gym.make(args.env)

# initialize random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
env.seed(args.seed)

policies = {
    'standard': StandardPolicy,
    'lstm': LSTMPolicy}
algorithms = {
    'vpg': PolicyGradient,
    'ppo': PPO}

# Parameters/Hyperparameters
discrete = not isinstance(env.action_space, gym.spaces.Box)
action_shape = (env.action_space.n,) if discrete else env.action_space.shape
model_dim = 64
gamma = 0.95
n_rollouts_per_batch = 40
n_updates_per_batch = 1 if args.alg == 'vpg' else 3
embedding_model = Dense(model_dim)

# Placeholders
obs_ph = Input((None,) + env.observation_space.shape)
act_ph = Input((None,) + (() if discrete else env.action_space.shape), dtype=tf.int32 if discrete else tf.float32)
val_ph = Input((None,))
seqlen_ph = Input((), dtype=tf.int32)

# Setup policy, experiment, graph
policy = policies[args.policy](
    action_shape, 'discrete' if discrete else 'continuous', embedding_model, model_dim,
    initial_logstd=args.logstd, n_layers_logits=1, n_layers_value=1, take_greedy_actions=False)

experiment = algorithms[args.alg](policy, distribution_strategy=tf.contrib.distribute.OneDeviceStrategy('/cpu:0'),
                                  entcoeff=1)
graph = TrainGraph.from_experiment(experiment, (obs_ph, act_ph, val_ph, seqlen_ph))

runner = PGEnvironmentRunner(env, policy, gamma, max_episode_steps=50)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

all_valid_rewards = []
all_rewards = []

runner_validation = PGEnvironmentRunner(env_validation, policy, gamma, max_episode_steps=50)
# Do Training
for t in itertools.count():
    validation_rollouts = []
    for _ in range(n_rollouts_per_batch):
        validation_rollouts.append(next(runner_validation))  # type: ignore
    validation_batch_rollout  = BatchRollout(validation_rollouts, variable_length=True, keep_as_separate_rollouts=True)


    if args.her:
        updateHer(validation_batch_rollout, env) # env's mobj flag and boundary list gets updated here

    rollouts = []
    for _ in range(n_rollouts_per_batch):
        rollouts.append(next(runner))  # type: ignore
    batch_rollout = BatchRollout(rollouts, variable_length=True, keep_as_separate_rollouts=True)

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

    obs_log = {'ob': validation_batch_rollout.obs}
    with open(os.path.join(logdir, '{}.dpkl'.format(t)), 'wb') as f:
        pickle.dump(obs_log, f)

    np.save('-'.join([args.env, args.policy, args.alg, 'logstd=' + str(args.logstd) + '-mobj=' + str(args.mobj) +
                      '-sparse=' + str(args.sparse) + '-her=' + str(args.her)]) + '.npy', np.array(all_valid_rewards))

    if args.her:
        batch_rollout.extend(validation_batch_rollout)

    if args.alg == 'ppo':
        experiment.update_old_model()
    for _ in range(n_updates_per_batch):
        loss = graph.run('update', (batch_rollout.obs, batch_rollout.act, batch_rollout.val, batch_rollout.seqlens))

    if t > 1500:
        break
