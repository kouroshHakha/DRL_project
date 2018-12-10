import argparse
import itertools
import os
import pickle
import time
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
from envs.goal_env_wrapper import GoalEnvWrapper
from envs.pointmass3d_discrete import PointMass3dd, goal_distance
from envs.pointmass4_cont import PointMass4Cont


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

def updateHer(batch_rollout, env, experience_buffer):
    # this function only works for pointmass 4 not circuits
    # TODO make it general to work with any sparse env
    # if env.__class__.__name__ is not 'PointMass':
    #     raise NotImplementedError

    assert isinstance(batch_rollout, BatchRollout),\
        'updateHer expects BatchRollout, but received: {}'.format(type(batch_rollout))

    rollouts = copy.deepcopy(batch_rollout.rollouts)
    # env.boundaries = []
    # env.multi_goal = True
    def score_fn(state, boundary):
        assert len(state) == 2, 'cur_state does not have the right dimensions, len={} vs. 2'.format(len(state))
        assert len(boundary) == 4, 'boundbox does not have the right dimensions, len={} vs. 4'.format(len(boundary))
        x_goal = (boundary[0]+boundary[1])/2
        y_goal = (boundary[2]+boundary[3])/2
        score = -abs((state[0]-x_goal)/x_goal)-abs((state[1]-y_goal)/y_goal)
        return score

    best_score_list = []
    boundary_list = []
    for roll in rollouts:
        if len(roll) <  max_ep_steps:
            new_bound = roll.obs[0][2:6]
            best_score_list.append(0)
        else:
            # score = [score_fn(state=ob[:2],boundary=ob[2:6]) for ob in roll.obs]
            # best_ob = copy.deepcopy(roll.obs[np.argmax(score)])
            index = random.randint(0, len(roll)-1)
            best_ob = copy.deepcopy(roll.obs[index])
            # best_score_list.append(np.max(score))
            best_score_list.append(score_fn(state=best_ob[:2], boundary=best_ob[2:6]))
            new_bound = env.get_boundary(best_ob[0], best_ob[1])

        experience_buffer.add(new_bound)
        # boundary_list.append(new_bound)
        # env.boundaries.append(new_bound)
    # sorted_indices = sorted([i for i in range(len(best_score_list))], key=lambda x:best_score_list[x], reverse=True)
    # sorted_boundaries = [boundary_list[i] for i in sorted_indices]
    # env.boundaries.append(boundary_list[np.argmax(best_score_list)])
    # env.boundaries+=sorted_boundaries[:10]
    # env.boundaries+=boundary_list
    print(np.max(best_score_list))
    # print([best_score_list[i] for i in sorted_indices][:10])
    env.multi_goal = True
    env.boundaries = experience_buffer.sample(n_rollouts_per_batch_validation)

def updateHerGoalEnvs(batch_rollout, env, experience_buffer):
    rollouts = copy.deepcopy(batch_rollout.rollouts)
    # env.goals = []
    def score_fn(achieved_goal, desired_goal):
        score = env.env.compute_reward(achieved_goal, desired_goal, info=None)
        return score

    best_score_list = []
    goal_list = []
    for roll in rollouts:
        if len(roll) <  max_ep_steps:
            new_goal = roll.obs[0][-3:]
            best_score_list.append(0)
        else:
            # index = random.randint(0, len(roll)-1)
            best_ob = copy.deepcopy(roll.obs[-1])
            best_score_list.append(score_fn(achieved_goal=best_ob[:3],desired_goal=best_ob[-3:]))
            new_goal = best_ob[:3]
        experience_buffer.add(new_goal)

    print(np.max(best_score_list))
    env.multi_goal = True
    env.goals+=experience_buffer.sample(n_rollouts_per_batch_validation)

def updateHer_pm3dd(batch_rollout, env):
    rollouts = copy.deepcopy(batch_rollout.rollouts)
    env.goals = []
    env.multi_goal = True
    best_score_list = []
    goal_list = []
    for roll in rollouts:
        if len(roll) <  max_ep_steps:
            new_goal = roll.obs[0][3:6]
            best_score_list.append(0)
        else:
            # index = random.randint(0, len(roll)-1)
            best_ob = copy.deepcopy(roll.obs[-1])
            best_score_list.append(-goal_distance(best_ob[:3], best_ob[3:6]))
            new_goal = best_ob[:3]
        goal_list.append(new_goal)

    env.goals+=goal_list
    print(np.max(best_score_list))

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
elif args.env == 'pm3dd':
    env = PointMass3dd(multi_goal=args.mobj, sparse=args.sparse)
    env_validation = PointMass3dd(sparse=args.sparse)
elif args.env == 'pm4c':
    env = PointMass4Cont(multi_goal=args.mobj, sparse=args.sparse)
    env_validation = PointMass4Cont(sparse=args.sparse)
else:
    # env = gym.make(args.env)
    env_validation = GoalEnvWrapper(args.env, seed=args.seed, sparse=args.sparse)
    env = GoalEnvWrapper(args.env, seed=args.seed, mobj = args.mobj, sparse=args.sparse)
# initialize random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
# env.seed(args.seed)

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
n_rollouts_per_batch_validation = 40
n_rollouts_per_batch_sub_goals = 40
max_ep_steps=  70
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

runner = PGEnvironmentRunner(env, policy, gamma, max_episode_steps=max_ep_steps)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

all_valid_rewards = []
all_rewards = []

runner_validation = PGEnvironmentRunner(env_validation, policy, gamma, max_episode_steps=max_ep_steps)
ex_buffer = ExperienceBuffer(bufferSize=40)
# Do Training
for t in itertools.count():
    validation_rollouts = []
    for _ in range(n_rollouts_per_batch_validation):
        validation_rollouts.append(next(runner_validation))  # type: ignore
    validation_batch_rollout  = BatchRollout(validation_rollouts, variable_length=True, keep_as_separate_rollouts=True)

    if args.her:
        # TODO make this part of the code better
        if args.env == 'pm4' or args.env == 'pm4c':
            updateHer(validation_batch_rollout, env, ex_buffer) # env's mobj flag and boundary list gets updated here
        elif args.env == 'pm3dd':
            updateHer_pm3dd(validation_batch_rollout, env) # env's mobj flag and boundary list gets updated here
        else:
            updateHerGoalEnvs(validation_batch_rollout, env, ex_buffer)



    rollouts = []
    for _ in range(n_rollouts_per_batch_sub_goals):
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

    if policy.action_space != 'discrete':
        logstd = graph.run(policy.action_distribution.logstd)
        actions = batch_rollout.act
        print('STD_ACTIONS: {}'.format(np.exp(logstd)))

    obs_log = {'ob': validation_batch_rollout.obs}
    with open(os.path.join(logdir, '{}.dpkl'.format(t)), 'wb') as f:
        pickle.dump(obs_log, f)

    obs_log_temp_goals = {'ob': batch_rollout.obs}
    with open(os.path.join(logdir, 'tg_{}.dpkl'.format(t)), 'wb') as f:
        pickle.dump(obs_log_temp_goals, f)

    np.save('-'.join([args.env, args.policy, args.alg, 'logstd=' + str(args.logstd) + '-mobj=' + str(args.mobj) +
                      '-sparse=' + str(args.sparse) + '-her=' + str(args.her)]) +
            '_' + str(n_rollouts_per_batch_validation) + '_' + str(n_rollouts_per_batch_sub_goals) +
            '.npy', np.array(all_valid_rewards))

    # if args.her:
    #     batch_rollout.extend(validation_batch_rollout)

    if args.alg == 'ppo':
        experiment.update_old_model()
    for _ in range(n_updates_per_batch):
        loss = graph.run('update', (batch_rollout.obs, batch_rollout.act, batch_rollout.val, batch_rollout.seqlens))

    if t > 1000:
        break
