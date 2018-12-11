import itertools
import os
import pickle
import copy
import random
import gym
import numpy as np
from util import ExperienceBuffer
import logz
import time

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

from rinokeras.rl.env_runners import PGEnvironmentRunner, BatchRollout
from rinokeras.rl.policies import StandardPolicy, LSTMPolicy
from rinokeras.rl.trainers import PolicyGradient, PPO
from rinokeras.train import TrainGraph

from envs.pointmass3d_discrete import goal_distance

def setup_logger(kwargs):
    logdir = os.path.join('data_plot', kwargs['exp_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
    logdir = os.path.join(logdir, str(kwargs['seed']))
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    del kwargs['env_actual_goal']
    del kwargs['env_sub_goals']
    logz.save_params(kwargs)

def store_pickle(object, fname):
    with open(fname, 'wb') as f:
        pickle.dump(object, f)

def train(
        exp_name, # experiment name to be used for logging
        env_name, # the env_name that was used to create an instance of the environment
        env_actual_goal, # this environment never changes the goal and rewards are based on the main task
        env_sub_goals, # in this environment upon reset the goal is sampled from the list of goals provided and rewards are given accordingly
        seed, # random seed for everything
        policy_type, # either 'lstm' or 'standard'
        algorithm, # either 'ppo' or 'vpg'
        init_logstd, # initial log_std for continuous actions
        use_her, # whether to use HER modification or not
        gamma=0.95,
        model_dim=64,
        n_layers=1,
        n_rollout_per_actual_goal=40, # the number of rollouts that env_actual_goal needs to run at the beginning of each iteration
        n_rollouts_per_sub_goals=40, # the number of rollouts that env_sub_goals need to run
        max_ep_steps=50, # max episode length for each rollout
        entcoeff = 1, # the coefficient for entropy penalty in the total loss function
        ex_buffer_size=40, # the size of experience buffer size
        max_iter=500, # the maximum number of iterations that the algorithm is run for
        ):

    setup_logger(locals())
    start = time.time()

    # define some hacky functions
    def updateHer(batch_rollout, env, experience_buffer):
        # this function only works for pointmass 4 not circuits
        # TODO make it general to work with any sparse env
        # if env.__class__.__name__ is not 'PointMass':
        #     raise NotImplementedError

        assert isinstance(batch_rollout, BatchRollout), \
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
                # index = random.randint(0, len(roll)-1)
                index = -1
                best_ob = copy.deepcopy(roll.obs[index])
                # best_score_list.append(np.max(score))
                best_score_list.append(score_fn(state=best_ob[:2], boundary=best_ob[2:6]))
                new_bound = env.get_boundary(best_ob[0], best_ob[1])

            experience_buffer.add(new_bound)

        print(np.max(best_score_list))
        env.multi_goal = True
        env.boundaries = experience_buffer.sample(n_rollout_per_actual_goal)

    def updateHerGoalEnvs(batch_rollout, env, experience_buffer):
        rollouts = copy.deepcopy(batch_rollout.rollouts)
        # env.goals = []
        def score_fn(achieved_goal, desired_goal):
            score = env.env.compute_reward(achieved_goal, desired_goal, info=None)
            return score

        best_score_list = []
        env.goals = []
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
        env.goals+=experience_buffer.sample(n_rollout_per_actual_goal)

    def updateHer_pm3dd(batch_rollout, env, experience_buffer):
        rollouts = copy.deepcopy(batch_rollout.rollouts)
        env.goals = []
        best_score_list = []
        for roll in rollouts:
            if len(roll) <  max_ep_steps:
                new_goal = roll.obs[0][3:6]
                best_score_list.append(0)
            else:
                # index = random.randint(0, len(roll)-1)
                best_ob = copy.deepcopy(roll.obs[-1])
                best_score_list.append(-goal_distance(best_ob[:3], best_ob[3:6]))
                new_goal = best_ob[:3]

            experience_buffer.add(new_goal)

        print(np.max(best_score_list))
        env.multi_goal = True
        env.goals+=experience_buffer.sample(n_rollout_per_actual_goal)


    # initialize random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    if hasattr(env_actual_goal, 'seed'):
        env_actual_goal.seed(seed)
    if hasattr(env_sub_goals, 'seed'):
        env_sub_goals.seed(seed)

    policies = {
        'standard': StandardPolicy,
        'lstm': LSTMPolicy}
    algorithms = {
        'vpg': PolicyGradient,
        'ppo': PPO}

    # Parameters/Hyperparameters
    discrete = not isinstance(env_actual_goal.action_space, gym.spaces.Box)
    action_shape = (env_actual_goal.action_space.n,) if discrete else env_actual_goal.action_space.shape

    n_updates_per_batch = 1 if algorithm == 'vpg' else 3
    embedding_model = Dense(model_dim)

    # Placeholders
    obs_ph = Input((None,) + env_actual_goal.observation_space.shape)
    act_ph = Input((None,) + (() if discrete else env_actual_goal.action_space.shape), dtype=tf.int32 if discrete else tf.float32)
    val_ph = Input((None,))
    seqlen_ph = Input((), dtype=tf.int32)

    # Setup policy, experiment, graph
    policy = policies[policy_type](
        action_shape, 'discrete' if discrete else 'continuous', embedding_model, model_dim,
        initial_logstd=init_logstd, n_layers_logits=n_layers, n_layers_value=n_layers, take_greedy_actions=False)

    experiment = algorithms[algorithm](policy, distribution_strategy=tf.contrib.distribute.OneDeviceStrategy('/cpu:0'),
                                      entcoeff=entcoeff)

    graph = TrainGraph.from_experiment(experiment, (obs_ph, act_ph, val_ph, seqlen_ph))

    runner_actual_goal = PGEnvironmentRunner(env_actual_goal, policy, gamma, max_episode_steps=max_ep_steps)
    runner_sub_goals = PGEnvironmentRunner(env_sub_goals, policy, gamma, max_episode_steps=max_ep_steps)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    all_rewards_for_g_star = []
    all_rewards_for_sub_g = []

    ex_buffer = ExperienceBuffer(bufferSize=ex_buffer_size)

    # Do Training
    for t in itertools.count():

        rollouts_g_star = []
        for _ in range(n_rollout_per_actual_goal):
            rollouts_g_star.append(next(runner_actual_goal))  # type: ignore
        batch_rollout_g_star  = BatchRollout(rollouts_g_star, variable_length=True, keep_as_separate_rollouts=True)

        if use_her:
            # TODO make this part of the code better
            if env_name == 'pm4' or env_name == 'pm4c':
                updateHer(batch_rollout_g_star, env_sub_goals, ex_buffer) # env's mobj flag and boundary list gets updated here
            elif env_name == 'pm3dd':
                updateHer_pm3dd(batch_rollout_g_star, env_sub_goals, ex_buffer) # env's mobj flag and boundary list gets updated here
            else:
                updateHerGoalEnvs(batch_rollout_g_star, env_sub_goals, ex_buffer)

        rollouts_sub_g = []
        for _ in range(n_rollouts_per_sub_goals):
            rollouts_sub_g.append(next(runner_sub_goals))  # type: ignore
        batch_rollout_sub_g = BatchRollout(rollouts_sub_g, variable_length=True, keep_as_separate_rollouts=True)

        mean_episode_reward = np.mean(batch_rollout_g_star.episode_rew)
        all_rewards_for_g_star.append(mean_episode_reward)
        mean_episode_steps = np.mean(batch_rollout_g_star.seqlens)
        current_episode_num = runner_actual_goal.episode_num


        logz.log_tabular("time", time.time()-start)
        logz.log_tabular("iteration", t)
        logz.log_tabular("g*_mean_reward", np.mean(batch_rollout_g_star.episode_rew))
        logz.log_tabular("g*_std_reward", np.std(batch_rollout_g_star.episode_rew))
        logz.log_tabular("g*_min_reward", np.min(batch_rollout_g_star.episode_rew))
        logz.log_tabular("g*_max_reward", np.max(batch_rollout_g_star.episode_rew))
        logz.log_tabular("g*_mean_ep_len", np.mean(batch_rollout_g_star.seqlens))
        logz.log_tabular("g*_std_ep_len", np.std(batch_rollout_g_star.seqlens))
        logz.log_tabular("g*_cur_ep_number", runner_actual_goal.episode_num)

        logz.log_tabular("sub_g_mean_reward", np.mean(batch_rollout_sub_g.episode_rew))
        logz.log_tabular("sub_g_std_reward", np.std(batch_rollout_sub_g.episode_rew))
        logz.log_tabular("sub_g_min_reward", np.min(batch_rollout_sub_g.episode_rew))
        logz.log_tabular("sub_g_max_reward", np.max(batch_rollout_sub_g.episode_rew))
        logz.log_tabular("sub_g_mean_ep_len", np.mean(batch_rollout_sub_g.seqlens))
        logz.log_tabular("sub_g_std_ep_len", np.std(batch_rollout_sub_g.seqlens))
        logz.log_tabular("sub_g_cur_ep_number", runner_sub_goals.episode_num)

        logz.dump_tabular()
        logz.pickle_tf_vars()


        if policy.action_space != 'discrete':
            logstd = graph.run(policy.action_distribution.logstd)
            actions = batch_rollout_sub_g.act
            print('STD_ACTIONS: {}'.format(np.exp(logstd)))

        viz_dir = os.path.join(logz.G.output_dir, 'viz')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        g_star_viz_fname = os.path.join(viz_dir, '{}.dpkl'.format(t))
        sub_g_viz_fname = os.path.join(viz_dir, '{}_tg.dpkl'.format(t))
        store_pickle({'ob': batch_rollout_g_star.obs}, g_star_viz_fname)
        store_pickle({'ob': batch_rollout_sub_g.obs}, sub_g_viz_fname)


        if algorithm == 'ppo':
            experiment.update_old_model()
        for _ in range(n_updates_per_batch):
            loss = graph.run('update', (batch_rollout_sub_g.obs, batch_rollout_sub_g.act, batch_rollout_sub_g.val, batch_rollout_sub_g.seqlens))

        if t > max_iter:
            break
