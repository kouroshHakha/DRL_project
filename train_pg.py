import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
import time
from util import *
import logz
import time
import IPython

def pathlength(path):
    return len(path["rew"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(Agent.train)[0]
    params = []
    for k in args:
        if (k in locals_) and (k != 'self'):
            params.append(locals_[k])
    logz.save_params(params)

class Agent(object):
    def __init__(self,
                 env,
                 computation_graph_args,
                 pg_flavor_args,
                 ):

        self.env = env

        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.hist_dim = computation_graph_args['hist_dim']
        self.state_dim = computation_graph_args['state_dim']
        self.mini_batch_size = computation_graph_args['mini_batch_size']
        self.roll_out_h = computation_graph_args['roll_out_h']

        self.learning_rate = computation_graph_args['learning_rate']
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)


        self.min_timesteps_per_batch = 1000
        self.max_path_length = 100

        self.gamma = pg_flavor_args['gamma']
        self.reward_to_go = pg_flavor_args['reward_to_go']
        self.nn_baseline = pg_flavor_args['nn_baseline']
        self.normalize_advantages = pg_flavor_args['normalize_advantages']

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def _define_placeholders(self):

        """

        :return:
        """

        self.sy_ob = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ob_dim])
        self.sy_golden_ob = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ob_dim])
        self.sy_ac = tf.placeholder(dtype=tf.float32, shape=[None, None,  self.ac_dim])
        self.sy_ac_prev = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ac_dim])
        self.sy_adv = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.sy_seq_len = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.sy_init_history = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim*2])
        print('[db] history', self.sy_init_history)


    def _policy_forward_pass(self, sy_ob, sy_ac_prev, sy_golden_ob):
        """

        :param sy_ob: [None, self.ob_dim] -> [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param sy_ac_prev: [None, self.ac_dim] -> [self.mini_batch_size, self.roll_out_h, self.ac_dim]
        :param sy_golden_ob: [None, self.ob_dim] -> [mini_batch, self.roll_out_h, self.ob_dim]
        :return:
            sy_ac_mean: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
            sy_ac_logstd: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        """

        input_shape_tensor = tf.shape(sy_ob)
        num_samples_in_batch = input_shape_tensor[0]
        num_time_steps = input_shape_tensor[1]
        sy_ob_in = tf.reshape(sy_ob, [-1, self.ob_dim])
        sy_ac_prev_in = tf.reshape(sy_ac_prev, [-1, self.ac_dim])
        sy_golden_ob_in = tf.reshape(sy_golden_ob, [-1, self.ob_dim])


        # sy_golden_ob_in = tf.concat([sy_golden_ob for _ in range(self.roll_out_h)], axis=1)

        layer_ob = FCLayer(self.ob_dim, self.state_dim, activation='relu', name='ob_fc')
        layer_ac_prev = FCLayer(self.ac_dim, self.state_dim, activation='relu', name='ac_prev_fc')
        layer_golden_ob = FCLayer(self.ob_dim, self.state_dim, activation='relu', name='golden_ob_fc')

        state_layer_in = tf.concat([layer_ob(sy_ob_in),
                                    layer_ac_prev(sy_ac_prev_in),
                                    layer_golden_ob(sy_golden_ob_in)], axis=1)

        layer_state = FCLayer(3*self.state_dim, self.state_dim, activation='relu', name='state_fc')

        state_lstm_in = tf.reshape(layer_state(state_layer_in), [num_samples_in_batch, num_time_steps, self.state_dim])

        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hist_dim, state_is_tuple=False)
        # print('[db] cell.state_size', type(self.lstm_cell.state_size))
        # print('[db] cell.state_size=', self.lstm_cell.state_size)
        self.lstm_out, self.lstm_history = tf.nn.dynamic_rnn(self.lstm_cell, state_lstm_in,
                                                             sequence_length=self.sy_seq_len,
                                                             initial_state=self.sy_init_history)

        print('[db] next_history', self.lstm_history)

        # self.lstm_core = LSTMCell(self.state_dim, self.hist_dim, minibatch_size=self.mini_batch_size)
        fc_out_in = tf.reshape(self.lstm_out, [-1, self.hist_dim])

        layer_mean = FCLayer(self.hist_dim, self.ac_dim, activation='relu', name='out_fc')
        layer_std = FCLayer(self.hist_dim, self.ac_dim, activation='relu', name='out_fc')

        sy_ac_mean = layer_mean(fc_out_in)
        sy_ac_logstd = layer_std(fc_out_in)

        # sy_ac_mean = tf.reshape(sy_ac_mean, [self.mini_batch_size, self.roll_out_h, self.ac_dim])
        # sy_ac_logstd = tf.reshape(sy_ac_logstd, [self.mini_batch_size, self.roll_out_h, self.ac_dim])

        return sy_ac_mean, sy_ac_logstd

    def _get_log_prob(self, policy_parameters, sy_ac):
        """
        :param policy_parameters: (sy_mean, sy_logstd)
            sy_mean: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
            sy_logstd: [None, self.ob_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        :param
            sy_ac: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        :return:
            sy_logprob: [None, ] -> [self.mini_batch_size * self.roll_out_h]
        """
        sy_mean, sy_logstd = policy_parameters
        mvn = tfp.distributions.MultivariateNormalDiag(loc=sy_mean, scale_diag=tf.exp(sy_logstd))
        sy_ac_in = tf.reshape(sy_ac, [self.mini_batch_size * self.roll_out_h, self.ac_dim])
        sy_logprob = mvn.log_prob(sy_ac_in)
        return sy_logprob

    def _sample_action(self, policy_parameters):

        sy_mean, sy_logstd = policy_parameters

        sy_sampled_ac = tf.add(sy_mean, tf.multiply(tf.exp(sy_logstd), tf.random_normal([self.ac_dim])))
        return sy_sampled_ac

    def build_computation_graph(self):

        self._define_placeholders()

        # The policy takes in observations over all time steps and produces a distribution over the action space
        # at all time steps
        self.policy_parameters = self._policy_forward_pass(self.sy_ob, self.sy_ac_prev, self.sy_golden_ob)
        print_debug("policy params", self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob = self._get_log_prob(self.policy_parameters, self.sy_ac)

        print_debug("log_prob", self.sy_logprob)
        print_debug("sy_adv", self.sy_adv)

        self.loss = -tf.reduce_mean(tf.multiply(self.sy_logprob, self.sy_adv))

        gradients = self.optimizer.compute_gradients(self.loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -1, 1), var)

        self.update_params = self.optimizer.apply_gradients(gradients)


        self.sy_sampled_ac = self._sample_action(self.policy_parameters)

    def sample_trajectories(self):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            path = self.sample_trajectory()
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self):
        ob, golden_obs_org, ac_prev = self.env.reset()
        obs, acs, ac_prevs, rewards, golden_obs = [], [], [], [], []
        steps = 0
        init_history = np.zeros([1, self.hist_dim*2])
        sy_seq_len = np.array([1])

        while True:
            obs.append(ob)
            ac_prevs.append(ac_prev)

            ob = np.reshape(ob, newshape=tuple([1, 1, self.ob_dim]))
            golden_ob = np.reshape(golden_obs_org, newshape=tuple([1, 1 , self.ob_dim]))
            golden_obs.append(golden_obs_org)
            ac_prev = np.reshape(ac_prev, newshape=([1,1,self.ac_dim]))

            feed_dict={
                self.sy_ob: ob,
                self.sy_golden_ob: golden_ob,
                self.sy_ac_prev: ac_prev,
                self.sy_seq_len: sy_seq_len,
                self.sy_init_history: init_history,
            }
            ac, history = self.sess.run([self.sy_sampled_ac, self.lstm_history], feed_dict=feed_dict)
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, info = self.env.step(ac)
            rewards.append(rew)
            ac_prev = ac
            init_history = history
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"obs" : np.array(obs, dtype=np.float32),
                "rew" : np.array(rewards, dtype=np.float32),
                "ac" : np.array(acs, dtype=np.float32),
                "prev_ac" : np.array(ac_prevs, dtype=np.float32),
                "golden_obs" : np.array(golden_obs, dtype=np.float32)}
        return path

    def sum_of_rewards(self, re):
        """
        KEERTANA

        Monte Carlo estimation of the Q function.

        :param re: shape: [self.mini_batch_size, self.roll_out_h, 1]
        :return: q: shape: [self.mini_batch_size, self.roll_out_h, 1]
        """
        q = np.array([])
        re = np.reshape(re, (self.mini_batch_size, self.roll_out_h))
        if self.reward_to_go:
            # if reward_to_go is set the sum of all future rewards should be returned
            #check that its getting correct values to iterate through
            for reward_path in re:
                gamma_array = np.power(self.gamma, range(len(reward_path)))
                temp_arr = []
                for t in range(len(reward_path)):
                    new_rew = reward_path[t:]
                    new_gamma = gamma_array[0:len(new_rew)]
                    temp_arr.append(np.dot(new_rew,new_gamma))
                q = np.append(q, np.array(temp_arr))
        else:
            # if reward_to_go is *not* set the vanilla monte carlo estimate should be used which is sum of rewards
            # for all of that trajectory
            for reward_path in re:   
                gamma_array = np.power(self.gamma, range(len(reward_path)))
                q = np.append(q, np.array([np.dot(reward_path, gamma_array)]*len(reward_path)))
        return q

    def compute_advantage(self, ob, q):
        """
        KEERTANA

        If baseline is set this should return the sum_of_rewards subtracted by the baseline value
        The baseline can be the average reward for simplicity or it could be the value function estimator used in ac
        algorithm, I think average return should give us at least a good starting point.

        :param ob: shape: [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param q: shape: [self.mini_batch_size, self.roll_out_h, 1]
        :return:
            adv: shape: [self.mini_batch_size, self.roll_out_h, 1]
        """
        if self.nn_baseline:
            b_n = np.mean(q)
            adv = q - b_n
        else:
            adv = q.copy()
        return adv

    def estimate_return(self, ob, re):
        """
            KEERTANA
            Estimates the returns over a set of trajectories.

            arguments:
                ob: shape: [self.mini_batch_size, self.roll_out_h, self.ob_dim]
                re: length: [self.mini_batch_size, self.roll_out_h, 1]

            returns:
                q: shape: [self.mini_batch_size, self.roll_out_h, 1]. Each element is representing the q value of
                sample n @ time step t.
                adv: shape: [self.mini_batch_size, self.roll_out_h, 1]. Each element is representing the advantage
                value of sample n @ time step t.
        """
        q = self.sum_of_rewards(re)
        adv = self.compute_advantage(ob, q)

        return q, adv

    def update_parameters(self, ph_ob, ph_golden_ob, ph_ac, ph_ac_prev, q, adv):
        """
        KEERTANA
        Update the parameters of the policy and (possibly) the neural network baseline,
        which is trained to approximate the value function.
        :param ph_ob: shape: [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param ph_golden_ob: shape: [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param ph_ac: shape: [self.mini_batch_size, self.roll_out_h, self.ac_dim]
        :param ph_ac_prev: shape: [self.mini_batch_size, self.roll_out_h, self.ac_dim]
        :param q: shape: [self.mini_batch_size, self.roll_out_h, 1]
        :param adv: shape: [self.mini_batch_size, self.roll_out_h, 1]
        :return:
            Nothing, just performs one step of gradient update on parameters of  policy and (possibly) value function
            estimator.
        """

        #if self.nn_baseline:
            #raise NotImplementedError
        sy_init_history = np.zeros([self.mini_batch_size, self.hist_dim*2], dtype=np.float32)
        sy_seq_len = np.zeros([self.mini_batch_size, ], dtype=np.int32)

        feed_dict = {
            self.sy_ob: ph_ob,
            self.sy_golden_ob: ph_golden_ob,
            self.sy_ac: ph_ac,
            self.sy_ac_prev: ph_ac_prev,
            self.sy_adv: adv,
            self.sy_seq_len: sy_seq_len,
            self.sy_init_history: sy_init_history,
        }
        # l, = self.sess.run([self.loss], feed_dict=feed_dict)
        # print("[Debug_training] l- {}".format(l))
        self.sess.run([self.update_params], feed_dict=feed_dict)
        # l, = self.sess.run([self.loss], feed_dict=feed_dict)
        # print("[Debug_training] l+ {}".format(l))

    def train(self, n_iter, logdir):
        total_timesteps = 0
        start = time.time()

        setup_logger(logdir, locals())

        for itr in range(n_iter):
            print("********** Iteration %i ************"%itr)
            paths, timesteps_this_batch = self.sample_trajectories()
            total_timesteps += timesteps_this_batch

            # sample a minibatch_size of random episode with a number of transitions >= unrollings_num
            random_path_indices = np.random.choice(len(paths), self.mini_batch_size)
            batch_obs, batch_acs, batch_ac_prevs, batch_rewards, batch_golden_obs = [], [], [], [], []
            random_paths = []
            for index in random_path_indices:
                path = paths[index]

                # 0:random_transitions_space is the range from which a random transition
                # can be picked up while having unrollings_num - 1 transitions after it
                random_transitions_space = pathlength(path) - self.roll_out_h
                random_start, = np.random.choice(random_transitions_space, 1)

                obs = path['obs'][random_start:random_start + self.roll_out_h]
                acs = path['ac'][random_start:random_start + self.roll_out_h]
                ac_prevs = path['prev_ac'][random_start:random_start + self.roll_out_h]
                golden_obs = path['golden_obs'][random_start:random_start + self.roll_out_h]
                rews = path['rew'][random_start:random_start + self.roll_out_h]

                batch_obs.append(obs)
                batch_acs.append(acs)
                batch_ac_prevs.append(ac_prevs)
                batch_golden_obs.append(golden_obs)
                batch_rewards.append(rews)
                random_paths.append(path)

            ph_ob = np.array(batch_obs)
            ph_golden_ob = np.array(batch_golden_obs)
            ph_ac = np.array(batch_acs)
            ph_ac_prev = np.array(batch_ac_prevs)
            re = np.array(batch_rewards)

            q, adv = self.estimate_return(ph_ob, re)
            # just checking the shapes to make sure
            print("[q_n] {}".format(q.shape))
            print("[adv_n] {}".format(adv.shape))
            self.update_parameters(ph_ob, ph_golden_ob, ph_ac, ph_ac_prev, q, adv)

            # # Log diagnostics
            # KEERTANA
            returns = [path["rew"].sum() for path in paths]
            ep_lengths = [pathlength(path) for path in paths]
            logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("Iteration", itr)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            logz.dump_tabular()
            logz.pickle_tf_vars()
