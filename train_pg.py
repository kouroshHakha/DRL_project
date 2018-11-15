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
    logz.configure_output_dir(logdir)
    args = inspect.getargspec(Agent.__init__)[0]
    # params = []
    # for k in args:
    #     if (k in locals_) and (k != 'self'):
    #         params.append(locals_[k])
    params = dict(
        exp_name=locals_['self'].env.__class__.__name__
    )
    logz.save_params(params)

class Agent(object):
    def __init__(self,
                 env,
                 animate,
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
        self.max_path_length = 10
        self.num_updates_per_iter = 1

        self.gamma = pg_flavor_args['gamma']
        self.reward_to_go = pg_flavor_args['reward_to_go']
        self.nn_baseline = pg_flavor_args['nn_baseline']
        self.normalize_advantages = pg_flavor_args['normalize_advantages']
        self.animate = animate

        np.random.seed(20)
        tf.set_random_seed(20)

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def _define_placeholders(self):

        self.sy_ob = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ob_dim])
        self.sy_golden_ob = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ob_dim])
        self.sy_ac = tf.placeholder(dtype=tf.float32, shape=[None, None,  self.ac_dim])
        self.sy_ac_prev = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ac_dim])
        self.sy_adv = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.sy_seq_len = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.sy_init_history = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim*2])
        print_debug('sy_init_history', self.sy_init_history)


    def _policy_forward_pass(self, sy_ob, sy_ac_prev, sy_golden_ob):
        """

        :param sy_ob: [None, self.ob_dim] -> [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param sy_ac_prev: [None, self.ac_dim] -> [self.mini_batch_size, self.roll_out_h, self.ac_dim]
        :param sy_golden_ob: [None, self.ob_dim] -> [mini_batch, self.roll_out_h, self.ob_dim]
        :return:
            sy_ac_mean: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
            sy_ac_logstd: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        """

        # input_shape_tensor = tf.shape(sy_ob)
        # num_samples_in_batch = input_shape_tensor[0]
        # num_time_steps = input_shape_tensor[1]
        # sy_ob_in = tf.reshape(sy_ob, [-1, self.ob_dim])
        # sy_ac_prev_in = tf.reshape(sy_ac_prev, [-1, self.ac_dim])
        # sy_golden_ob_in = tf.reshape(sy_golden_ob, [-1, self.ob_dim])
        self.sy_meta_state = tf.concat([sy_ob,
                                       sy_golden_ob,
                                       sy_ac_prev], axis=-1)
        # sy_ob_out = tf.layers.dense(sy_ob_in, self.state_dim, tf.nn.relu, name='ob_fc')
        # sy_ac_prev_out = tf.layers.dense(sy_ac_prev_in, self.state_dim, tf.nn.relu, name='ac_prev_fc')
        # sy_golden_ob_out = tf.layers.dense(sy_golden_ob_in, self.state_dim, tf.nn.relu, name='golden_ob_fc')
        self.sy_meta_lstm_in = build_mlp(self.sy_meta_state, self.state_dim, scope='input', n_layers=2, hidden_dim=20, output_activation=tf.nn.relu)
        # print_debug('sy_meta_lstm_in', self.sy_meta_lstm_in)

        # self.sy_state_layer_in = tf.concat([sy_ob_out,
        #                                     sy_ac_prev_out,
        #                                     sy_golden_ob_out], axis=1)
        #
        # self.sy_state_layer_out = tf.layers.dense(self.sy_state_layer_in, self.state_dim, tf.nn.relu, name='state_fc')
        # self.state_lstm_in = tf.reshape(self.sy_state_layer_out, [num_samples_in_batch, num_time_steps, self.state_dim])

        #Create LSTM cells of length batch_size
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim, state_is_tuple=False, name='lstm')
        # self.lstm_out = list([])
        #
        # state = self.sy_init_history
        # for i in range(self.roll_out_h):
        #     # out, state = self.lstm_cell(self.state_lstm_in[:,i,:], state)
        #     out, state = self.lstm_cell(self.sy_meta_lstm_in[:,i,:], state)
        #     if i == 0:
        #         self.state_1 = state
        #     self.lstm_out.append(out)
        # #outputs batch size,  time step, ob_dim
        # self.lstm_out = tf.transpose(tf.stack(self.lstm_out), perm=[1,0,2])
        # print_debug('lstm_out', self.lstm_out)

        self.lstm_out, self.state_1 = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                                        inputs=self.sy_meta_lstm_in,
                                                        initial_state=self.sy_init_history,
                                                        dtype=tf.float32)

        # self.fc_out_in = tf.reshape(self.lstm_out, [-1, self.hist_dim])
        # sy_ac_mean = tf.layers.dense(self.fc_out_in, self.ac_dim, activation=None, name='out_mean_fc')
        # sy_ac_logstd = tf.layers.dense(self.fc_out_in, self.ac_dim, activation=None, name='out_std_fc')
        # sy_policy_params = tf.layers.dense(self.fc_out_in, 2*self.ac_dim, activation=None, name='out_policy_fc')
        sy_policy_params = tf.layers.dense(self.lstm_out, 2*self.ac_dim, activation=None, name='out_policy_fc')
        # sy_policy_params = build_mlp(self.lstm_out, self.ac_dim*2, scope='output', n_layers=2, hidden_dim=128)
        sy_policy_params = tf.reshape(sy_policy_params, [-1, self.ac_dim*2])
        # print('sy_policy_params:{}'.format(sy_policy_params))
        sy_ac_mean = sy_policy_params[:,:self.ac_dim]
        sy_ac_logstd = sy_policy_params[:,self.ac_dim:]

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
        sy_ac_in = tf.reshape(sy_ac, [tf.shape(sy_ac)[0] * tf.shape(sy_ac)[1], self.ac_dim])
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

        mean, log_std = self.policy_parameters
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients_mean = tf.gradients(mean,self.variables)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob = self._get_log_prob(self.policy_parameters, self.sy_ac)

        print_debug("log_prob", self.sy_logprob)
        print_debug("sy_adv", self.sy_adv)

        self.loss = -tf.reduce_mean(tf.multiply(self.sy_logprob, self.sy_adv))

        self.gradients = self.optimizer.compute_gradients(self.loss)
        # for i, (grad, var) in enumerate(self.gradients):
        #     if grad is not None:
        #         self.gradients[i] = (tf.clip_by_value(grad, -1, 1), var)
        #
        self.update_params = self.optimizer.apply_gradients(self.gradients)
        # self.update_params = self.optimizer.minimize(self.loss)


        self.sy_sampled_ac = self._sample_action(self.policy_parameters)

    def sample_trajectories(self, animate_this_episode):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            path = self.sample_trajectory(animate_this_episode)
            # print('trajectory sampled')
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, animate_this_episode=False):
        ob, golden_obs_org, ac_prev = self.env.reset()
        obs, next_obs, acs, ac_prevs, rewards, golden_obs = [], [], [], [], [], []
        steps = 0
        init_history = np.zeros([1, self.hist_dim*2])
        sy_seq_len = np.array([1])

        zero_pad_obs = np.zeros([1, self.roll_out_h-1, self.ob_dim])
        zero_pad_ac_prev = np.zeros([1, self.roll_out_h-1, self.ac_dim])

        while True:
            if animate_this_episode:
                self.env.render()
                time.sleep(0.01)

            obs.append(ob)
            ac_prevs.append(ac_prev)

            ob = np.reshape(ob, newshape=tuple([1, 1, self.ob_dim]))
            golden_ob = np.reshape(golden_obs_org, newshape=tuple([1, 1, self.ob_dim]))
            golden_obs.append(golden_obs_org)
            ac_prev = np.reshape(ac_prev, newshape=([1,1,self.ac_dim]))

            feed_dict={
                self.sy_ob: ob,#np.append(ob, zero_pad_obs, axis=1),
                self.sy_golden_ob: golden_ob, #np.append(golden_ob, zero_pad_obs, axis=1),
                self.sy_ac_prev: ac_prev,#np.append(ac_prev, zero_pad_ac_prev, axis=1),
                #self.sy_seq_len: sy_seq_len,
                self.sy_init_history: init_history,
            }

            #feed_dict[self.state_lstm_in]= np.ones([1, 1, self.state_dim])

            ac, history = self.sess.run([self.sy_sampled_ac, self.state_1], feed_dict=feed_dict)
            ac = ac[0]
            if steps % 10 == 0 and animate_this_episode:
                mean, std= self.sess.run(self.policy_parameters, feed_dict=feed_dict)
                print('Mean: {}'.format(mean[0]))
                print('std: {}'.format(np.exp(std[0])))
                print('ac: {}'.format(ac))
                # print('history: {}'.format(history[0]))

                
            acs.append(ac)
            next_ob, rew, done, info = self.env.step(ac)
            rewards.append(rew)
            next_obs.append(next_ob)

            ac_prev = ac
            init_history = history
            ob = next_ob
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"obs" : np.array(obs, dtype=np.float32),
                "next_obs": np.array(next_obs, dtype=np.float32),
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
        # re = np.reshape(re, (self.mini_batch_size, self.roll_out_h))
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
        sy_init_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
        # sy_seq_len = np.zeros([self.mini_batch_size, ], dtype=np.int32)

        feed_dict = {
            self.sy_ob: ph_ob,
            self.sy_golden_ob: ph_golden_ob,
            self.sy_ac: ph_ac,
            self.sy_ac_prev: ph_ac_prev,
            self.sy_adv: adv,
            # self.sy_seq_len: sy_seq_len,
            self.sy_init_history: sy_init_history,
        }

        l, = self.sess.run([self.loss], feed_dict=feed_dict)

        #test_tnsr_value = self.sess.run(self.fc_out_in, feed_dict=feed_dict)
        #print_debug("test_tnsr_value", test_tnsr_value)

        #for grad, var in self.gradients:
        #    if grad is not None:
        #        grad_value = self.sess.run(grad, feed_dict=feed_dict)
        #        print("var: {}".format(var.name))
        #        print("-"*30)
        #        print("{}".format(grad_value))
        #    else:
        #        print("var: {}".format(var.name))
        #        print("-"*30)
        #        print("grad is None")

        print("[Debug_training] l- {}".format(l))
        self.sess.run([self.update_params], feed_dict=feed_dict)
        l, = self.sess.run([self.loss], feed_dict=feed_dict)
        print("[Debug_training] l+ {}".format(l))

    def pad(self, path):
        rem_obs_array = np.array([path['next_obs'][-1] for _ in range(self.roll_out_h - pathlength(path))])
        rem_acs_array = np.array([path['ac'][-1] for _ in range(self.roll_out_h - pathlength(path))])
        rem_ac_prevs_array = np.array([path['ac'][-1] for _ in range(self.roll_out_h - pathlength(path))])
        rem_golden_obs_array = np.array([path['golden_obs'][-1] for _ in range(self.roll_out_h - pathlength(path))])
        rem_rew_array = np.array([path['rew'][-1] for _ in range(self.roll_out_h - pathlength(path))])

        obs = np.concatenate((path['obs'], rem_obs_array), axis=0)
        acs = np.concatenate((path['ac'], rem_acs_array), axis=0)
        ac_prevs = np.concatenate((path['prev_ac'], rem_ac_prevs_array), axis=0)
        golden_obs = np.concatenate((path['golden_obs'], rem_golden_obs_array), axis=0)
        rews = np.concatenate((path['rew'], rem_rew_array), axis=0)
        print("padding happened -> path_len = {}".format(pathlength(path)))
        return obs, acs, ac_prevs, golden_obs, rews

    def train(self, n_iter, logdir):
        total_timesteps = 0
        start = time.time()
        # step_count = 0

        setup_logger(logdir, locals())
        dirname = logz.G.output_dir

        for itr in range(n_iter):
            print("********** Iteration %i ************"%itr)
            print("// Sampling Trajectories ...")
            animate = self.animate and (itr % 80 == 0) and (itr > 0)
            paths, timesteps_this_batch = self.sample_trajectories(animate)
            total_timesteps += timesteps_this_batch

            # IPython.embed()
            obs_itr = np.empty(shape=[0,paths[0]['obs'].shape[1]], dtype=np.float32)

            for _ in range(self.num_updates_per_iter):
                # sample a minibatch_size of random episode with a number of transitions >= unrollings_num
                # random_path_indices = np.random.choice(len(paths), self.mini_batch_size, replace=False)
                # batch_obs, batch_acs, batch_ac_prevs, batch_rewards, batch_golden_obs = [], [], [], [], []
                # random_paths = []
                # for index in random_path_indices:
                # #     path = paths[index]
                # #
                # #     # 0:random_transitions_space is the range from which a random transition
                # #     # can be picked up while having unrollings_num - 1 transitions after it
                # #     random_transitions_space = pathlength(path) - self.roll_out_h
                # #
                # #     if random_transitions_space < 0:
                # #         obs, acs, ac_prevs, golden_obs, rews = self.pad(path)
                # #     else:
                # #         if random_transitions_space != 0:
                # #             random_start, = np.random.choice(random_transitions_space, 1)
                # #         else:
                # #             random_start = 0
                # #         obs = path['obs'][random_start:random_start + self.roll_out_h]
                # #         acs = path['ac'][random_start:random_start + self.roll_out_h]
                # #         ac_prevs = path['prev_ac'][random_start:random_start + self.roll_out_h]
                # #         golden_obs = path['golden_obs'][random_start:random_start + self.roll_out_h]
                # #         rews = path['rew'][random_start:random_start + self.roll_out_h]
                # #
                # #     obs_itr = np.concatenate([obs_itr, obs], axis=0)
                # #     batch_obs.append(obs)
                # #     batch_acs.append(acs)
                # #     batch_ac_prevs.append(ac_prevs)
                # #     batch_golden_obs.append(golden_obs)
                # #     batch_rewards.append(rews)
                # #     random_paths.append(path)
                # #
                # # ph_ob = np.array(batch_obs)
                # # ph_golden_ob = np.array(batch_golden_obs)
                # # ph_ac = np.array(batch_acs)
                # # ph_ac_prev = np.array(batch_ac_prevs)
                # re = np.array(batch_rewards)

                ph_ob = np.stack([path['obs'] for path in paths], axis=0)
                ph_golden_ob = np.stack([path['golden_obs'] for path in paths], axis=0)
                re = np.stack([path['rew'] for path in paths], axis=0)
                ph_ac = np.stack([path['ac'] for path in paths], axis=0)
                ph_ac_prev = np.stack([path['prev_ac'] for path in paths], axis=0)
                # obs_itr = np.concatenate([obs_itr, ph_ob[]], axis=0)

                print("// Estimating return ...")
                q, adv = self.estimate_return(ph_ob, re)
                # just checking the shapes to make sure
                # print("[q_n] {}".format(q.shape))
                # print("[adv_n] {}".format(adv.shape))
                print("// taking gradient step ...")
                self.update_parameters(ph_ob, ph_golden_ob, ph_ac, ph_ac_prev, q, adv)

            # # Log diagnostics
            np.save(os.path.join(dirname, '{}'.format(itr)), obs_itr)
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


