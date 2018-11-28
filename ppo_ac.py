import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import inspect
import time
from util import *
import logz
import time
import IPython
import pickle
import scipy.signal

def batch_norm(a, mean, std):
    return (a - mean) / (std + 1e-8)

def batch_denorm(a, mean, std):
    return a*std + mean

def pathlength(path):
    return len(path["rew"])

def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir)
    args = inspect.getargspec(PPO.__init__)[0]
    # params = []
    # for k in args:
    #     if (k in locals_) and (k != 'self'):
    #         params.append(locals_[k])
    params = dict(
        exp_name=locals_['self'].env.__class__.__name__
    )
    logz.save_params(params)

class PPO(object):
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
        self.optimizer = tf.train.AdamOptimizer()
        self.l2reg = False


        self.number_of_trajectories_per_iter = self.mini_batch_size
        self.max_path_length = 30
        self.min_timesteps_per_batch = self.max_path_length*self.number_of_trajectories_per_iter


        self.num_grad_steps_per_target_update = 4
        self.num_target_updates = 4

        self.random_goal = False

        num_epochs = 3
        self.num_ppo_updates = num_epochs*(self.number_of_trajectories_per_iter // self.mini_batch_size)

        self.use_lstm = True
        self.hist_dim = 2*self.hist_dim if self.use_lstm else self.hist_dim


        self.gamma = pg_flavor_args['gamma']
        self.reward_to_go = pg_flavor_args['reward_to_go']
        self.nn_baseline = pg_flavor_args['nn_baseline']
        self.normalize_advantages = pg_flavor_args['normalize_advantages']
        seed = pg_flavor_args['seed']
        self.animate = animate

        # initialize random seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.env.seed(seed)

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def _define_placeholders(self):

        self.sy_ob = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ob_dim])
        self.sy_ac = tf.placeholder(dtype=tf.float32, shape=[None, None,  self.ac_dim])
        self.sy_ac_prev = tf.placeholder(dtype=tf.float32, shape=[None, None, self.ac_dim])
        self.sy_adv = tf.placeholder(dtype=tf.float32, shape=[None, None, ])
        self.sy_init_history = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim]) #*2
        self.sy_re_prev = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

    ####################################################################################
        ##### Critic
        ####################################################################################
        self.sy_target_values = tf.placeholder(dtype=tf.float32, shape=[None, None,])
        self.init_critic_history = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim]) #*2

        ####################################################################################
        ##### PPO
        ####################################################################################
        self.sy_old_log_prob_n = tf.placeholder(shape=[None, None], name="fixed_log_prob", dtype=tf.float32)



    def _build_actor(self, sy_ob, sy_ac_prev):
        """

        :param sy_ob: [None, self.ob_dim] -> [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param sy_ac_prev: [None, self.ac_dim] -> [self.mini_batch_size, self.roll_out_h, self.ac_dim]
        :param sy_golden_ob: [None, self.ob_dim] -> [mini_batch, self.roll_out_h, self.ob_dim]
        :return:
            sy_ac_mean: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
            sy_ac_logstd: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        """

        with tf.variable_scope('actor'):

            self.sy_meta_state = tf.concat([sy_ob,
                                            sy_ac_prev], axis=-1)
            self.sy_meta_lstm_in = build_mlp(self.sy_meta_state, self.state_dim, scope='input', n_layers=2, hidden_dim=20, output_activation=tf.nn.relu)

            #Create LSTM cells of length batch_size
            if self.use_lstm:
                self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim//2, state_is_tuple=False, name='lstm')
            else:
                self.lstm_cell = tf.nn.rnn_cell.GRUCell(self.hist_dim, activation=tf.tanh, name='gru')

            self.lstm_out, self.updated_actor_history = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                                            inputs=self.sy_meta_lstm_in,
                                                            initial_state=self.sy_init_history,
                                                            dtype=tf.float32)

            sy_policy_params = build_mlp(self.lstm_out, self.ac_dim*2, scope='out_policy', n_layers=2, hidden_dim=20)
            sy_policy_params = tf.reshape(sy_policy_params, [-1, self.ac_dim*2])
            sy_ac_mean = sy_policy_params[:,:self.ac_dim]
            sy_ac_logstd = sy_policy_params[:,self.ac_dim:]

            return sy_ac_mean, sy_ac_logstd

    def ppo_loss(self, new_log_probs, old_log_probs, advantages, clip_epsilon=0.1, entropy_coeff=0):
        """
        given:
            clip_epsilon

        arguments:
            advantages (mini_bsize,)
            states (mini_bsize,)
            actions (mini_bsize,)
            fixed_log_probs (mini_bsize,)

        intermediate results:
            states, actions --> log_probs
            log_probs, fixed_log_probs --> ratio
            advantages, ratio --> surr1
            ratio, clip_epsilon, advantages --> surr2
            surr1, surr2 --> policy_surr_loss
        """

        # self.ratio = tf.exp(-old_log_probs + new_log_probs)
        # surr1 = -advantages * self.ratio
        # surr2 = -advantages * tf.clip_by_value(self.ratio, 1.0 - clip_epsilon,  1.0 + clip_epsilon)
        # self.policy_surr_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

        ratio = tf.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, clip_value_min=1.0-clip_epsilon, clip_value_max=1.0+clip_epsilon) * advantages
        self.policy_surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        probs = tf.exp(new_log_probs)
        self.entropy = tf.reduce_sum(-(new_log_probs * probs)) * entropy_coeff

        # debug
        self.prob_ent = probs
        self.log_prob_ent = new_log_probs
        # self.entropy = tf.reduce_sum(self.ndist.entropy()) * entropy_coeff
        policy_surr_total_loss = self.policy_surr_loss - self.entropy
        return policy_surr_total_loss

    def _build_critic(self):
        with tf.variable_scope('critic'):
            critic_regul = tf.contrib.layers.l2_regularizer(1e-3) if self.l2reg else None
            sy_critic_meta_state = tf.concat([self.sy_ob,
                                              self.sy_ac_prev], axis=-1)

            sy_critic_meta_lstm_in = build_mlp(sy_critic_meta_state, self.state_dim, scope='input', n_layers=2,
                                               hidden_dim=20, output_activation=tf.nn.relu, kernel_regularizer=critic_regul)
            if self.use_lstm:
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim//2, state_is_tuple=False, name='lstm')
            else:
                lstm_cell = tf.nn.rnn_cell.GRUCell(self.hist_dim, activation=tf.tanh, name='gru')

            lstm_out, self.critic_history = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                              inputs=sy_critic_meta_lstm_in,
                                                              initial_state=self.init_critic_history,
                                                              dtype=tf.float32)

            # self.critic_prediction  = tf.squeeze(build_mlp(lstm_out, 1, activation=None, scope='out_fc',
            #                                                n_layers=2,
            #                                                hidden_dim=20,
            #                                                kernel_regularizer=critic_regul))
            self.critic_prediction  = tf.squeeze(tf.layers.dense(lstm_out, 1, activation=None, name='out_fc',
                                                                 kernel_regularizer=critic_regul))

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
        # return sy_logprob
        return tf.reshape(sy_logprob, [tf.shape(self.sy_ac)[0], tf.shape(self.sy_ac)[1]])

    def _sample_action(self, policy_parameters):

        sy_mean, sy_logstd = policy_parameters

        sy_sampled_ac = tf.add(sy_mean, tf.multiply(tf.exp(sy_logstd), tf.random_normal([self.ac_dim])))
        return sy_sampled_ac

    def build_computation_graph(self):

        self._define_placeholders()

        # The policy takes in observations over all time steps and produces a distribution over the action space
        # at all time steps
        self.policy_parameters = self._build_actor(self.sy_ob, self.sy_ac_prev)
        # print_debug("policy params", self.policy_parameters)

        self.sy_ac_mean, self.sy_ac_logstd = self.policy_parameters
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients_mean = tf.gradients(self.sy_ac_mean,self.variables)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob = self._get_log_prob(self.policy_parameters, self.sy_ac)

        # print_debug("log_prob", self.sy_logprob)
        # print_debug("sy_adv", self.sy_adv)

        # self.loss = -tf.reduce_mean(tf.multiply(self.sy_logprob, sy_adv))

        # self.update_actor_op = self.optimizer.minimize(self.loss)

        self.sy_sampled_ac = self._sample_action(self.policy_parameters)


        ####################################################################################
        ##### Critic
        ####################################################################################
        self._build_critic()
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_values, self.critic_prediction)
        self.critic_update_op = tf.train.AdamOptimizer().minimize(self.critic_loss)

        ####################################################################################
        ##### PPO
        ####################################################################################
        sy_adv = tf.reshape(self.sy_adv, [-1,])
        sy_old_log_prob_n = tf.reshape(self.sy_old_log_prob_n, [-1,])
        sy_logprob = tf.reshape(self.sy_logprob, [-1,])
        self.policy_total_loss = self.ppo_loss(sy_logprob, sy_old_log_prob_n, sy_adv)
        self.update_actor_op = self.optimizer.minimize(self.policy_total_loss)
        self.gradients = self.optimizer.compute_gradients(self.policy_total_loss)


    def sample_trajectories(self, animate_this_episode):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            anim = animate_this_episode and (len(paths) % 10 == 0)
            if anim:
                print("-"*30)
            path = self.sample_trajectory(anim)
            # print('trajectory sampled')
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, animate_this_episode=False):

        obs, next_obs, acs, ac_prevs, rewards, rewards_prev, terminal, input_histories = [], [], [], [], [], [], [], []
        means, stds = [], []

        if self.random_goal:
            z_obs = []
            z_end = np.random.uniform(0,1,2)
            z_start = np.random.uniform(0,1,2)
            ob = self.env.reset(z_start, z_end)
        else: ob = self.env.reset()
        steps = 1
        ac_prev = np.ones([1,1,self.ac_dim])*(1)
        re_prev = -3 #self.env.worst_rew
        init_history = np.zeros([self.hist_dim])#*2

        while True:
            if animate_this_episode:
                self.env.render()
                time.sleep(0.01)
                self.env.close()

            obs.append(ob)
            input_histories.append(init_history)
            ac_prevs.append(ac_prev)
            rewards_prev.append(re_prev)

            feed_dict={
                self.sy_ob: ob[None, None, :],
                self.sy_ac_prev: ac_prev,
                self.sy_init_history: init_history[None, :],
                self.sy_re_prev: np.ones([1, 1, 1]) * re_prev
            }

            ac, history_out = self.sess.run([self.sy_sampled_ac, self.updated_actor_history], feed_dict=feed_dict)
            mean, log_std = self.sess.run([self.sy_ac_mean, self.sy_ac_logstd], feed_dict=feed_dict)
            ac = ac[0]
            history_out = history_out[0]

            if animate_this_episode:
                print('Mean: {}'.format(mean[0]))
                print('std: {}'.format(np.exp(log_std[0])))
                print('ac: {}'.format(ac))
                print('history: {}'.format(history_out))


            next_ob, rew, done, info = self.env.step(ac)
            next_ob = np.squeeze(next_ob)
            acs.append(ac)
            rewards.append(rew)
            next_obs.append(next_ob)
            terminal.append(done)

            means.append(mean[0])
            stds.append(np.exp(log_std[0]))

            ac_prev = ac[None, None, :]
            init_history = history_out
            ob = next_ob
            re_prev = rew
            steps += 1

            if done or steps > self.max_path_length:
                terminal[-1] = True
                break

        path = {"obs" : np.array(obs, dtype=np.float32),
                "next_obs": np.array(next_obs, dtype=np.float32),
                "rew" : np.array(rewards, dtype=np.float32),
                "rew_prev": np.array(rewards_prev, dtype=np.float32),
                "ac" : np.squeeze(np.array(acs, dtype=np.float32)),
                "prev_ac" : np.squeeze(np.array(ac_prevs, dtype=np.float32)),
                "history_in" : np.array(input_histories, dtype=np.float32),
                "terminal": np.array(terminal, dtype=np.float32),
                "means": np.squeeze(np.array(means, dtype=np.float32)),
                "stds": np.squeeze(np.array(stds, dtype=np.float32)),}

        if self.random_goal:
            path['z_obs'] = np.array(z_obs, dtype=np.float32)

        return path

    def rtg(self, re):
        assert re.ndim == 2
        rtg = np.zeros(shape=re.shape)
        for i in range(re.shape[1]):
            gamma_vec = self.gamma**np.arange(0, re.shape[1]-1-i)
            rtg[:, i] = np.sum(re[:, i:], axis=1)
        return rtg

    def compute_q(self, re):
        rewards = re[:, ::-1]
        rtg = scipy.signal.lfilter([1], [1, -self.gamma], rewards, axis=1)[:,::-1]
        # gamma_vector = np.power(self.gamma, range(re.shape[1]))
        # re_dicounted = re*gamma_vector
        # return self.rtg(re_dicounted)
        return rtg

    def estimate_advantage(self, ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, re_prev, ph_terminal):
        init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim], dtype=np.float32) #*2
        # next_h = self.sess.run(self.updated_critic_history, feed_dict={self.sy_ob: ph_ob[:, 0, :][:, None, :],
        #                                                                self.sy_ac_prev: ph_ac_prev[:, 0][:, None, None],
        #                                                                self.sy_re_prev: re_prev[:, 0][:, None, None],
        #                                                                self.sy_critic_history_in: init_critic_history})
        #
        # next_values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_next_ob,
        #                                                                self.sy_ac_prev: ph_ac[:,:, None],
        #                                                                self.sy_re_prev: re[:, :, None],
        #                                                                self.sy_critic_history_in: next_h})
        # q = re + self.gamma * next_values * (1-ph_terminal)
        q = self.compute_q(re)
        #
        # # print('[debug] q:{}, re:{}, next_values:{}, ph_terminal:{}'.format(q.shape, re.shape, next_values.shape, ph_terminal.shape))
        curr_values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_ob,
                                                                       self.sy_ac_prev: ph_ac_prev,
                                                                       self.sy_re_prev: re_prev[:, :, None],
                                                                       self.init_critic_history: init_critic_history,
                                                                       })
        adv = q - curr_values
        if self.normalize_advantages:
            adv = (adv - np.mean(adv.flatten())) / (np.std(adv.flatten()) + 1e-8)
        return adv, q
        # bl_vec = np.mean(re, axis=0)
        # bl = np.broadcast_to(bl_vec, re.shape)
        # adv = q - bl
        # return adv

    def update_critic(self, ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, re_prev, ph_terminal, ph_q):
        init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim], dtype=np.float32) #*2
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            next_h = self.sess.run(self.critic_history, feed_dict={self.sy_ob: ph_ob[:, 0, :][:, None, :],
                                                                           self.sy_ac_prev: ph_ac_prev[:, 0][:, None],
                                                                           self.sy_re_prev: re_prev[:, 0][:, None, None],
                                                                           self.init_critic_history: init_critic_history})
            if i % self.num_grad_steps_per_target_update == 0:
                # the notion of state should shift time not just observation
                next_values_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_next_ob,
                                                                                 self.sy_ac_prev: ph_ac,
                                                                                 self.sy_re_prev: re[:, :, None],
                                                                                 self.init_critic_history: next_h})

                target_n = re + self.gamma * next_values_n * (1 - ph_terminal)
            _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
                                    feed_dict={self.sy_ob: ph_ob,
                                               self.sy_ac_prev: ph_ac_prev,
                                               self.sy_re_prev: re_prev[:, :, None],
                                               self.init_critic_history: init_critic_history,
                                               self.sy_target_values: target_n})

            # init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim], dtype=np.float32) #*2
            # for _ in range(10):
            #     _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
            #                             feed_dict={self.sy_ob: ph_ob,
            #                                        self.sy_ac_prev: ph_ac_prev,
            #                                        self.sy_re_prev: re_prev[:, :, None],
            #                                        self.init_critic_history: init_critic_history,
            #                                        self.sy_target_values: ph_q})

    def update_actor(self, ph_ob, ph_ac, ph_ac_prev, re_prev, adv, ph_old_log_prob):
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
        sy_init_history = np.zeros([ph_ob.shape[0], self.hist_dim], dtype=np.float32) #*2
        # sy_seq_len = np.zeros([self.mini_batch_size, ], dtype=np.int32)

        feed_dict = {
            self.sy_ob: ph_ob,
            self.sy_ac: ph_ac,
            self.sy_ac_prev: ph_ac_prev,
            self.sy_re_prev: re_prev[:, :, None],
            self.sy_adv: adv,
            self.sy_init_history: sy_init_history,
            self.sy_old_log_prob_n: ph_old_log_prob
        }

        # for grad, var in self.gradients:
        #    if grad is not None:
        #        grad_value = self.sess.run(grad, feed_dict=feed_dict)
        #        print("var: {}".format(var.name))
        #        print("-"*30)
        #        print("{}".format(grad_value))
        #    else:
        #        print("var: {}".format(var.name))
        #        print("-"*30)
        #        print("grad is None")

        # IPython.embed()

        # mean_b, logstd, log_prob_dens = self.sess.run([self.sy_ac_mean, self.sy_ac_logstd, self.sy_logprob], feed_dict=feed_dict)
        # std_b = np.exp(logstd)
        # prob_dens_b = np.exp(log_prob_dens)

        # entropy, surr_loss, loss = self.sess.run([self.entropy, self.policy_surr_loss, self.policy_total_loss], feed_dict=feed_dict)
        # print("[Debug_training] e- {}, surr_loss- {}, total_loss- {}".format(entropy, surr_loss, loss))

        self.sess.run([self.update_actor_op], feed_dict=feed_dict)
        entropy, surr_loss, loss = self.sess.run([self.entropy, self.policy_surr_loss, self.policy_total_loss], feed_dict=feed_dict)
        prob, log_prob = self.sess.run([self.prob_ent, self.log_prob_ent], feed_dict=feed_dict)

        # print("[Debug_training] e+ {}, surr_loss+ {}, total_loss+ {}".format(entropy, surr_loss, loss))

        # mean_a, logstd, log_prob_dens = self.sess.run([self.sy_ac_mean, self.sy_ac_logstd, self.sy_logprob], feed_dict=feed_dict)
        # std_a = np.exp(logstd)
        # prob_dens_a = np.exp(log_prob_dens)
        # IPython.embed()

        return entropy, surr_loss, loss, prob, log_prob


    def train(self, n_iter, logdir):
        total_timesteps = 0
        start = time.time()
        # step_count = 0

        setup_logger(logdir, locals())
        dirname = logz.G.output_dir

        for itr in range(n_iter):
            print("********** Iteration %i ************"%itr)
            print("// Sampling Trajectories ...")
            animate = self.animate and (itr % 99 == 0) and (itr > 0)
            paths, timesteps_this_batch = self.sample_trajectories(animate)
            total_timesteps += timesteps_this_batch

            # IPython.embed()
            # obs_itr = np.empty(shape=[0,paths[0]['obs'].shape[1]], dtype=np.float32)

            if self.random_goal:
                ph_z_obs = np.stack([path['z_obs'] for path in paths], axis=0)

            ph_ob = np.stack([path['obs'] for path in paths], axis=0)
            ph_next_ob = np.stack([path['next_obs'] for path in paths], axis=0)
            re = np.stack([path['rew'] for path in paths], axis=0)
            ph_re_prev = np.stack([path['rew_prev'] for path in paths], axis=0)
            ph_ac = np.stack([path['ac'] for path in paths], axis=0)
            ph_ac_prev = np.stack([path['prev_ac'] for path in paths], axis=0)
            ph_terminal = np.stack([path['terminal'] for path in paths], axis=0)
            ph_actor_history_in = np.stack([path['history_in'] for path in paths], axis=0)

            ph_means = np.stack([path['means'] for path in paths], axis=0)
            ph_stds = np.stack([path['stds'] for path in paths], axis=0)

            old_prob_nt = self.sess.run(self.sy_logprob, feed_dict={self.sy_ob: ph_ob,
                                                                    self.sy_ac_prev: ph_ac_prev,
                                                                    self.sy_re_prev: ph_re_prev[:, :, None],
                                                                    self.sy_init_history: np.zeros([ph_ob.shape[0], self.hist_dim], dtype=np.float32), #*2
                                                                    self.sy_ac: ph_ac,
                                                                    })
            # if itr >= 45:
            #     IPython.embed()
            ph_old_prob = np.reshape(old_prob_nt, newshape=[ph_ob.shape[0], ph_ob.shape[1]])


            ph_q = None
            print("// taking gradient steps on critic ...")
            self.update_critic(ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, ph_re_prev, ph_terminal, ph_q)

            print("// getting new advantage estimates ...")
            ph_adv, ph_q = self.estimate_advantage(ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, ph_re_prev, ph_terminal)

            for _ in range(self.num_ppo_updates):

                random_path_indices = np.random.choice(len(paths), self.mini_batch_size, replace=False)
                # random_path_indices = list(range(len(paths)))
                ob = ph_ob[random_path_indices]
                ac = ph_ac[random_path_indices]
                ac_prev = ph_ac_prev[random_path_indices]
                re_prev = ph_re_prev[random_path_indices]
                adv = ph_adv[random_path_indices]
                old_log_prob = old_prob_nt[random_path_indices]
                ent, loss, total_loss, prob, log_prob = self.update_actor(ob, ac, ac_prev, re_prev, adv, old_log_prob)

            # # Log diagnostics
            if self.env.__class__.__name__ == "PointMass":
                # for visualization we need to append the last observation in ph_next_obs to ph_obs and
                # repeat it ac_dim times to make it consistent with the other parts of the ph_obs array
                # therefore, for example, for time horizon of 5 and action dimm of 2 there should be 12
                # total time steps for visualization per batch
                last_ob = [ph_next_ob[:,-1][:, None, :]]
                ob = np.concatenate([ph_ob]+last_ob*self.ac_dim, axis=1)
                obs_log = {'ob': ob}
                with open(os.path.join(dirname, '{}.dpkl'.format(itr)), 'wb') as f:
                    pickle.dump(obs_log, f)

            if itr > 1:
                print('std: ',ph_stds[0])
                print('mean: ',ph_means[0])
                print('ac: ',ph_ac[0])
                print('next_ob:', ph_next_ob[0])
                print('re:', re[0])
                print('q:', ph_q[0])
                print('adv:', ph_adv[0])
                print('cur_val:', ph_q[0] - ph_adv[0])

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



