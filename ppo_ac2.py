"""
A new vesion of ppo based on a new structure of MDP in progress, latest stable version is ppo_ac.py
"""
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
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.l2reg = True

        self.number_of_trajectories_per_iter = 64
        self.max_path_length = 5
        self.min_timesteps_per_batch = self.max_path_length*self.ac_dim*self.number_of_trajectories_per_iter

        self.num_grad_steps_per_target_update = 4
        self.num_target_updates = 4

        self.random_goal = False

        num_epochs = 10
        self.num_ppo_updates = num_epochs*(self.number_of_trajectories_per_iter // self.mini_batch_size)


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
        self.sy_ac = tf.placeholder(dtype=tf.float32, shape=[None, None,  1])
        self.sy_ac_prev = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        self.sy_adv = tf.placeholder(dtype=tf.float32, shape=[None, None, ])
        self.sy_actor_history_in = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim * 2])
        self.sy_re_prev = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        # self.sy_ob = tf.placeholder(dtype=tf.float32, shape=[None, self.ob_dim])
        # self.sy_actor_history_in = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim * 2])
        # self.sy_ac = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # self.sy_ac_prev = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # self.sy_adv = tf.placeholder(dtype=tf.float32, shape=[None, ])

        ####################################################################################
        ##### Critic
        ####################################################################################
        self.sy_target_values = tf.placeholder(dtype=tf.float32, shape=[None, None ])
        self.sy_critic_history_in = tf.placeholder(dtype=tf.float32, shape=[None, self.hist_dim * 2])

        ####################################################################################
        ##### PPO
        ####################################################################################
        self.sy_old_log_prob_n = tf.placeholder(shape=[None, None], name="fixed_log_prob", dtype=tf.float32)



    def _build_actor(self):
        """

        :param sy_ob: [None, self.ob_dim] -> [self.mini_batch_size, self.roll_out_h, self.ob_dim]
        :param sy_ac_prev: [None, self.ac_dim] -> [self.mini_batch_size, self.roll_out_h, self.ac_dim]
        :param sy_golden_ob: [None, self.ob_dim] -> [mini_batch, self.roll_out_h, self.ob_dim]
        :return:
            sy_ac_mean: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
            sy_ac_logstd: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        """

        with tf.variable_scope('actor'):

            self.sy_meta_state = tf.concat([self.sy_ob,
                                            self.sy_ac_prev,
                                            #self.sy_re_prev,
                                            ], axis=-1)
            self.sy_meta_lstm_in = build_mlp(self.sy_meta_state, self.state_dim, scope='input', n_layers=2, hidden_dim=20, output_activation=tf.nn.relu)

            #Create LSTM cells of length batch_size
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim, state_is_tuple=False, name='lstm')
            self.lstm_out, self.updated_actor_history = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                                                          inputs=self.sy_meta_lstm_in,
                                                                          initial_state=self.sy_actor_history_in,
                                                                          dtype=tf.float32)

            sy_policy_params = build_mlp(self.lstm_out, 2, scope='out_policy', n_layers=2, hidden_dim=20)
            # sy_policy_params = tf.layers.dense(self.lstm_out, 2, activation=None, name='out_policy_fc')
            sy_policy_params = tf.reshape(sy_policy_params, [-1, 2])
            self.sy_ac_mean = sy_policy_params[:,:1]
            self.sy_ac_logstd = sy_policy_params[:,1:]

            # making std a trainable variable independent of inputs
            # self.sy_ac_mean = tf.layers.dense(self.lstm_out, 1, activation=None, name='out_policy_fc')
            # self.sy_ac_mean = tf.reshape(self.sy_ac_mean, [-1, 1])
            # sy_ac_logstd = tf.get_variable(shape=[1], dtype=tf.float32, name='log_std', trainable=True)
            # self.sy_ac_logstd = tf.fill(tf.shape(self.sy_ac_mean), 1.0)*sy_ac_logstd

    def ppo_loss(self, new_log_probs, old_log_probs, advantages, clip_epsilon=0.1, entropy_coeff=1e-4):
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
        self.ratio = tf.exp(new_log_probs - old_log_probs)
        surr1 = self.ratio * advantages
        surr2 = tf.clip_by_value(self.ratio, clip_value_min=1.0-clip_epsilon, clip_value_max=1.0+clip_epsilon) * advantages
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
            sy_critic_meta_state = tf.concat([self.sy_ob,
                                              self.sy_ac_prev,
                                              # self.sy_re_prev,
                                              ], axis=-1)

            critic_regul = tf.contrib.layers.l2_regularizer(1e-3) if self.l2reg else None
            sy_critic_meta_lstm_in = build_mlp(sy_critic_meta_state, self.state_dim, scope='input', n_layers=2,
                                               hidden_dim=20, output_activation=tf.nn.relu, kernel_regularizer=critic_regul)

            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim, state_is_tuple=False, name='lstm')
            # lstm_out, self.updated_critic_history = lstm_cell(sy_critic_meta_lstm_in, self.sy_critic_history_in)
            lstm_out, self.updated_critic_history = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                                      inputs=sy_critic_meta_lstm_in,
                                                                      initial_state=self.sy_critic_history_in,
                                                                      dtype=tf.float32)
            self.critic_prediction  = tf.squeeze(tf.layers.dense(lstm_out, 1, activation=None, name='out_fc', kernel_regularizer=critic_regul))


    def _build_log_prob(self):
        """
        :param policy_parameters: (sy_mean, sy_logstd)
            sy_mean: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
            sy_logstd: [None, self.ob_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        :param
            sy_ac: [None, self.ac_dim] -> [self.mini_batch_size * self.roll_out_h, self.ac_dim]
        :return:
            sy_logprob: [None, ] -> [self.mini_batch_size, self.roll_out_h]
        """
        # self.mvn = tfp.distributions.MultivariateNormalDiag(loc=self.sy_ac_mean, scale_diag=tf.exp(self.sy_ac_logstd))
        # sy_logprob = self.mvn.log_prob(sy_ac_in)
        self.ndist = tfp.distributions.Normal(loc=self.sy_ac_mean, scale=tf.exp(self.sy_ac_logstd))
        sy_ac_in = tf.reshape(self.sy_ac, [tf.shape(self.sy_ac)[0] * tf.shape(self.sy_ac)[1], 1])
        sy_logprob = self.ndist.log_prob(sy_ac_in)
        return tf.reshape(sy_logprob, [tf.shape(self.sy_ac)[0], tf.shape(self.sy_ac)[1]])

    def _sample_action(self):

        sy_mean, sy_logstd = self.sy_ac_mean, self.sy_ac_logstd

        sy_sampled_ac = tf.add(sy_mean, tf.multiply(tf.exp(sy_logstd), tf.random_normal([1])))
        return sy_sampled_ac

    def build_computation_graph(self):

        self._define_placeholders()

        # The policy takes in observations over all time steps and produces a distribution over the action space
        # at all time steps
        self._build_actor()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients_mean = tf.gradients(self.sy_ac_mean, self.variables)

        print_debug('means', self.sy_ac_mean)
        print_debug('stds', self.sy_ac_logstd)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob = self._build_log_prob()


        print_debug("log_prob", self.sy_logprob)
        print_debug("sy_adv", self.sy_adv)

        # self.loss = -tf.reduce_mean(tf.multiply(self.sy_logprob, sy_adv))

        # self.update_actor_op = self.optimizer.minimize(self.loss)

        self.sy_sampled_ac = self._sample_action()


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
        env_ac = []
        means, stds = [], []
        if self.random_goal:
            z_obs = []
            z_end = np.random.uniform(0,1,2)
            z_start = np.random.uniform(0,1,2)
            ob = self.env.reset(z_start, z_end)
        else: ob = self.env.reset()
        steps = 1
        ac_prev = -1
        re_prev = -3 #self.env.worst_rew
        init_history = np.zeros([self.hist_dim*2])

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
                self.sy_ac_prev: np.ones([1,1,1])*ac_prev,
                self.sy_actor_history_in: init_history[None, :],
                self.sy_re_prev: np.ones([1, 1, 1]) * re_prev
            }

            ac, history_out = self.sess.run([self.sy_sampled_ac, self.updated_actor_history], feed_dict=feed_dict)
            mean, log_std = self.sess.run([self.sy_ac_mean, self.sy_ac_logstd], feed_dict=feed_dict)
            ac = ac[0,0]
            history_out = history_out[0]
            env_ac.append(ac)
            if animate_this_episode:
                print('Mean: {}'.format(mean[0,0]))
                print('std: {}'.format(np.exp(log_std[0,0])))
                print('ac: {}'.format(ac))
                print('history: {}'.format(history_out))

            if steps % self.ac_dim == 0:
                next_ob, rew, done, info = self.env.step(np.array(env_ac))
                next_ob = np.squeeze(next_ob)
                env_ac = []
            else:
                next_ob, rew, done, info = ob, 0, False, None

            acs.append(ac)
            rewards.append(rew)
            next_obs.append(next_ob)
            terminal.append(done)
            if self.random_goal:
                z_obs.append(z_end)

            means.append(mean[0,0])
            stds.append(np.exp(log_std[0,0]))

            ac_prev = ac
            init_history = history_out
            ob = next_ob
            re_prev = rew
            steps += 1

            if done or steps > (self.max_path_length * self.ac_dim):
                terminal[-1] = True
                break

        # IPython.embed()
        path = {"obs" : np.array(obs, dtype=np.float32),
                "next_obs": np.array(next_obs, dtype=np.float32),
                "rew" : np.array(rewards, dtype=np.float32),
                "rew_prev": np.array(rewards_prev, dtype=np.float32),
                "ac" : np.array(acs, dtype=np.float32),
                "prev_ac" : np.array(ac_prevs, dtype=np.float32),
                "history_in" : np.array(input_histories, dtype=np.float32),
                "terminal": np.array(terminal, dtype=np.float32),
                "means": np.array(means, dtype=np.float32),
                "stds": np.array(stds, dtype=np.float32),}
        if self.random_goal:
            path['z_obs'] = np.array(z_obs, dtype=np.float32)
        return path
    def rtg(self, re):
        assert re.ndim == 2
        rtg = np.zeros(shape=re.shape)
        for i in range(re.shape[1]):
            rtg[:, i] = np.sum(re[:, i:], axis=1)
        return rtg

    def compute_q(self, re):
        gamma_vector = np.power(self.gamma, range(re.shape[1]))
        re_dicounted = re*gamma_vector
        return self.rtg(re_dicounted)

    def estimate_advantage(self, ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, re_prev, ph_terminal):
        init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
        next_h = self.sess.run(self.updated_critic_history, feed_dict={self.sy_ob: ph_ob[:, 0, :][:, None, :],
                                                                       self.sy_ac_prev: ph_ac_prev[:, 0][:, None, None],
                                                                       self.sy_re_prev: re_prev[:, 0][:, None, None],
                                                                       self.sy_critic_history_in: init_critic_history})

        next_values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_next_ob,
                                                                       self.sy_ac_prev: ph_ac[:,:, None],
                                                                       self.sy_re_prev: re[:, :, None],
                                                                       self.sy_critic_history_in: next_h})
        q = re + self.gamma * next_values * (1-ph_terminal)

        q = self.compute_q(re)
        # print('[debug] q:{}, re:{}, next_values:{}, ph_terminal:{}'.format(q.shape, re.shape, next_values.shape, ph_terminal.shape))
        curr_values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_ob,
                                                                       self.sy_ac_prev: ph_ac_prev[:,:, None],
                                                                       self.sy_re_prev: re_prev[:, :, None],
                                                                       self.sy_critic_history_in: init_critic_history,
                                                                       })
        adv = q - curr_values
        if self.normalize_advantages:
            adv = (adv - np.mean(adv.flatten())) / (np.std(adv.flatten()) + 1e-8)
        return adv
        # return q

    def update_critic(self, ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, re_prev, ph_terminal):
        init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            next_h = self.sess.run(self.updated_critic_history, feed_dict={self.sy_ob: ph_ob[:, 0, :][:, None, :],
                                                                           self.sy_ac_prev: ph_ac_prev[:, 0][:, None, None],
                                                                           self.sy_re_prev: re_prev[:, 0][:, None, None],
                                                                           self.sy_critic_history_in: init_critic_history})
            if i % self.num_grad_steps_per_target_update == 0:
                # the notion of state should shift time not just observation
                next_values_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_next_ob,
                                                                                 self.sy_ac_prev: ph_ac[:,:, None],
                                                                                 self.sy_re_prev: re[:, :, None],
                                                                                 self.sy_critic_history_in: next_h})
                target_n = re + self.gamma * next_values_n * (1 - ph_terminal)
            _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
                                    feed_dict={self.sy_ob: ph_ob,
                                               self.sy_ac_prev: ph_ac_prev[:,:,None],
                                               self.sy_re_prev: re_prev[:, :, None],
                                               self.sy_critic_history_in: init_critic_history,
                                               self.sy_target_values: target_n})

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
        sy_init_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
        # sy_seq_len = np.zeros([self.mini_batch_size, ], dtype=np.int32)

        feed_dict = {
            self.sy_ob: ph_ob,
            self.sy_ac: ph_ac[:,:, None],
            self.sy_ac_prev: ph_ac_prev[:,:, None],
            self.sy_re_prev: re_prev[:, :, None],
            self.sy_adv: adv,
            self.sy_actor_history_in: sy_init_history,
            self.sy_old_log_prob_n: ph_old_log_prob
        }


        #
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

        # entropy, surr_loss, loss = self.sess.run([self.entropy, self.policy_surr_loss, self.policy_total_loss], feed_dict=feed_dict)
        # print("[Debug_training] e- {}, surr_loss- {}, total_loss- {}".format(entropy, surr_loss, loss))
        self.sess.run([self.update_actor_op], feed_dict=feed_dict)
        entropy, surr_loss, loss = self.sess.run([self.entropy, self.policy_surr_loss, self.policy_total_loss], feed_dict=feed_dict)
        prob, log_prob = self.sess.run([self.prob_ent, self.log_prob_ent], feed_dict=feed_dict)
        # print("[Debug_training] e+ {}, surr_loss+ {}, total_loss+ {}".format(entropy, surr_loss, loss))


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


            ## check if means and stds are equal when policy is rolled out
            # ret_mean, ret_log_std = self.sess.run([self.sy_ac_mean, self.sy_ac_logstd], feed_dict={self.sy_ob: ph_ob,
            #                                                                                        self.sy_ac_prev: ph_ac_prev[:,:, None],
            #                                                                                        self.sy_re: ph_re_prev[:,:, None],
            #                                                                                        self.sy_actor_history_in: np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32),
            #                                                                                        })
            # ret_mean = np.reshape(ret_mean, newshape=ph_means.shape)
            # ret_log_std = np.reshape(ret_log_std, newshape=ph_stds.shape)
            # ret_std = np.exp(ret_log_std)

            old_prob_nt = self.sess.run(self.sy_logprob, feed_dict={self.sy_ob: ph_ob,
                                                                    self.sy_ac_prev: ph_ac_prev[:,:,None],
                                                                    self.sy_re_prev: ph_re_prev[:, :, None],
                                                                    self.sy_actor_history_in: np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32),
                                                                    self.sy_ac: ph_ac[:,:,None],
                                                                    })
            # if itr >= 45:
            #     IPython.embed()

            ph_old_prob = np.reshape(old_prob_nt, newshape=[ph_ob.shape[0], ph_ob.shape[1]])
            print("// taking gradient steps on critic ...")
            self.update_critic(ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, ph_re_prev, ph_terminal)

            print("// getting new advantage estimates ...")
            ph_adv = self.estimate_advantage(ph_ob, ph_ac, ph_ac_prev, ph_next_ob, re, ph_re_prev, ph_terminal)

            # if itr >= 45:
            #     IPython.embed()

            print("// taking gradient step on actor ...")
            # sy_init_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
            # feed_dict = {
            #     self.sy_ob: ph_ob,
            #     self.sy_ac: ph_ac[:,:, None],
            #     self.sy_ac_prev: ph_ac_prev[:,:, None],
            #     self.sy_re_prev: ph_re_prev[:, :, None],
            #     # self.sy_adv: adv,
            #     self.sy_actor_history_in: sy_init_history,
            #     self.sy_old_log_prob_n: old_prob_nt
            # }
            # ratio_minus = self.sess.run(self.ratio, feed_dict=feed_dict)
            # ratio_minus = np.reshape(ratio_minus, [-1,])
            # print('ratio_before_update:{}'.format(ratio_minus[43]))
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
                # print("e:{}, l1:{}, tl:{}".format(ent,
                #                                   loss,
                #                                   total_loss))


        # ratio_plus = self.sess.run(self.ratio, feed_dict=feed_dict)
            # ratio_plus = np.reshape(ratio_plus, [-1,])
            # print('ratio_after_update:{}'.format(ratio_plus[43]))

            # if itr >= 45:
            #     IPython.embed()
            #
            # new_prob = self.sess.run(self.sy_logprob, feed_dict={self.sy_ob: ph_ob,
            #                                                      self.sy_ac_prev: ph_ac_prev[:,:,None],
            #                                                      self.sy_re_prev: ph_re_prev[:, :, None],
            #                                                      self.sy_actor_history_in: np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32),
            #                                                      self.sy_ac: ph_ac[:,:,None],
            #                                                      })
            #
            # print("-"*30+'// the index of trajectories which met goal at least once')
            # goal_met_trajectories = []
            # for i in range(len(paths)):
            #     if 10.0 in paths[i]['rew']:
            #         goal_met_trajectories.append(i)
            # print(goal_met_trajectories)
            # traj_id_with_increased_prob = []
            # print("-"*30+'// the index of trajectories got higher probability after update')
            # for i in range(len(paths)):
            #     if np.exp(np.sum(old_prob_nt[i])) <= np.exp(np.sum(new_prob[i])):
            #         traj_id_with_increased_prob.append(i)
            # print(traj_id_with_increased_prob)
            # print("-"*50+'// mean of ac, mean, and std')
            #
            # actions, action_means, action_stds = ph_ac.flatten(), ph_means.flatten(), ph_stds.flatten()
            # ac0, ac1 = [actions[k] for k in range(len(actions)) if k%2==0], [actions[k] for k in range(len(actions)) if k%2!=0]
            # mu0, mu1=  [action_means[k] for k in range(len(action_means)) if k%2==0], [action_means[k] for k in range(len(action_means)) if k%2!=0]
            # std0, std1=  [action_stds[k] for k in range(len(action_stds)) if k%2==0], [action_stds[k] for k in range(len(action_stds)) if k%2!=0]
            # print('mu0:{}'.format(np.mean(mu0)))
            # print('std0:{}'.format(np.mean(std0)))
            # print('ac0:{}'.format(np.mean(ac0)))
            # print('mu1:{}'.format(np.mean(mu1)))
            # print('std1:{}'.format(np.mean(std1)))
            # print('ac1:{}'.format(np.mean(ac1)))

            # # Log diagnostics
            # if itr >= 43 and itr <= 47:
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
            if itr > 1:
                print('std: ',ph_stds[0])
                print('mean: ',ph_means[0])
                print('ac: ',ph_ac[0])
                print('ent:', ent)
                print('loss:', loss)
                print('t_loss:', total_loss)
                print('prob:', prob[:10])
                print('log_prob:', log_prob[:10])
            #     IPython.embed()
