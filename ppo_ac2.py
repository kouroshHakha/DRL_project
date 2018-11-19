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
        self.optimizer = tf.train.AdamOptimizer()

        self.number_of_trajectories_per_iter = 64
        self.max_path_length = 10
        self.min_timesteps_per_batch = self.max_path_length*self.ac_dim*self.number_of_trajectories_per_iter

        self.num_grad_steps_per_target_update = 4
        self.num_target_updates = 4

        num_epochs = 5
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

            self.sy_meta_state = tf.concat([self.sy_ob, self.sy_ac_prev], axis=-1)
            self.sy_meta_lstm_in = build_mlp(self.sy_meta_state, self.state_dim, scope='input', n_layers=2, hidden_dim=20, output_activation=tf.nn.relu)

            #Create LSTM cells of length batch_size
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim, state_is_tuple=False, name='lstm')
            self.lstm_out1, self.updated_actor_history1 = self.lstm_cell(self.sy_meta_lstm_in[:,0,:], self.sy_actor_history_in)
            self.lstm_out, self.updated_actor_history = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                                                          inputs=self.sy_meta_lstm_in,
                                                                          initial_state=self.sy_actor_history_in,
                                                                          dtype=tf.float32)

            sy_policy_params = tf.layers.dense(self.lstm_out, 2, activation=None, name='out_policy_fc')
            sy_policy_params = tf.reshape(sy_policy_params, [-1, 2])
            self.sy_ac_mean = sy_policy_params[:,:1]
            self.sy_ac_logstd = sy_policy_params[:,1:]

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
        ratio = tf.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, clip_value_min=1.0-clip_epsilon, clip_value_max=1.0+clip_epsilon) * advantages
        policy_surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        probs = tf.exp(new_log_probs)
        entropy = tf.reduce_sum(-(new_log_probs * probs))
        policy_surr_loss -= entropy_coeff * entropy
        return policy_surr_loss

    def _build_critic(self):
        with tf.variable_scope('critic'):
            sy_critic_meta_state = tf.concat([self.sy_ob,
                                              self.sy_ac_prev], axis=-1)

            sy_critic_meta_lstm_in = build_mlp(sy_critic_meta_state, self.state_dim, scope='input', n_layers=2,
                                               hidden_dim=20, output_activation=tf.nn.relu)

            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hist_dim, state_is_tuple=False, name='lstm')
            # lstm_out, self.updated_critic_history = lstm_cell(sy_critic_meta_lstm_in, self.sy_critic_history_in)
            lstm_out, self.updated_critic_history = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                                      inputs=sy_critic_meta_lstm_in,
                                                                      initial_state=self.sy_critic_history_in,
                                                                      dtype=tf.float32)
            self.critic_prediction  = tf.squeeze(tf.layers.dense(lstm_out, 1, activation=None, name='out_fc'))


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
        mvn = tfp.distributions.MultivariateNormalDiag(loc=self.sy_ac_mean, scale_diag=tf.exp(self.sy_ac_logstd))
        sy_ac_in = tf.reshape(self.sy_ac, [tf.shape(self.sy_ac)[0] * tf.shape(self.sy_ac)[1], 1])
        sy_logprob = mvn.log_prob(sy_ac_in)
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
        self.gradients_mean = tf.gradients(self.sy_ac_mean,self.variables)

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
        self.policy_surr_loss = self.ppo_loss(sy_logprob, sy_old_log_prob_n, sy_adv)
        self.update_actor_op = self.optimizer.minimize(self.policy_surr_loss)


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
        obs, next_obs, acs, ac_prevs, rewards, terminal, input_histories = [], [], [], [], [], [], []
        env_ac = []
        ob = self.env.reset()
        steps = 1
        ac_prev = -1
        init_history = np.zeros([self.hist_dim*2])

        while True:
            if animate_this_episode:
                self.env.render()
                time.sleep(0.01)
                self.env.close()

            obs.append(ob)
            input_histories.append(init_history)
            ac_prevs.append(ac_prev)

            feed_dict={
                self.sy_ob: ob[None, None, :],
                self.sy_ac_prev: np.ones([1,1,1])*ac_prev,
                self.sy_actor_history_in: init_history[None, :],
            }

            ac, history_out = self.sess.run([self.sy_sampled_ac, self.updated_actor_history], feed_dict=feed_dict)
            ac = ac[0,0]
            history_out = history_out[0]
            env_ac.append(ac)
            if animate_this_episode:
                mean, log_std = self.sess.run([self.sy_ac_mean, self.sy_ac_logstd], feed_dict=feed_dict)
                print('Mean: {}'.format(mean[0]))
                print('std: {}'.format(np.exp(log_std[0])))
                print('ac: {}'.format(ac))
                print('history: {}'.format(history_out[0]))

            if steps % self.ac_dim == 0:
                next_ob, rew, done, info = self.env.step(np.array(env_ac))
                next_ob = np.squeeze(next_ob)
                env_ac = []
            else:
                next_ob, rew, done = ob, 0, False

            acs.append(ac)
            rewards.append(rew)
            next_obs.append(next_ob)
            terminal.append(done)

            ac_prev = ac
            init_history = history_out
            ob = next_ob
            steps += 1

            if done or steps > (self.max_path_length * self.ac_dim):
                break

        # IPython.embed()
        path = {"obs" : np.array(obs, dtype=np.float32),
                "next_obs": np.array(next_obs, dtype=np.float32),
                "rew" : np.array(rewards, dtype=np.float32),
                "ac" : np.array(acs, dtype=np.float32),
                "prev_ac" : np.array(ac_prevs, dtype=np.float32),
                "history_in" : np.array(input_histories, dtype=np.float32),
                "terminal": np.array(terminal, dtype=np.float32)}
        return path


    def estimate_advantage(self, ph_ob, ph_ac_prev, ph_next_ob, re, ph_terminal):
        init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
        next_values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_next_ob,
                                                                       self.sy_ac_prev: ph_ac_prev[:,:, None],
                                                                       self.sy_critic_history_in: init_critic_history,
                                                                       })
        q = re + self.gamma * next_values * (1-ph_terminal)
        # print('[debug] q:{}, re:{}, next_values:{}, ph_terminal:{}'.format(q.shape, re.shape, next_values.shape, ph_terminal.shape))
        curr_values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_ob,
                                                                       self.sy_ac_prev: ph_ac_prev[:,:, None],
                                                                       self.sy_critic_history_in: init_critic_history,
                                                                       })
        adv = q - curr_values
        if self.normalize_advantages:
            adv = (adv - np.mean(adv.flatten())) / (np.std(adv.flatten()) + 1e-8)
        return adv

    def update_critic(self, ph_ob, ph_ac_prev, ph_next_ob, re, ph_terminal):
        init_critic_history = np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32)
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                next_values_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob: ph_next_ob,
                                                                                 self.sy_ac_prev: ph_ac_prev[:,:, None],
                                                                                 self.sy_critic_history_in: init_critic_history})
                target_n = re + self.gamma * next_values_n * (1 - ph_terminal)
            _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
                                    feed_dict={self.sy_ob: ph_ob,
                                               self.sy_ac_prev: ph_ac_prev[:,:,None],
                                               self.sy_critic_history_in: init_critic_history,
                                               self.sy_target_values: target_n})

    def update_actor(self, ph_ob, ph_ac, ph_ac_prev, adv, ph_old_log_prob):
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
            self.sy_adv: adv,
            self.sy_actor_history_in: sy_init_history,
            self.sy_old_log_prob_n: ph_old_log_prob
        }

        l, = self.sess.run([self.policy_surr_loss], feed_dict=feed_dict)

        print("[Debug_training] l- {}".format(l))
        self.sess.run([self.update_actor_op], feed_dict=feed_dict)
        l, = self.sess.run([self.policy_surr_loss], feed_dict=feed_dict)
        print("[Debug_training] l+ {}".format(l))


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
            IPython.embed()

            # obs_itr = np.empty(shape=[0,paths[0]['obs'].shape[1]], dtype=np.float32)

            ph_ob = np.stack([path['obs'] for path in paths], axis=0)
            ph_next_ob = np.stack([path['next_obs'] for path in paths], axis=0)
            re = np.stack([path['rew'] for path in paths], axis=0)
            ph_ac = np.stack([path['ac'] for path in paths], axis=0)
            ph_ac_prev = np.stack([path['prev_ac'] for path in paths], axis=0)
            ph_terminal = np.stack([path['terminal'] for path in paths], axis=0)
            ph_actor_history_in = np.stack([path['history_in'] for path in paths], axis=0)
            # # Log diagnostics
            if self.env.__class__.__name__ == "PointMass":
                obs_log = dict(
                    ob=ph_ob,
                )
                with open(os.path.join(dirname, '{}.dpkl'.format(itr)), 'wb') as f:
                    pickle.dump(obs_log, f)

            old_prob_nt = self.sess.run(self.sy_logprob, feed_dict={self.sy_ob: ph_ob,
                                                                    self.sy_ac_prev: ph_ac_prev[:,:,None],
                                                                    self.sy_actor_history_in: np.zeros([ph_ob.shape[0], self.hist_dim*2], dtype=np.float32),
                                                                    self.sy_ac: ph_ac[:,:,None],
                                                                    })
            IPython.embed()

            # ph_old_prob = np.reshape(old_prob_nt, newshape=[ph_ob.shape[0], ph_ob.shape[1]])
            print("// taking gradient steps on critic ...")
            self.update_critic(ph_ob, ph_ac_prev, ph_next_ob, re, ph_terminal)
            IPython.embed()
            print("// getting new advantage estimates ...")
            ph_adv = self.estimate_advantage(ph_ob, ph_ac_prev, ph_next_ob, re, ph_terminal)
            IPython.embed()

            for _ in range(self.num_ppo_updates):

                random_path_indices = np.random.choice(len(paths), self.mini_batch_size, replace=False)
                ob = ph_ob[random_path_indices]
                ac = ph_ac[random_path_indices]
                ac_prev = ph_ac_prev[random_path_indices]
                adv = ph_adv[random_path_indices]
                old_log_prob = old_prob_nt[random_path_indices]
                print("// taking gradient step on actor ...")
                self.update_actor(ob, ac, ac_prev, adv, old_log_prob)

            IPython.embed()


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


