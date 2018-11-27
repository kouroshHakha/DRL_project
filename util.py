import tensorflow as tf
import numpy as np

class PPOReplayBuffer(object):
    '''
    replay buffer for PPO algorithm
    store fixed log probs, advantages, and returns for use in multiple updates

    n.b. samples must be added as a batch, and we assume that the
    batch is the same size as that of the simple buffer
    '''

    def __init__(self, simple_buffer):
        self.simple_buffer = simple_buffer
        self.max_size = self.simple_buffer.max_size
        self.flush()

    def flush(self):
        self.simple_buffer.flush()
        self._log_probs = np.zeros((self.max_size, 1))
        self._advantages = np.zeros((self.max_size, 1))
        self._returns = np.zeros((self.max_size, 1))

    def add_samples(self, lp, adv, ret):
        self._log_probs = lp
        self._advantages = adv
        self._returns = ret

    def get_samples(self, indices):
        return dict(
            log_probs = self._log_probs[indices],
            advantages = self._advantages[indices],
            returns = self._returns[indices],
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.simple_buffer._size, batch_size)
        simple = self.simple_buffer.get_samples(indices)
        ppo = self.get_samples(indices)
        return {**simple, **ppo}

def build_mlp(input_layer,
              output_dim,
              scope,
              n_layers=1,
              hidden_dim=500,
              activation=tf.nn.relu,
              output_activation=None,
              reuse=False,
              kernel_regularizer=None):
    layer = input_layer
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            layer = tf.layers.dense(layer, hidden_dim, activation=activation, kernel_regularizer=kernel_regularizer)
        layer = tf.layers.dense(layer, output_dim, activation=output_activation, kernel_regularizer=kernel_regularizer)
    return layer

def print_debug(name, value):
    print("[debug] {}={}".format(name,value))