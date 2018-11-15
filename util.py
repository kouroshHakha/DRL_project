import tensorflow as tf
import numpy as np
import math

def build_mlp(input_layer,
              output_dim,
              scope,
              n_layers=1,
              hidden_dim=500,
              activation=tf.nn.relu,
              output_activation=None,
              reuse=False):
    layer = input_layer
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            layer = tf.layers.dense(layer, hidden_dim, activation=activation)
        layer = tf.layers.dense(layer, output_dim, activation=output_activation)
    return layer

def print_debug(name, value):
    print("[debug] {}={}".format(name,value))