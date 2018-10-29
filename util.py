import tensorflow as tf
import numpy as np
import math

activations_dict = {
    'relu': tf.nn.relu,
    'tanh': tf.tanh,
    'sigmoid': tf.sigmoid
}

class _CompositeLayer:


    def __init__(self, components, op):

        """
        constructs a layer that operates as a composite of other primitive layers

        Parameters:
        ----------
        components: list
            the list of layers participating in the composition
        op: function
            the composite operation of the layer
        """

        self.components = components
        self.op = op


    def __call__(self, X):
        """
        computes and returns the the output of the layer

        Parameters:
        ----------
        X: Tensor
            The input tensor

        Returns: Tensor
            defined by self.op
        """

        return self.op(self, X)

    def assign_to(self, other, session):
        """
        Assigns the current layer to an other

        Parameters:
        ----------
        other: _CompositeLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """

        if isinstance(other, _CompositeLayer) and other.op == self.op:
            for i, component in enumerate(self.components):
                component.assign_to(other.components[i], session)
        else:
            raise TypeError("Cannot assign _CompositeLayer: mismatch in type or op")


    def clone(self):
        """
        Clones the current layer into a new layer instance
        """
        cloned_components = [component.clone() for component in self.components]
        clone = _CompositeLayer(cloned_components, self.op)

        return clone


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """
        var_list = []

        for layer in self.components:
            var_list.extend(layer.get_variables())

        return var_list

def Sequence(layers):
    """
    defines a _CompositeLayer that runs component layers in sequence

    Parameters:
    ----------
    layers: list
        component layers
    """

    def sequence_op(obj, X):
        output = X
        for layer in obj.components:
            output = layer(output)

        return output

    return _CompositeLayer(layers, sequence_op)


def Merge(layers, axis):
    """
    defines a _CompositeLayer that merges the output of component layers

    Parameters:
    ----------
    layers: list
        component layers
    axis: int
        the axis to merge on
    """

    def merge_op(obj, X):
        return tf.concat([layer(X) for layer in obj.components], axis)

    return _CompositeLayer(layers, merge_op)

class _OperationalLayer:


    def __init__(self, op, params):
        """
        constrcuts a layer that merely transform data as the propgagte
        through the network

        Parameters:
        ----------
        op: function
            The operation to be done on the data
        params: list
            the list of parameters needed to perform the op
        """
        self.op = op
        self.params = params


    def __call__(self, X):
        """
        Performs the defined operation on the input data

        Parameters:
        ----------
        X: Tensor
        Returns: Tensor
        """
        return self.op(self, X)


    def assign_to(self, other, session):
        """
        Assigns the current layer to an other

        Parameters:
        ----------
        other: _CompositeLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """
        if not isinstance(other, _OperationalLayer) or not self.op == other.op:
            raise TypeError("Cannot assign _OperationalLayer: mismatch in type or op")
        else:
            self.params = other.params


    def clone(self):
        """
        Clones the current layer into a new layer instance
        """
        return _OperationalLayer(self.op, self.params[:])


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """
        return []

def Reshape(new_shape):
    """
    defines an _OperationalLayer that reshapes the input to new_shape

    Parameters:
    ----------
    new_shape: list | tuple | function
    Returns: _OperationalLayer
    """

    def reshape_op(obj, X):
        dummy_lambda = lambda x:x
        new_shape = obj.params[0]
        if isinstance(obj.params[0], type(dummy_lambda)):
            old_shape = X.get_shape().as_list()
            print('[db] old;', old_shape)
            new_shape = obj.params[0](old_shape)
            print('[db] new;', new_shape)

        return tf.reshape(X, new_shape)

    return _OperationalLayer(reshape_op, [new_shape])


def Unroll(axis, num=None):
    """
    defines an _OperationalLayer that unpacks a tensor along a given axis

    Parameters:
    ----------
    axis: int
    num: int
        the numeber if tensors to unpack form the gievn tensor
    Returns: _OperationalLayer
    """

    def unroll_op(obj, X):
        return tf.unstack(X, obj.params[0], obj.params[1])

    return _OperationalLayer(unroll_op, [num, axis])


def Roll(axis):
    """
    defines an _OperationalLayer that packs a list of tensors on a given axis

    Parameters:
    -----------
    axis: int
    Returns: _OperationalLayer
    """

    def roll_op(obj, X):
        return tf.stack(X, axis=obj.params[0])

    return _OperationalLayer(roll_op, [axis])


class FCLayer(object):

    # a static variable to keep count of FCLayers created
    created_count = 0

    def __init__(self, input_size, output_size, activation='relu', name='fclayer'):
        """
        constructs a fully-connected layer

        Parameters:
        ----------
        input_size: int
            the size of the input vector to the layer
        output_size: int
            the size of the output vector from the layer
        activation: string
            the name of the activation function
        name: string
            the name of the layer (useful for saving and loading a model)
        """

        global activations_dict

        FCLayer.created_count += 1

        self.id = name + '_' + str(FCLayer.created_count) if name == 'fclayer' else name
        self.input_size = input_size
        self.output_size = output_size
        self.stddev = min(0.02, math.sqrt(2. / input_size))
        self.activation = activation
        self.activation_fn = activations_dict[activation]

        self.weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=self.stddev), name=self.id + "_w")
        self.bias = tf.Variable(tf.zeros([output_size]), name=self.id + "_b")

    def __call__(self, X):
        """
        computes and returns W^TX + b when the object is called

        Parameters:
        ----------
        X: Tensor
            the input vector to compute the layer output on

        Returns: Tensor
            W^TX+ b
        """

        return self.activation_fn(tf.matmul(X, self.weights) + self.bias)

    def assign_to(self, other, session):
        """
        Assigns the parameters of the current layer to an other

        Parameters:
        ----------
        other: FCLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """
        if isinstance(other, FCLayer) and self.input_size == other.input_size and self.output_size == other.output_size:
            weights_assign = self.weights.assign(other.weights)
            bias_assign = self.bias.assign(other.bias)

            session.run([weights_assign, bias_assign])
        else:
            raise TypeError("Cannot assign FCLayer: mismatch in type or size")


    def clone(self):
        """
        Clones the current layer into a new layer instance
        """

        clone = FCLayer(self.input_size, self.output_size, self.activation, self.id + '_clone')
        clone.weights = tf.Variable(self.weights.initialized_value(), name=self.id + '_clone_w')
        clone.bias = tf.Variable(self.bias.initialized_value(), name=self.id + '_clone_b')

        return clone


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """

        return [self.weights, self.bias]

class LSTMCell(object):

    # a static variable to keep count of LSTMCell created
    created_count = 0

    def __init__(self, input_size, num_hidden, minibatch_size, name='lstmcell'):
        """
        Constructs an LSTM Cell

        Parameters:
        ----------
        input_size: int
            the size of the single input vector to the cell
        num_hidden: int
            the number of hidden nodes in the cell
        minibatch_size: int
            the number of the input vectors in the input matrix
        """

        LSTMCell.created_count += 1

        self.id = name + '_' + str(LSTMCell.created_count) if name == 'lstmcell' else name

        self.input_size = input_size
        self.num_hidden = num_hidden
        self.minibatch_size = minibatch_size

        self.input_weights = tf.Variable(tf.truncated_normal([self.input_size, self.num_hidden * 4], -0.1, 0.1), name=self.id + '_wi')
        self.output_weights = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_hidden * 4], -0.1, 0.1), name=self.id + '_wo')
        self.bias = tf.Variable(tf.zeros([self.num_hidden * 4]), name=self.id + '_b')

        self.prev_output = tf.Variable(tf.zeros([self.minibatch_size, self.num_hidden]), trainable=False, name=self.id+'_o')
        self.prev_state = tf.Variable(tf.zeros([self.minibatch_size, self.num_hidden]), trainable=False, name=self.id+'_s')

    def __call__(self, X):
        """
        Performs the LSTM's forget, input and output operations
        according to: http://arxiv.org/pdf/1402.1128v1.pdf without peepholes

        Parameters:
        ----------
        X: list[Tensor]
            The input list to process by the LSTM
        """
        outputs = tf.TensorArray(tf.float32, len(X))
        inputs = tf.TensorArray(tf.float32, len(X))
        t = tf.constant(0, dtype=tf.int32)

        for i, step_input in enumerate(X):
            inputs = inputs.write(i, step_input)

        def step_op(time, prev_state, prev_output, inputs_list, outputs_list):
            time_step = inputs_list.read(time)
            gates = tf.matmul(time_step, self.input_weights) + tf.matmul(prev_output, self.output_weights) + self.bias
            gates = tf.reshape(gates, [-1, self.num_hidden, 4])

            input_gate = tf.sigmoid(gates[:, :, 0])
            forget_gate = tf.sigmoid(gates[:, :, 1])
            candidate_state = tf.tanh(gates[:, :, 2])
            output_gate = tf.sigmoid(gates[:, :, 3])

            state = forget_gate * prev_state + input_gate * candidate_state
            output = output_gate * tf.tanh(state)
            new_outputs = outputs_list.write(time, output)

            return time + 1, state, output, inputs_list, new_outputs

        _, state, output, _, final_outputs = tf.while_loop(
            cond=lambda time, *_: time < len(X),
            body= step_op,
            loop_vars=(t, self.prev_state, self.prev_output, inputs, outputs),
            parallel_iterations=32,
            swap_memory=True
        )

        self.prev_state.assign(state)
        self.prev_output.assign(output)

        return [final_outputs.read(t) for t in range(len(X))]

    def assign_to(self, other, session):
        """
        Assigns the parameters of the cuurrent cell to another's

        Parameters:
        ----------
        other: LSTMCell
            The cell to darw the parameters from
        session: tf.Session
            The tensorflow session to run the assignments
        """
        shape_set = set([self.input_size, self.num_hidden, self.minibatch_size])
        other_shape_set = set([other.input_size, other.num_hidden, other.minibatch_size])

        if isinstance(other, LSTMCell) and shape_set == other_shape_set:
            input_weights_assign = self.input_weights.assign(other.input_weights)
            output_weights_assign = self.output_weights.assign(other.output_weights)
            bias_assign = self.bias.assign(other.bias)
            prev_state_assign = self.prev_state.assign(other.prev_state)
            prev_output_assign = self.prev_output.assign(other.prev_output)

            session.run([input_weights_assign, output_weights_assign, bias_assign, prev_state_assign, prev_output_assign])
        else:
            raise TypeError("Cannot assign an LSTMCell: type or size mismatch")

    def clone(self):
        """
        Clones the current cell to another LSTMCell instance
        """

        clone = LSTMCell(self.input_size, self.num_hidden, self.minibatch_size, self.id + '_clone')
        clone.input_weights = tf.Variable(self.input_weights.initialized_value(), name=self.id + '_clone_wi')
        clone.output_weights = tf.Variable(self.output_weights.initialized_value(), name=self.id + '_clone_wo')
        clone.bias = tf.Variable(self.bias.initialized_value(), name=self.id + '_clone_b')
        clone.prev_state = tf.Variable(self.prev_state.initialized_value(), trainable=False, name=self.id + '_clone_s')
        clone.prev_output = tf.Variable(self.prev_output.initialized_value(), trainable=False, name=self.id + '_clone_o')

        return clone

    def clear(self, session):
        """
        clears the hidden state of the LSTM
        """
        zero_state = self.prev_state.assign(np.zeros((self.minibatch_size, self.num_hidden), dtype=np.float32))
        zero_output = self.prev_output.assign(np.zeros((self.minibatch_size, self.num_hidden), dtype=np.float32))

        session.run([zero_state, zero_output])

    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """

        return [
            self.input_weights,
            self.output_weights,
            self.bias,
            self.prev_state,
            self.prev_output
        ]

def print_debug(name, value):
    print("[debug] {}={}".format(name,value))