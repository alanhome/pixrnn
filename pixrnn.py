import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

def skew(inputs, scope = 'skew'):
    with tf.name_scope(scope):
        _, height, _, _ = inputs.get_shape().as_list()
        return tf.pack([tf.pad(row, [[0, 0], [idx, height - 1 - idx], [0, 0]]) 
                        for idx, row in enumerate(tf.unpack(inputs, axis=1))], 1)

def unskew(inputs, scope = 'unskew'):
    with tf.name_scope(scope):
        _, height, width, _ = inputs.get_shape().as_list()
        return tf.pack([tf.slice(row, [0, idx, 0], [-1, width - height + 1, -1]) 
                        for idx, row in enumerate(tf.unpack(inputs, axis=1))], 1)

def get_mask(mask_type, shape):
    if mask_type is None:
        return None
    
    mask_type = mask_type.lower()
    if mask_type != 'a' and mask_type != 'b':
        return None;
    
    kernel_h, kernel_w, channel, output_channel = shape
    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, 'kernel height and width should be odd number'

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    mask = np.ones(shape, dtype=np.float32)
    mask[center_h, center_w+1 : , : , : ] = 0.
    mask[center_h+1 : , : , : , : ] = 0.
    
    if mask_type == 'a':
        mask[center_h, center_w, : , : ] = 0.

    return tf.constant(mask, dtype = tf.float32)

def conv2d(inputs,
           output_channel,
           mask_type,
           kernel_shape,
           strides = [1, 1],
           padding = 'SAME',
           activation_fn = None,
           weights_initializer = WEIGHT_INITIALIZER,
           weights_regularizer = None,
           biases_initializer = tf.zeros_initializer,
           biases_regularizer = None,
           scope = 'maskconv2d'):
    with tf.variable_scope(scope):
        batch_size, height, width, channel = inputs.get_shape().as_list()

        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        W_shape = [kernel_h, kernel_w, channel, output_channel]
        
        W = tf.get_variable('weights',
                            W_shape,
                            tf.float32,
                            weights_initializer,
                            weights_regularizer)
        
        mask = get_mask(mask_type, W_shape)
        
        if mask is not None:
            W = tf.mul(W, mask, name = 'mask_weights')

        outputs = tf.nn.conv2d(inputs,
                               W, 
                               [1, stride_h, stride_w, 1], 
                               padding = padding, 
                               name = 'conv2d_outputs')
        
        if biases_initializer is not None:
            biases = tf.get_variable('biases',
                                     [output_channel],
                                     tf.float32,
                                     biases_initializer,
                                     biases_regularizer)
            
            outputs = tf.nn.bias_add(outputs, biases, name = 'outputs_plus_b')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs, name = 'outputs_with_fn')

        return outputs

class DiagnalLSTMCell(rnn_cell.RNNCell):
    def __init__(self, hidden_dims, height):
            self._height = height
            self._hidden_dims = hidden_dims

            self._col_dims = self._hidden_dims * self._height
            self._state_size = 2 * self._col_dims
            self._output_size = self._col_dims

    def __call__(self, i2s, state, scope='DiagnalBiLSTMCell'):
        _, i2s_dims = i2s.get_shape().as_list()
        assert i2s_dims == 4 * self._col_dims, 'i2s dims is wrong'
        
        print scope
        
        c_pre, h_pre = tf.split(1, 2, state)
        
        with tf.variable_scope(scope):
            h_pre_col = tf.reshape(h_pre, [-1, self._height, 1, self._hidden_dims])
            conv_s2s = conv2d(h_pre_col, 4 * self._hidden_dims, 'B', (3, 1), scope = 's2s')
            s2s = tf.reshape(conv_s2s, [-1, 4 * self._height * self._hidden_dims])

            i, f, ci, o = tf.split(1, 4, tf.sigmoid(tf.add(i2s, s2s)))
            c = tf.add(tf.mul(ci, i), tf.mul(c_pre, f))
            h = tf.mul(o, tf.tanh(c))
            
            new_state = tf.concat(1, [c, h])            
            return h, new_state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.state_size
    
    @property
    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, stat_size], dtype)

def diagnal_lstm(inputs, conf, scope = 'diagnal_lstm'):
    with tf.variable_scope(scope):
        skewed_inputs = skew(inputs, scope = 'skew_input')
        i2s = conv2d(skewed_inputs, 4 * conf.hidden_dims, [1, 1], 'B', scope = 'i2s')
        
        column_wise_inputs = tf.transpose(i2s, [0, 2, 1, 3])
        batch, width, height, channel = get_shape(column_wise_inputs)
        rnn_inputs = tf.reshape(column_wise_inputs, [batch, width, -1])
        rnn_input_list = tf.unpack(rnn_inputs, axis=1)
        cell = DiagonalLSTMCell(conf.hidden_dims, height, channel)

        output_list, state_list = tf.nn.rnn(cell,
                                            inputs=rnn_input_list, 
                                            dtype=tf.float32)

        packed_outputs = tf.pack(output_list, 1)
        width_wise_outputs = tf.reshape(packed_outputs, [-1, width, height, conf.hidden_dims])
        skewed_outputs = tf.transpose(width_wise_outputs, [0, 2, 1, 3])
        outputs = unskew(skewed_outputs, scope = 'unskew_output')

        return outputs
