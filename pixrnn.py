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
    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width should be odd number"

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    mask = np.ones(shape, dtype=np.float32)
    mask[center_h, center_w+1 : , : , : ] = 0.
    mask[center_h+1 : , : , : , : ] = 0.
    
    if mask_type == 'a':
        mask[center_h, center_w, : , : ] = 0.

    return tf.constant(mask, dtype=tf.float32) 

def conv2d(inputs,
           output_channel,
           mask_type,
           kernel_shape,
           strides=[1,1],
           padding='SAME',
           activation_fn=None,
           weights_initializer=WEIGHT_INITIALIZER,
           weights_regularizer=None,
           biases_initializer=tf.zeros_initializer,
           biases_regularizer=None,
           scope='maskconv2d'):
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
            W = tf.multiply(W, mask, name = 'mask_weights')
        
        outputs = tf.nn.conv2d(inputs,
                               W, 
                               [1, stride_h, stride_w, 1], 
                               padding=padding, 
                               name='conv2d_outputs')
        
        if biases_initializer is not None:
            biases = tf.get_variable('biases',
                                     [output_channel],
                                     tf.float32,
                                     biases_initializer,
                                     biases_regularizer)
            
            outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')
        
        
        if activation_fn is not None:
            outputs = activation_fn(outputs, name='outputs_with_fn')

        return W, outputs
