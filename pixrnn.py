import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

def skew(inputs, scope = "skew"):
    with tf.name_scope(scope):
        _, height, _, _ = inputs.get_shape().as_list()
        return tf.pack([tf.pad(row, [[0, 0], [idx, height - 1 - idx], [0, 0]]) for idx, row in enumerate(tf.unpack(inputs, axis=1))], 1)

def unskew(inputs, scope = "unskew"):
    with tf.name_scope(scope):
        _, height, width, _ = inputs.get_shape().as_list()
        return tf.pack([tf.slice(row, [0, idx, 0], [-1, width - height + 1, -1]) for idx, row in enumerate(tf.unpack(inputs, axis=1))], 1)
