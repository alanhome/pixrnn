from pixrnn import *
import scipy.misc
import os
import logging

flags = tf.app.flags
flags.DEFINE_integer("hidden_dims", 1, "dimesion of hidden states of LSTM or Conv layers")
conf = flags.FLAGS

inputs = tf.placeholder(tf.float32, [1, 7, 6 ,1])

outputs = bidiagonal_lstm(inputs, conf)


#test  = tf.get_collection('test', scope='DiagnalBiLSTMCell')


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

image = np.ones((7, 6))
image = image.reshape(1, 7, 6, 1)
output = sess.run([outputs], feed_dict={inputs : image})

print 'output'
print output[0].reshape(7, 6)




