from pixrnn import *
import scipy.misc

i2s = tf.placeholder(tf.float32, [1, 4 * 4])
state = tf.placeholder(tf.float32, [1, 8])

lstm = DiagnalLSTMCell(1, 4)

newh, newstate = lstm(i2s, state)

test  = tf.get_collection('test', scope='DiagnalBiLSTMCell')

cst = tf.constant(np.ones((1, 4)), dtype = tf.float32)
out = tf.add(cst, newh)
init = tf.global_variables_initializer()
i2s_data = np.ones((1, 16))
state_data = np.ones((1, 8))

sess = tf.Session()
sess.run(init)

test, newh, newstate = sess.run([test, newh, newstate], feed_dict={i2s: i2s_data, state: state_data})

print 'test'
print test

print 'newh'
print newh.reshape(1, 4)

print 'newstate'
print newstate.reshape(2, 4)


#image = np.ones((7, 6, 3))
#image = image.reshape(1, 7, 6, 3)
#W, b = sess.run([W, b], feed_dict={a:image})
#e = e.astype(np.uint8)

#print W.reshape(3,3,3)
#print b.reshape(7,6)
#e,f = sess.run([b,c], feed_dict={a:images})
#scipy.misc.toimage(images[0], cmin=0, cmax=1.0).save("a.jpg")
#scipy.misc.toimage(e[0], cmin=0, cmax=1.0).save("b.jpg")
#scipy.misc.toimage(f[0], cmin=0, cmax=1.0).save("c.jpg")
#scipy.misc.toimage(e[0], cmin=0, cmax=1.0).save("e.jpg")

