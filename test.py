from pixrnn import *
import scipy.misc


images = np.array([scipy.misc.imread("input.jpg")])


a = tf.placeholder(tf.float32, [1, 7, 6 ,1])

W, b = conv2d(a, 1, "A", (3 , 3), scope = 'a')
#c = conv2d(c, 1, "A", (3 , 3), scope = 'c')
#d = conv2d(d, 1, "A", (3, 3), scope = 'd')
#e  = tf.concat(3, [b, c, d])
#print e
W_value = tf.get_variable('W_value_mask', W.get_shape())
tf.assign(W_value, W)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

image = np.ones((7, 6, 1))
image = image.reshape(1, 7, 6, 1)
W, b, c = sess.run([W, b, W_value], feed_dict={a:image})
#e = e.astype(np.uint8)

print W.reshape(3,3,1)
print b.reshape(7,6)
print c.reshape(3,3,1)
#e,f = sess.run([b,c], feed_dict={a:images})
#scipy.misc.toimage(images[0], cmin=0, cmax=1.0).save("a.jpg")
#scipy.misc.toimage(e[0], cmin=0, cmax=1.0).save("b.jpg")
#scipy.misc.toimage(f[0], cmin=0, cmax=1.0).save("c.jpg")
#scipy.misc.toimage(e[0], cmin=0, cmax=1.0).save("e.jpg")

