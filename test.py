from pixrnn import *
import scipy.misc


images = np.array([scipy.misc.imread("input.jpg")])

a = tf.placeholder(tf.uint8, [1,3264,4896,3])
#a = tf.Variable(images)
b = skew(a)
c = unskew(b)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


e,f = sess.run([b,c], feed_dict={a:images})

scipy.misc.toimage(images[0], cmin=0, cmax=1.0).save("a.jpg")
scipy.misc.toimage(e[0], cmin=0, cmax=1.0).save("b.jpg")
scipy.misc.toimage(f[0], cmin=0, cmax=1.0).save("c.jpg")
