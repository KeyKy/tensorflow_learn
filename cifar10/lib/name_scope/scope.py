from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import sco
def fun():
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='weights')
    print(weights.name)

with tf.name_scope('my_scope'):
    sco.fun2()



