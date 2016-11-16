from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import tensorflow as tf


def fun2():
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.1), name='weights2')
    print(weights.name)
    return weights
