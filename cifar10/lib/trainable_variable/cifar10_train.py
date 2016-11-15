
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import cifar10
import numpy as np
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('max_steps', 1000000,
                           """Number of batches to run.""")
tf.app.flags.DEFINE_string('log_device_placement', False, 
                           """Whether to log device placement.""")

def train():
    with tf.Graph().as_default():
        images, labels = cifar10.distorted_inputs()

        logits = cifar10.inference(images)

        loss = cifar10.loss(logits, labels)
        
        global_step = tf.Variable(0, trainable=False)
        train_op = cifar10.train(loss, global_step=global_step)

        summary_op = tf.merge_all_summaries()

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        trainable_var = tf.trainable_variables()
        print([v.name for v in trainable_var])
        var = trainable_var[0]
        res = sess.run(var)
        print(type(res))

def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()


















