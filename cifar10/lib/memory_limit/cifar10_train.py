
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
        memlim = tf.ConfigProto()
        memlim.gpu_options.allow_growth = True

        sess = tf.Session(config=memlim)
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            if step % 5 == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()


















