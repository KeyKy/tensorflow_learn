
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import cifar10

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

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            logit = sess.run([logits]) 
            print(logit[0][0,:]) # logit is a list
        


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()


















