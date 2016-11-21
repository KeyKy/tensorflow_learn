from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os, sys
from six.moves import urllib
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import time
from datetime import datetime
import svhn

TRAIN_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
TEST_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
EXTRA_URL = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/tmp/svhn',
                           """svhn download directory.""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/svhn_log',
                           """svhn train directory.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                           """batch size.""")
tf.app.flags.DEFINE_integer('num_epochs', 2,
                           """number of epochs.""")
tf.app.flags.DEFINE_integer('max_step', 20000,
                           """max step of iteration""")

def _download(dest_directory, url):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not tf.gfile.Exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes. ')
    else:
        print(filepath, ' Existed.')


def maybe_download():
    dest_directory = FLAGS.data_dir
    if not tf.gfile.Exists(dest_directory):
        tf.gfile.MakeDirs(dest_directory)
    _download(dest_directory, TRAIN_URL)
    _download(dest_directory, TEST_URL)
    _download(dest_directory, EXTRA_URL)

class DataSet(object):
    def __init__(self, images, labels, dtype):
        assert images.shape[0] == labels.shape[0], (
            'iamges.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        self._num_examples = images.shape[0]
        if dtype == dtypes.float32:
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._cur = 0
        self.shuffle()

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels

    def shuffle(self):
        self._perm = np.random.permutation(np.arange(self._num_examples))

    def next_batch(self, batch_size):
        if self._cur + batch_size > self._num_examples:
            ds_inds1 = self._perm[self._cur:]
            self.shuffle()
            num_pads = self._cur + batch_size - self._num_examples
            ds_inds2 = self._perm[:num_pads]
            self._cur = num_pads
            ds_inds = np.append(ds_inds1, ds_inds2)
        else:
            ds_inds = self._perm[self._cur:self._cur+batch_size]
        return self._images[ds_inds], self._labels[ds_inds]

def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    labels_one_hot = np.zeros([num_labels, num_classes])
    zero_ind = np.where(labels == 10)[0]
    labels[zero_ind] = 0
    for i in range(num_labels):
        labels_one_hot[i,labels[i,0]] = 1
    return labels_one_hot

def extract_images_labels(f, one_hot=False, num_classes=10):
    images = f['X'].transpose(3, 0, 1, 2)
    labels = f['y'].squeeze()
    zeros_label_inds = np.where(labels == 10)[0]
    labels[zeros_label_inds] = 0
    if one_hot:
        return images, dense_to_one_hot(labels, num_classes)
    return images, labels

def read_data_sets(one_hot, dtype=dtypes.float32, num_classes=10):
    print('read_data_sets')
    train_filepath = os.path.join(FLAGS.data_dir, 'train_32x32.mat')
    train_mat = scipy.io.loadmat(train_filepath)
    print('reading test')
    test_filepath = os.path.join(FLAGS.data_dir, 'test_32x32.mat')
    test_mat = scipy.io.loadmat(test_filepath)
    print('read done.')
    #extra_filepath = os.path.join(FLAGS.data_dir, 'extra_32x32.mat')
    #extra_mat = scipy.io.loadmat(extra_filepath)
    print('extracting image and label')
    train_images, train_labels = extract_images_labels(train_mat, one_hot, num_classes)
    test_images, test_labels = extract_images_labels(test_mat, one_hot, num_classes)
    print('extract done.')
    #extra_images, extra_labels = extract_images_labels(extra_mat, one_hot, num_classes)
    train = DataSet(train_images, train_labels, dtype=dtype)
    #train = DataSet(test_images, test_labels, dtype=dtype)
    #validation = DataSet(extra_images, extra_labels, dtype=dtype)
    #validation = DataSet(test_images.copy(), test_labels.copy(), dtype=dtype)
    test = DataSet(test_images, test_labels, dtype=dtype)

    #return base.Datasets(train=train, validation=validation, test=test)
    return base.Datasets(train=train, validation=train, test=test)

def fill_feed_dict(data_sets, images_pl, labels_pl):
    images_feed, labels_feed = data_sets.next_batch(FLAGS.batch_size)
    
    feed_dict = {images_pl:images_feed, labels_pl:labels_feed}
    return feed_dict

def train():
    data_sets = read_data_sets(one_hot=False)
    print('read_data_sets done.')
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        image_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 32, 32, 3))
        label_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size))
        distorted_images = tf.map_fn(lambda img: tf.random_crop(img, [24,24,3]), image_placeholder)
        distorted_images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), distorted_images)
        distorted_images = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=63), distorted_images)
        distorted_images = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), distorted_images)
        float_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), distorted_images)
        
        logits = svhn.inference(float_images)
        loss = svhn.loss(logits, label_placeholder)
        train_op = svhn.train(loss, global_step)
        init_op = tf.global_variables_initializer()
        summary = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options
            ))
        sess.run(init_op)

        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        for step in range(FLAGS.max_step):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, image_placeholder, label_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 10 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
        sess.close()

def main(_):
    maybe_download()
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()








