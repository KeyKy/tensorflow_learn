from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tarfile
import tensorflow as tf
import os
import sys
import hashlib

from six.moves import urllib

import cifar10_input

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
MD5SUM = 'c32a1d4ab5d03f1284b67883e8d87530'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 128, 
                            """Number of images to process in a batch.""")

def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    cifar_filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, cifar_filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes. ')
    else:
        md5sum = hashlib.md5(open(filepath, 'r').read()).hexdigest()
        assert(md5sum == MD5SUM), "md5 check filed"
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    return images, labels



