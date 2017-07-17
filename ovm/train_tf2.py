from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import numpy as np
import ipdb
from preprocessing.inception_preprocessing import preprocess_image
from nets import nets_factory


image_ph = tf.placeholder(tf.float32, shape=(1, None, None, 3))
label_ph = tf.placeholder(tf.int32, shape=(1,))
transformed_image = tf.map_fn(lambda img: preprocess_image(img, 224, 224, is_training=True), image_ph)

network_fn = nets_factory.get_network_fn('mobilenet_v1', num_classes=3, is_training=False)
logits, _ = network_fn(transformed_image)
tf.summary.histogram('pre_activations', logits)
final_tensor = tf.nn.softmax(logits, name='ovm_results')
tf.summary.histogram('activations', final_tensor)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_ph, logits=logits)
    with tf.name_scope('total'):
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy_mean)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_step = optimizer.minimize(cross_entropy_mean)


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options))

all_vars = tf.global_variables()
var_to_restore = [v for v in all_vars if not v.name.startswith('MobilenetV1/Logits/Conv2d_1c_1x1/')]
saver = tf.train.Saver(var_to_restore)
saver.restore(sess, './checkpoints/mobilenet_v1_1.0_224.ckpt')

added_vars = [v for v in all_vars if v.name.startswith('MobilenetV1/Logits/Conv2d_1c_1x1/')]
init_op = tf.variables_initializer(added_vars)



merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)
sess.run(init_op)
sess.run(added_vars)



image_data = cv2.imread('/data_shared/datasets/ILSVRC2015/Data/CLS-LOC/val/ILSVRC2012_val_00029756.JPEG')
image_data = image_data[np.newaxis,:,:,:]
image_data = image_data[:,:,:,(2,1,0)] # BGR -> RGB
label_data = np.array([0,])



train_summary, _ = sess.run([merged, train_step], feed_dict={image_ph: image_data, label_ph: label_data})
train_writer.add_summary(train_summary, 0)
#ipdb.set_trace()
#sd = 0
