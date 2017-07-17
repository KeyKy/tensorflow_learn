import tensorflow as tf
import numpy as np

image_ph = tf.placeholder()
transformed_image = tf.map_fn(lambda img: img, image_ph)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options))

out = sess.run(transformed_image, feed_dict={transformed_image: [np.zeros(5,5)]*5})

import ipdb; ipdb.set_trace()
asd = 0
