import os, argparse, logging
import numpy as np

from dlcommon.data.tensorflow import data_iter
from dlcommon.data import augmenter, dataloader
from dlcommon.argparse import data_args, data_aug_args

import common
from datasets import persons

import tensorflow as tf
from preprocessing.inception_preprocessing import preprocess_image
from nets import nets_factory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train person exist")

    common.add_common_args(parser)
    data_args.add_data_args(parser)
    data_aug_args.add_data_aug_args(parser)

    args = parser.parse_args()
    print args
    if args.image_shape is not None:
        data_shape = [int(s) for s in args.image_shape.split(',')]
    if args.mean is not None:
        mean = np.array([float(s) for s in args.mean.split(',')])
    if args.std is not None:
        std = np.array([float(s) for s in args.std.split(',')])

    #train_tf = augmenter.Augmenter(data_shape, args.resize, args.rand_crop,
    #        args.rand_resize, args.rand_mirror, mean, std)
    trainset = persons.Persons(root=args.root, train=True, transform=None)
    trainloader = dataloader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, collate_fn=common.collate_fn)
    TFIter = data_iter.pyImageDataIter(trainloader)
    TFIter.reset()

    # pre-proc
    image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))
    transformed_image = preprocess_image(image_ph, 224, 224, is_training=True)

    # network
    image_batch = tf.placeholder(tf.float32, shape=(8, 224, 224, 3))
    label_batch = tf.placeholder(tf.int32, shape=(8,))
    network_fn = nets_factory.get_network_fn('mobilenet_v1', num_classes=3, is_training=True)
    logits, _ = network_fn(image_batch)
    tf.summary.histogram('pre_activations', logits, collections=['train'])
    final_tensor = tf.nn.softmax(logits, name='ovm_results')
    tf.summary.histogram('activations', final_tensor, collections=['train'])
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_batch, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean, collections=['train'])
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_step = optimizer.minimize(cross_entropy_mean)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options))

    # fine-tune and init
    all_vars = tf.global_variables()
    var_to_restore = [v for v in all_vars if not v.name.startswith('MobilenetV1/Logits/Conv2d_1c_1x1/')]
    saver = tf.train.Saver(var_to_restore)
    saver.restore(sess, './checkpoints/mobilenet_v1_1.0_224.ckpt')
    added_vars = [v for v in all_vars if v.name.startswith('MobilenetV1/Logits/Conv2d_1c_1x1/')]
    init_op = tf.variables_initializer(added_vars)
    sess.run(init_op)
    sess.run(added_vars)

    # summary
    s_training = tf.summary.merge_all('train')
    train_writer = tf.summary.FileWriter('./summ', sess.graph)

    count = 0
    for epoch in range(50):
        for batch_data, batch_label in TFIter:
            batch_image = []
            for i in range(len(batch_data[0])):
                image_data = batch_data[0][i]
                transformed_img = sess.run(transformed_image, feed_dict={image_ph: image_data})
                batch_image.append(transformed_img)

            batch_image = np.stack(batch_image, axis=0)
            train_summary, _ = sess.run([s_training, train_step],
                    feed_dict={image_batch: batch_image, label_batch: batch_label[0]})
            train_writer.add_summary(train_summary, count)
            count += 1
            print('runing count {}'.format(count))
        TFIter.reset()
    import ipdb; ipdb.set_trace()
    asd = 0

