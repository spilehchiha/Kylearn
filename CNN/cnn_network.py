from framework.network import Network
import tensorflow as tf
import collections


class Cnn_3layers(Network):
    def __init__(self):
        super().__init__()

    def network(self, inputs, num_classes, is_training=True):
        conv1 = tf.layers.conv1d(inputs, 32, 3, 1, 'same', name='conv1')
        bn1 = tf.layers.batch_normalization(conv1)
        act1 = tf.nn.leaky_relu(bn1)
        conv2 = tf.layers.conv1d(act1, 64, 3, 1, 'same', name='conv2')
        bn2 = tf.layers.batch_normalization(conv2)
        act2 = tf.nn.leaky_relu(bn2)
        conv3 = tf.layers.conv1d(act2, 128, 3, 1, 'valid', name='conv3')
        bn3 = tf.layers.batch_normalization(conv3)
        act3 = tf.nn.leaky_relu(bn3)
        net = tf.layers.flatten(act3)
        dense = tf.layers.dense(net, num_classes)
        return dense
