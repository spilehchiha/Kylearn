from framework.network import Network
import tensorflow as tf
import collections


class Resnet(Network):
    def __init__(self):
        super().__init__()

    def network(self, inputs, num_classes, is_training=True):

        def resnet_43(inputs,
                      num_classes,
                      global_pool=True,
                      output_stride=None):
            blocks = [
                resnet_block('block1', base_depth=32, num_units=2, stride=2),
                resnet_block('block2', base_depth=64, num_units=4, stride=2),
                resnet_block('block3', base_depth=128, num_units=4, stride=2),
                resnet_block('block4', base_depth=256, num_units=4, stride=1),
            ]
            return resnet(inputs =inputs, num_classes = num_classes, blocks = blocks,
                          global_pool=global_pool, output_stride=output_stride,
                          include_root_block=True)

        def resnet_block(scope, base_depth, num_units, stride):

            return collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])(scope, bottleneck, [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': 1
            }] * (num_units - 1) + [{
                'depth': base_depth * 4,
                'depth_bottleneck': base_depth,
                'stride': stride
            }])

        def resnet(inputs,
                   num_classes,
                   blocks,
                   global_pool=True,
                   output_stride=None,
                   include_root_block=True):
            net = inputs
            if include_root_block:
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The output_stride needs to be a multiple of 4.')
                    output_stride /= 4
                net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=1, strides=1, name='conv1', padding='same')

            net = stack_blocks_dense(net, blocks, output_stride)
            net = tf.layers.batch_normalization(inputs=net, training=is_training, momentum=0.999)
            net = tf.nn.leaky_relu(net)

            if global_pool:
                net = tf.reduce_mean(net, 1, name='pool5', keep_dims=True)


            net = tf.layers.flatten(net)
            net = tf.layers.dense(inputs=net, units=num_classes)
            logits = tf.nn.sigmoid(net)

            return logits

        def stack_blocks_dense(net, blocks, output_stride=None):
            current_stride = 1

            for block in blocks:
                with tf.variable_scope(block.scope, 'block', [net]) as sc:
                    for i, unit in enumerate(block.args):
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

                        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                            if output_stride is not None and current_stride == output_stride:
                                net = block.unit_fn(net, **dict(unit, stride=1))

                            else:
                                net = block.unit_fn(net, **unit)
                                current_stride *= unit.get('stride', 1)

            if output_stride is not None and current_stride != output_stride:
                raise ValueError('The target output_stride cannot be reached.')

            return net

        def bottleneck(inputs, depth, depth_bottleneck, stride):
            depth_in = inputs.shape.dims[-1].value
            preact = tf.layers.batch_normalization(inputs=inputs, training=is_training, momentum=0.999)
            preact = tf.nn.leaky_relu(preact)

            if depth == depth_in:
                shortcut = subsample(inputs, stride)
            else:
                shortcut = tf.layers.conv2d(inputs=preact, filters=depth, kernel_size=1, strides=stride,
                                            activation=None,
                                            name='shortcut')

            residual = tf.layers.conv2d(inputs=preact, filters=depth_bottleneck, kernel_size=1, strides=1,
                                        name='conv1')
            residual = tf.layers.conv2d(inputs=residual, filters=depth_bottleneck, kernel_size=3, strides=stride,
                                        padding='SAME', name='conv2')
            # residual = subsample(residual, factor=stride)
            residual = tf.layers.conv2d(inputs=residual, filters=depth, kernel_size=1, strides=1,
                                        activation=None,
                                        name='conv3')

            output = shortcut + residual
            return output

        def subsample(inputs, factor):
            if factor == 1:
                return inputs
            else:
                return tf.layers.max_pooling2d(inputs, 1, strides=factor)

        logits = resnet_43(inputs=inputs, num_classes=num_classes)
        return logits
