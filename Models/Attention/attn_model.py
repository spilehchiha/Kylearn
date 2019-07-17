import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
import collections


class Attn_model_1d(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num, dev_num,
                 batch_size, lr, regression=False, threshold=0.99):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='features')
            self.input_dev = tf.placeholder(tf.float32, [None, dev_num], name='dev_type')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='alarm')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')


        with tf.variable_scope('scaling_attention'):
            # initialize weight to all one and not using bias, this case the values in the
            # attention matrix that doesn't contribute to the classification will be 1.
            # That means not updated since the value of the corresponding PM value is always 0.
            w1 = tf.get_variable('attn_w', [dev_num, feature_num], trainable=True, initializer=tf.initializers.ones)
            # if use more than one attentions, use tf.scan
            attn_1 = tf.matmul(self.input_dev, w1)
            # attn_1 = tf.layers.batch_normalization(inputs=attn_1, training=self.is_training, momentum=0.999)
            self.scaling_attention = tf.nn.relu(attn_1)
            attn_1 = tf.expand_dims(self.scaling_attention, axis=-1)
            scaled_input_x = tf.multiply(self.input_x, attn_1)

        if regression:
            self.bias_attention = tf.constant(0, dtype=tf.float32)
        else:
            with tf.variable_scope('bias_attention'):
                w2 = tf.get_variable('attn_w', [dev_num, num_classes], trainable=True, initializer=initializer)
                self.bias_attention = tf.matmul(self.input_dev, w2)
                self.bias_attention = tf.layers.batch_normalization(inputs=self.bias_attention)
                self.bias_attention = tf.nn.relu(self.bias_attention)
                # better for visualization
                self.attn2 = tf.nn.sigmoid(self.bias_attention)


        with tf.variable_scope('logits'):

            '''is_training: Whether to return the output in training mode 
            (normalized with statistics of the current batch) or in inference mode 
            (normalized with moving statistics)'''
            '''In another word, training = True when the batch is large enough, 
            and False if the batch is small'''

            net = self.classifier(network, scaled_input_x, num_classes=num_classes,
                                  is_training=self.is_training)
            self.logits = net + self.bias_attention
            # self.logits = tf.multiply(net, self.bias_attention)
            self.logits = tf.layers.batch_normalization(inputs=self.logits)

        if regression:
            with tf.variable_scope('error'):
                self.proba = tf.nn.sigmoid(self.logits)
                self.error = self.proba - self.input_y
                self.loss = tf.reduce_mean(tf.square(self.error))
                threshold = tf.constant(threshold)
                condition = tf.greater_equal(self.proba, threshold)
                self.prediction = tf.where(condition, tf.ones_like(self.proba), tf.zeros_like(self.proba), name='prediction')
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.input_y), tf.float32))
        else:
            with tf.variable_scope('error'):
                self.proba = tf.nn.sigmoid(self.logits)
                self.loss = tf.reduce_mean(
                    # use `sparse`, no need to one-hot the `input_y`
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.input_y, logits = self.logits))
                self.prediction = tf.nn.softmax(self.logits)
                self.prediction = tf.argmax(self.prediction, 1)
                self.real = tf.argmax(self.input_y, 1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.real), tf.float32))

        self.best_loss = 100000

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes, is_training):

        return network.network(inputs=input, num_classes=num_classes, is_training=is_training)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=round(
                                                           dataset.train_set.shape[0] / self.batch_size),
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            accuracy, loss, _ = self.run([self.accuracy, self.loss, self.train_op],
                                        feed_dict={self.input_x: batch['x'],
                                                   self.input_dev: batch['dev'],
                                                   self.input_y: batch['y'],
                                                   self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss= ' + str(loss) + '    train_acc= '+str(accuracy)+'          round' + str(step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.val_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, val_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                 feed_dict={self.input_x: val_data['x'],
                                            self.input_dev: val_data['dev'],
                                            self.input_y: val_data['y'],
                                            self.is_training: True})
        print('val_loss= ' + str(loss) + '    val_acc= '+str(accuracy)+'          round: ' + str(step))
        '''early stoping'''
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == 10:
            stop_training = True
        else:
            stop_training = False

        return stop_training


    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return prediction

    def get_accuracy(self, data, is_training=False):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.input_y: data.test_set['y'],
            self.is_training: is_training
        })
        return accuracy

    def get_logits(self, data, is_training=False):
        logits = self.run([self.logits], feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return logits

    def get_proba(self, data, is_training=False):
        proba = self.run(self.proba, feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return proba

    def get_attn_matrix(self, onehot_dev):
        scaling_attention, bias_attention = self.run([self.scaling_attention, self.bias_attention],
                                                     feed_dict={self.input_dev: onehot_dev,
                                                                self.is_training: False})
        return scaling_attention, bias_attention

class Attn_model_2d(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num, dev_num,
                 batch_size, lr, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='features')
            self.input_dev = tf.placeholder(tf.float32, [None, dev_num], name='dev_type')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='alarm')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        with tf.variable_scope('scaling_attention'):
            # initialize weight to all one and not using bias, this case the values in the
            # attention matrix that doesn't contribute to the classification will be 1.
            # That means not updated since the value of the corresponding PM value is always 0.

            # ----------------------------------------
            w1 = tf.get_variable('attn_w', [dev_num, input_shape[0], feature_num], trainable=True, initializer=tf.initializers.ones)
            # tf.einsum (Einstein summation)
            # [?, 11] * [11, 3, 45] -> [?, 3, 45]
            attn_1 = tf.einsum('ni,ijk->njk', self.input_dev, w1)
            self.scaling_attention = tf.nn.relu(attn_1)
            # expand attention matrix to a 4-D tensor to match the input_x
            # ------------------------------------------
            #
            # w1 = tf.get_variable('attn_w', [dev_num, input_shape[0]*feature_num], trainable=True, initializer=tf.initializers.ones)
            # attn_1 = tf.matmul(self.input_dev, w1)
            # self.scaling_attention = tf.reshape(attn_1, shape=[-1, 3, 45])
            #
            # # ------------------------------------------

            attn_1 = tf.expand_dims(self.scaling_attention, axis=-1)
            # Dot product to scale the input
            scaled_input_x = tf.multiply(self.input_x, attn_1)
            # scaled_input_x = tf.layers.batch_normalization(inputs=scaled_input_x, training=self.is_training, momentum=0.999)
            # scaled_input_x = tf.nn.relu(scaled_input_x)

        if regression:
            self.bias_attention = tf.constant(0, dtype=tf.float32)
        else:
            with tf.variable_scope('bias_attention'):
                w2 = tf.get_variable('attn_w', [dev_num, num_classes], trainable=True, initializer=initializer)
                self.bias_attention = tf.matmul(self.input_dev, w2)
                self.bias_attention = tf.layers.batch_normalization(inputs=self.bias_attention)
                self.bias_attention = tf.nn.relu(self.bias_attention)
                # better for visualization
                self.attn2 = tf.nn.sigmoid(self.bias_attention)


        with tf.variable_scope('logits'):

            '''is_training: Whether to return the output in training mode 
            (normalized with statistics of the current batch) or in inference mode 
            (normalized with moving statistics)'''
            '''In another word, training = True when the batch is large enough, 
            and False if the batch is small'''

            net = self.classifier(network, scaled_input_x, num_classes=num_classes,
                                  is_training=self.is_training)
            self.logits = net + self.bias_attention
            # self.logits = tf.multiply(net, self.bias_attention)
            self.logits = tf.layers.batch_normalization(inputs=self.logits)

        if regression:
            with tf.variable_scope('error'):
                self.proba = tf.nn.sigmoid(self.logits)
                self.error = self.proba - self.input_y
                self.loss = tf.reduce_mean(tf.square(self.error))
                threshold = tf.constant(threshold)
                condition = tf.greater_equal(self.proba, threshold)
                self.prediction = tf.where(condition, tf.ones_like(self.proba), tf.zeros_like(self.proba), name='prediction')
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.input_y), tf.float32))
        else:
            with tf.variable_scope('error'):
                self.proba = tf.nn.sigmoid(self.logits)
                self.loss = tf.reduce_mean(
                    # use `sparse`, no need to one-hot the `input_y`
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.input_y, logits = self.logits))
                self.prediction = tf.nn.softmax(self.logits)
                self.prediction = tf.argmax(self.prediction, 1)
                self.real = tf.argmax(self.input_y, 1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.real), tf.float32))

        self.best_loss = 100000

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=self.patience_max)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes, is_training):

        return network.network(inputs=input, num_classes=num_classes, is_training=is_training)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=round(
                                                           dataset.train_set.shape[0] / self.batch_size),
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            accuracy, loss, _ = self.run([self.accuracy, self.loss, self.train_op],
                                        feed_dict={self.input_x: batch['x'],
                                                   self.input_dev: batch['dev'],
                                                   self.input_y: batch['y'],
                                                   self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss= ' + str(loss) + '    train_acc= '+str(accuracy)+'          round' + str(step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.val_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, val_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                 feed_dict={self.input_x: val_data['x'],
                                            self.input_dev: val_data['dev'],
                                            self.input_y: val_data['y'],
                                            self.is_training: True})
        print('val_loss= ' + str(loss) + '    val_acc= '+str(accuracy)+'          round: ' + str(step))
        '''early stoping'''
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == self.patience_max:
            stop_training = True
        else:
            stop_training = False

        return stop_training


    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return prediction

    def get_accuracy(self, data, is_training=False):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.input_y: data.test_set['y'],
            self.is_training: is_training
        })
        return accuracy

    def get_logits(self, data, is_training=False):
        logits = self.run([self.logits], feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return logits

    def get_proba(self, data, is_training=False):
        proba = self.run(self.proba, feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return proba

    def get_attn_matrix(self, onehot_dev):
        scaling_attention, bias_attention = self.run([self.scaling_attention, self.bias_attention],
                                                     feed_dict={self.input_dev: onehot_dev,
                                                                self.is_training: False})
        return scaling_attention, bias_attention

class Attn_model_2d_timeSeries(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num, dev_num,
                 batch_size, lr, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='features')
            self.input_dev = tf.placeholder(tf.float32, [None, dev_num], name='dev_type')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='alarm')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        with tf.variable_scope('scaling_attention'):
            # initialize weight to all one and not using bias, this case the values in the
            # attention matrix that doesn't contribute to the classification will be 1.
            # That means not updated since the value of the corresponding PM value is always 0.
            w1 = tf.get_variable('attn_w', [dev_num, feature_num], trainable=True, initializer=tf.initializers.ones)
            # tf.einsum (Einstein summation)
            # [?, 11] * [11, 45] -> [?, 45]
            attn_1 = tf.einsum('ni,ik->nk', self.input_dev, w1)
            # attn_1 = tf.layers.batch_normalization(inputs=attn_1, training=self.is_training, momentum=0.999)
            self.scaling_attention = tf.nn.relu(attn_1)
            # expand attention matrix to a 4-D tensor to match the input_x

        with tf.variable_scope('time_attention'):
            w2 = tf.get_variable('attn_t', [dev_num, input_shape[0]], trainable=True, initializer=tf.initializers.ones)
            # [?, 11] * [11, 3] -> [?, 3]
            attn_2 = tf.einsum('ni,ij->nj', self.input_dev, w2)
            self.time_attention = tf.nn.relu(attn_2)

        # Outer product
        # [?, 3] * [?, 45] -> [?, 3, 45]
        self.input_attention = tf.einsum('nj,nk->njk', self.time_attention, self.scaling_attention)
        # [?, 3, 45] -> [?, 3, 45, 1]
        self.input_attention = tf.expand_dims(self.input_attention, axis = -1)

        # Dot product to scale the input
        # [?, 3, 45, 1] X [?, 3, 45, 1]
        scaled_input_x = tf.multiply(self.input_attention, self.input_x)

        if regression:
            self.bias_attention = tf.constant(0, dtype=tf.float32)
        else:
            with tf.variable_scope('bias_attention'):
                w2 = tf.get_variable('attn_w', [dev_num, num_classes], trainable=True, initializer=initializer)
                self.bias_attention = tf.matmul(self.input_dev, w2)
                self.bias_attention = tf.layers.batch_normalization(inputs=self.bias_attention)
                self.bias_attention = tf.nn.relu(self.bias_attention)
                # better for visualization
                self.attn2 = tf.nn.sigmoid(self.bias_attention)


        with tf.variable_scope('logits'):

            '''is_training: Whether to return the output in training mode 
            (normalized with statistics of the current batch) or in inference mode 
            (normalized with moving statistics)'''
            '''In another word, training = True when the batch is large enough, 
            and False if the batch is small'''

            net = self.classifier(network, scaled_input_x, num_classes=num_classes,
                                  is_training=self.is_training)
            self.logits = net + self.bias_attention
            # self.logits = tf.multiply(net, self.bias_attention)
            self.logits = tf.layers.batch_normalization(inputs=self.logits)

        if regression:
            with tf.variable_scope('error'):
                self.proba = tf.nn.sigmoid(self.logits)
                self.error = self.proba - self.input_y
                self.loss = tf.reduce_mean(tf.square(self.error))
                threshold = tf.constant(threshold)
                condition = tf.greater_equal(self.proba, threshold)
                self.prediction = tf.where(condition, tf.ones_like(self.proba), tf.zeros_like(self.proba), name='prediction')
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.input_y), tf.float32))
        else:
            with tf.variable_scope('error'):
                self.proba = tf.nn.sigmoid(self.logits)
                self.loss = tf.reduce_mean(
                    # use `sparse`, no need to one-hot the `input_y`
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.input_y, logits = self.logits))
                self.prediction = tf.nn.softmax(self.logits)
                self.prediction = tf.argmax(self.prediction, 1)
                self.real = tf.argmax(self.input_y, 1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.real), tf.float32))

        self.best_loss = 100000

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=self.patience_max)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes, is_training):

        return network.network(inputs=input, num_classes=num_classes, is_training=is_training)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=round(
                                                           dataset.train_set.shape[0] / self.batch_size),
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            accuracy, loss, _ = self.run([self.accuracy, self.loss, self.train_op],
                                        feed_dict={self.input_x: batch['x'],
                                                   self.input_dev: batch['dev'],
                                                   self.input_y: batch['y'],
                                                   self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss= ' + str(loss) + '    train_acc= '+str(accuracy)+'          round' + str(step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.val_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, val_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                 feed_dict={self.input_x: val_data['x'],
                                            self.input_dev: val_data['dev'],
                                            self.input_y: val_data['y'],
                                            self.is_training: True})
        print('val_loss= ' + str(loss) + '    val_acc= '+str(accuracy)+'          round: ' + str(step))
        '''early stoping'''
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == self.patience_max:
            stop_training = True
        else:
            stop_training = False

        return stop_training


    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return prediction

    def get_accuracy(self, data, is_training=False):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.input_y: data.test_set['y'],
            self.is_training: is_training
        })
        return accuracy

    def get_logits(self, data, is_training=False):
        logits = self.run([self.logits], feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return logits

    def get_proba(self, data, is_training=False):
        proba = self.run(self.proba, feed_dict={
            self.input_x: data.test_set['x'],
            self.input_dev: data.test_set['dev'],
            self.is_training: is_training
        })
        return proba

    def get_attn_matrix(self, onehot_dev):
        scaling_attention, time_attention, input_attention = self.run(
            [self.scaling_attention, self.time_attention, self.input_attention],
            feed_dict={self.input_dev: onehot_dev, self.is_training: False})
        return scaling_attention, time_attention, input_attention



