import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
import collections


class AttentionModel(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, lr, batch_size, feature_num, dev_num,
                 attn_num):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='features')
            self.input_dev = tf.placeholder(tf.float32, [None, dev_num], name='dev_type')
            self.input_y = tf.placeholder(tf.float32, [None, 12], name='alarm')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        with tf.name_scope('scaling_attention'):
            w1 = tf.get_variable('attn_w', [dev_num, feature_num], trainable=True, initializer=tf.initializers.ones)
            b1 = tf.get_variable('attn_b', [1, feature_num], trainable=True, initializer=tf.initializers.zeros)
            # if use more than one attentions, use tf.scan
            attn_1 = tf.matmul(self.input_dev, w1) + b1
            attn_1 = tf.nn.tanh(attn_1)
            attn_1 = tf.expand_dims(attn_1, axis=-1)
            mul_0 = tf.multiply(self.input_x, attn_1)

        with tf.variable_scope('bias_attention'):
            w2 = tf.get_variable('attn_w', [dev_num, num_classes], trainable=True, initializer=tf.initializers.random_uniform)
            b2 = tf.get_variable('attn_b', [1, num_classes], trainable=True, initializer=tf.initializers.random_uniform)
            attn_2 = tf.matmul(self.input_dev, w2) + b2

        with tf.variable_scope('regressor'):
            net = self.classifier(network, mul_0, num_classes=num_classes,
                                  is_training=self.is_training)
            net = attn_2 + net
            self.logits = tf.nn.sigmoid(net)
            self.error = self.logits - self.input_y
            self.loss = tf.reduce_mean(tf.square(self.error))

        self.best_loss = 1000000000

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=11)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes, is_training):

        return network.network(inputs=input, num_classes=num_classes, is_training=is_training)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset, lr):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=round(
                                                           dataset.train_set.shape[0] / self.batch_size),
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            results, loss, _ = self.run([self.logits, self.loss, self.train_op],
                                        feed_dict={self.input_x: batch['x'],
                                                   self.input_y: batch['y'],
                                                   self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss= ' + str(loss) + '          round' + str(step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.val_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, val_data):
        step, results = self.run([self.global_step, self.loss],
                                 feed_dict={self.input_x: val_data['x'],
                                            self.input_y: val_data['y'],
                                            self.is_training: False})
        print(' val_loss = ' + str(results) + '          round: ' + str(step))
        '''early stoping'''
        loss = results
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

    def plot(self, dataset, threshold=0.5):
        results = self.run([self.logits], feed_dict={
            self.input_x: dataset.test_set['x'],
            self.is_training: False
        })

        results = np.array(results).squeeze()
        print(results)
        results[results >= threshold] = 1
        results[results < threshold] = 0

        cm = cm_metrix(dataset.test_set['y'], results)

        cm_analysis(cm, ['Normal', 'malfunction'], precision=True)

    def predict_proba(self, dataset):
        results = self.run([self.logits], feed_dict={
            self.input_x: dataset.test_set['x'],
            self.is_training: False})
        print(collections.Counter(np.array(results).flatten()))

        return results
