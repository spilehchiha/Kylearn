import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
from evaluation.metrics import confusion_matrix
import collections

class CNN_model(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes,
                 batch_size, lr, regression = False, threshold=0.99, patience = 10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        with tf.variable_scope('logits'):
            self.logits = self.classifier(network, self.input_x, num_classes=num_classes)

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
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits))
                self.prediction = tf.nn.softmax(self.logits)
                self.prediction = tf.argmax(self.prediction, 1)
                self.real = tf.argmax(self.input_y, 1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.real), tf.float32))

        self.best_loss = 1000

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=self.patience_max)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes):

        return network.network(inputs=input, num_classes=num_classes)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=round(dataset.train_set.shape[0]/self.batch_size),
                                                       max_step=10000000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            accuracy, loss, _ = self.run([self.accuracy, self.loss, self.train_op],
                                                    feed_dict={self.input_x: batch['x'],
                                                               self.input_y: batch['y']}
                                                               )
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

    def evaluate(self, eval_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                         feed_dict={self.input_x: eval_data['x'],
                                                    self.input_y: eval_data['y']
                                                    })
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

    def get_prediction(self, data):
        prediction = self.run(self.prediction, feed_dict={
            self.input_x: data['x']
        })
        return prediction

    def get_accuracy(self, data):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.input_x: data['x'],
            self.input_y: data['y']
        })
        return accuracy

    def get_logits(self, data):
        logits = self.run(self.logits, feed_dict={
            self.input_x: data['x']
        })
        return logits

    def get_proba(self, data):
        proba = self.run(self.proba, feed_dict={
            self.input_x: data['x']
        })
        return proba

