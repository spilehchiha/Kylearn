import numpy as np
import tensorflow as tf
from utils import utils
from Models.Mean_Teacher.mt_utils import getter_ema, model_vars
from framework.model import Model
import collections
import functools
from utils.log import log_down


class Mean_Teacher_model_1d(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num, dev_num,
                 batch_size, lr, max_step=100000,
                 wd=0.02, ema_decay=0.999, warmup_pos=0.4, coefficient=50, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.logger = log_down('train_log')
        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.max_step = max_step
        initializer = tf.contrib.layers.variance_scaling_initializer()

        wd *= lr
        self.warmup = tf.clip_by_value(tf.to_float(self.global_step * batch_size) / (warmup_pos * (2469001 * 1024)), 0, 1)

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='features')
            self.input_dev = tf.placeholder(tf.float32, [None,], name='dev_type')
            self.input_y = tf.placeholder(tf.float32, [None,], name='alarm')
            self.unlabel_x = tf.placeholder(tf.float32, [None] + input_shape, name='unlabeled_features')
            self.unlabel_dev = tf.placeholder(tf.float32, [None,], name='unlabeled_dev_type')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')


        with tf.variable_scope('scaling_attention'):
            input_attn = tf.get_variable('input_attention', [dev_num, feature_num], trainable=True, initializer=tf.initializers.ones)
            self.input_attention = tf.nn.leaky_relu(input_attn)
            input_matrices = tf.gather(self.input_attention, tf.cast(self.input_dev, tf.int32))
            input_matrices = tf.expand_dims(input_matrices, axis=-1)
            input_matrices_un = tf.gather(self.input_attention, tf.cast(self.unlabel_dev, tf.int32))
            input_matrices_un = tf.expand_dims(input_matrices_un, axis=-1)

        with tf.variable_scope('bias_attention'):
            output_attn = tf.get_variable('output_attention', [dev_num, num_classes], trainable=True, initializer=tf.initializers.zeros)
            self.output_attention = tf.nn.leaky_relu(output_attn)
            output_bias = tf.gather(self.output_attention, tf.cast(self.input_dev, tf.int32))


        # apply attention matrix
        scaled_input_x = tf.multiply(self.input_x, input_matrices)
        scaled_unlabeled_x = tf.multiply(self.unlabel_x, input_matrices_un)
        # #not applying attention matrix
        # scaled_input_x = self.input_x
        # scaled_unlabeled_x = self.unlabel_x


        # Mean-Teacher
        # Labeled part

        logits = self.classifier(network, scaled_input_x, num_classes=num_classes,
                              is_training=self.is_training)
        self.logits = logits + output_bias

        # Update BN before here
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # EMA
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        ema_op = ema.apply(model_vars())
        ema_getter = functools.partial(getter_ema, ema)

        # Unlabeled part: Consistency Loss
        logits_t = self.classifier(network, scaled_unlabeled_x, num_classes=num_classes,
                              is_training=self.is_training, getter=ema_getter)
        logits_teacher = tf.stop_gradient(logits_t)
        logits_student = self.classifier(network, scaled_unlabeled_x, num_classes=num_classes,
                              is_training=self.is_training) # This part better use augmented `unlabeled_x`
        loss_consistency = tf.reduce_mean((tf.nn.softmax(logits_teacher) - tf.nn.softmax(logits_student)) ** 2, -1)
        self.loss_c = tf.reduce_mean(loss_consistency)

        onehot_y = tf.cast(self.input_y, tf.int32)
        self.onehot_y = tf.one_hot(onehot_y, num_classes)

        with tf.variable_scope('error'):
            self.proba = tf.nn.sigmoid(self.logits)
            self.loss = tf.reduce_mean(
                # use `sparse`, no need to one-hot the `input_y`
                tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.onehot_y, logits = self.logits))
            self.prediction = tf.nn.softmax(self.logits)
            self.prediction = tf.argmax(self.prediction, 1)
            self.real = tf.argmax(self.onehot_y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.real), tf.float32))

        tf.summary.scalar('losses/xe', self.loss)
        tf.summary.scalar('losses/mt', self.loss_c)
        self.best_loss = 100000

        # operations
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in model_vars('logits') if 'kernel' in v.name])
        train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.loss_c * self.warmup * coefficient,
                                                       colocate_gradients_with_ops=True,
                                                       global_step=self.global_step)
        with tf.control_dependencies([train_op]):
            self.train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.classifier(network, scaled_input_x, num_classes=num_classes,
                              is_training=True)
        self.train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])


        self.saver = tf.train.Saver(max_to_keep=self.patience_max)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes, is_training, name='logits', getter=None):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE, custom_getter=getter):
            net = network.network(inputs=input, num_classes=num_classes, is_training=is_training)

        return net

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=100,
                                                       max_step=self.max_step)  # batch*evaluation_span = dataset size = one epoch

        for labeled, unlabeled in dataset.training_generator(batch_size=self.batch_size, portion= 0.5):

            accuracy, loss, lossc, wm, _ = self.run([self.accuracy, self.loss,
                                                     self.loss_c, self.warmup,
                                                 self.train_op],
                                        feed_dict={self.input_x: labeled['x'],
                                                   self.input_dev: labeled['dev'],
                                                   self.input_y: labeled['y'],
                                                   self.unlabel_x: unlabeled['x'],
                                                   self.unlabel_dev: unlabeled['dev'],
                                                   self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss = ' + str(loss) +
                      '     consis_loss= '+str(lossc) +
                      '     train_acc= '+str(accuracy) +
                      '     step ' + str(step_control['step']) +
                      '     warmup' + str(wm))
            if step_control['time_to_stop']:
                self.save_checkpoint()
                break
            if step_control['time_to_evaluate']:
                self.evaluate(dataset.val_set)
                self.save_checkpoint()


    def evaluate(self, val_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                 feed_dict={self.input_x: val_data['x'],
                                            self.input_dev: val_data['dev'],
                                            self.input_y: val_data['y'],
                                            self.is_training: False})
        self.logger.info('val_loss= ' + str(loss) + '    val_acc= '+str(accuracy)+'          round: ' + str(step))

    def tuning(self, real_data):
        _, loss, accuracy = self.run([self.train_bn, self.loss, self.accuracy],
                                     feed_dict={self.input_x: real_data['x'],
                                                self.input_dev: real_data['dev'],
                                                self.input_y: real_data['y'],
                                                self.is_training: False})

    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.is_training: is_training
        })
        return prediction

    def get_accuracy(self, data, is_training=False):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.input_y: data['y'],
            self.is_training: is_training
        })
        return accuracy

    def get_logits(self, data, is_training=False):
        logits = self.run([self.logits], feed_dict={
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.is_training: is_training
        })
        return logits

    def get_proba(self, data, is_training=False):
        proba = self.run(self.proba, feed_dict={
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.is_training: is_training
        })
        return proba

    def get_attn_matrix(self):
        scaling_attention, bias_attention = self.run([self.input_attention, self.output_attention])
        return scaling_attention, bias_attention

class Mean_Teacher_model_2d(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num, dev_num,
                 batch_size, lr, max_step=100000,
                 wd=0.02, ema_decay=0.999, warmup_pos=0.4, coefficient=50, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.logger = log_down('train_log')
        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.max_step = max_step

        wd *= lr
        self.warmup = tf.clip_by_value(tf.to_float(self.global_step * batch_size) / (warmup_pos * (2469001 * 1024)), 0, 1)

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='features')
            self.input_dev = tf.placeholder(tf.float32, [None,], name='dev_type')
            self.input_y = tf.placeholder(tf.float32, [None,], name='alarm')
            self.unlabel_x = tf.placeholder(tf.float32, [None] + input_shape, name='unlabeled_features')
            self.unlabel_dev = tf.placeholder(tf.float32, [None,], name='unlabeled_dev_type')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        m = input_shape[0]

        with tf.variable_scope('scaling_attention'):
            input_attn = tf.get_variable('input_attention', [dev_num, m, feature_num], trainable=True, initializer=tf.initializers.ones)
            self.input_attention = tf.nn.leaky_relu(input_attn)
            input_matrices = tf.gather(self.input_attention, tf.cast(self.input_dev, tf.int32))
            input_matrices = tf.expand_dims(input_matrices, axis=-1)
            input_matrices_un = tf.gather(self.input_attention, tf.cast(self.unlabel_dev, tf.int32))
            input_matrices_un = tf.expand_dims(input_matrices_un, axis=-1)

        with tf.variable_scope('bias_attention'):
            output_attn = tf.get_variable('output_attention', [dev_num, num_classes], trainable=True, initializer=tf.initializers.zeros)
            self.output_attention = tf.nn.leaky_relu(output_attn)
            output_bias = tf.gather(self.output_attention, tf.cast(self.input_dev, tf.int32))


        # apply attention matrix
        scaled_input_x = tf.multiply(self.input_x, input_matrices)
        scaled_unlabeled_x = tf.multiply(self.unlabel_x, input_matrices_un)
        # #not applying attention matrix
        # scaled_input_x = self.input_x
        # scaled_unlabeled_x = self.unlabel_x


        # Mean-Teacher
        # Labeled part

        logits = self.classifier(network, scaled_input_x, num_classes=num_classes,
                              is_training=self.is_training)
        self.logits = logits + output_bias

        # Update BN before here
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # EMA
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        ema_op = ema.apply(model_vars())
        ema_getter = functools.partial(getter_ema, ema)

        # Unlabeled part: Consistency Loss
        logits_t = self.classifier(network, scaled_unlabeled_x, num_classes=num_classes,
                              is_training=self.is_training, getter=ema_getter)
        logits_teacher = tf.stop_gradient(logits_t)
        logits_student = self.classifier(network, scaled_unlabeled_x, num_classes=num_classes,
                              is_training=self.is_training) # This part better use augmented `unlabeled_x`
        loss_consistency = tf.reduce_mean((tf.nn.softmax(logits_teacher) - tf.nn.softmax(logits_student)) ** 2, -1)
        self.loss_c = tf.reduce_mean(loss_consistency)

        onehot_y = tf.cast(self.input_y, tf.int32)
        self.onehot_y = tf.one_hot(onehot_y, num_classes)

        with tf.variable_scope('error'):
            self.proba = tf.nn.sigmoid(self.logits)
            self.loss = tf.reduce_mean(
                # use `sparse`, no need to one-hot the `input_y`
                tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.onehot_y, logits = self.logits))
            self.prediction = tf.nn.softmax(self.logits)
            self.prediction = tf.argmax(self.prediction, 1)
            self.real = tf.argmax(self.onehot_y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.real), tf.float32))

        tf.summary.scalar('losses/xe', self.loss)
        tf.summary.scalar('losses/mt', self.loss_c)
        self.best_loss = 100000

        # operations
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in model_vars('logits') if 'kernel' in v.name])
        train_op = tf.train.AdadeltaOptimizer(lr).minimize(self.loss + self.loss_c * self.warmup * coefficient,
                                                       colocate_gradients_with_ops=True,
                                                       global_step=self.global_step)
        with tf.control_dependencies([train_op]):
            self.train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.classifier(network, scaled_input_x, num_classes=num_classes,
                              is_training=True)
        self.train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])


        self.saver = tf.train.Saver(max_to_keep=self.patience_max)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def classifier(self, network, input, num_classes, is_training, name='logits', getter=None):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE, custom_getter=getter):
            net = network.network(inputs=input, num_classes=num_classes, is_training=is_training)

        return net

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=100,
                                                       max_step=self.max_step)  # batch*evaluation_span = dataset size = one epoch

        for labeled, unlabeled in dataset.training_generator(batch_size=self.batch_size, portion= 0.5):

            accuracy, loss, lossc, wm, _ = self.run([self.accuracy, self.loss,
                                                     self.loss_c, self.warmup,
                                                 self.train_op],
                                        feed_dict={self.input_x: labeled['x'],
                                                   self.input_dev: labeled['dev'],
                                                   self.input_y: labeled['y'],
                                                   self.unlabel_x: unlabeled['x'],
                                                   self.unlabel_dev: unlabeled['dev'],
                                                   self.is_training: False})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss = ' + str(loss) +
                      '     consis_loss= '+str(lossc) +
                      '     train_acc= '+str(accuracy) +
                      '     step ' + str(step_control['step']) +
                      '     warmup' + str(wm))
            if step_control['time_to_stop']:
                self.save_checkpoint()
                break
            if step_control['time_to_evaluate']:
                self.evaluate(dataset.val_set)
                self.save_checkpoint()


    def evaluate(self, val_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                 feed_dict={self.input_x: val_data['x'],
                                            self.input_dev: val_data['dev'],
                                            self.input_y: val_data['y'],
                                            self.is_training: False})
        self.logger.info('val_loss= ' + str(loss) + '    val_acc= '+str(accuracy)+'          round: ' + str(step))

    def tuning(self, real_data):
        _, loss, accuracy = self.run([self.train_bn, self.loss, self.accuracy],
                                     feed_dict={self.input_x: real_data['x'],
                                                self.input_dev: real_data['dev'],
                                                self.input_y: real_data['y'],
                                                self.is_training: False})

    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.is_training: is_training
        })
        return prediction

    def get_accuracy(self, data, is_training=False):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.input_y: data['y'],
            self.is_training: is_training
        })
        return accuracy

    def get_logits(self, data, is_training=False):
        logits = self.run([self.logits], feed_dict={
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.is_training: is_training
        })
        return logits

    def get_proba(self, data, is_training=False):
        proba = self.run(self.proba, feed_dict={
            self.input_x: data['x'],
            self.input_dev: data['dev'],
            self.is_training: is_training
        })
        return proba

    def get_attn_matrix(self):
        scaling_attention, bias_attention = self.run([self.input_attention, self.output_attention])
        return scaling_attention, bias_attention