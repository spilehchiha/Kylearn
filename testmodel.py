import functools
import tensorflow as tf
from Framework import utils
import logging
logging.basicConfig(level=logging.INFO, filename= 'train_log',filemode='a')
LOG = logging.getLogger('train_log')
LOG.setLevel(logging.INFO)

class Model():

    def __init__(self):
        with tf.variable_scope("input"):
            self.x_labeled = tf.placeholder(tf.float32, [None, 32, 32, 1], name='labeled')
            self.x_unlabeled = tf.placeholder(tf.float32, [None, 32, 32, 1], name='unlabeled')
            self.label = tf.placeholder(tf.float32, [None, 10], name='alarm')
            self.step = tf.train.get_or_create_global_step()
            self.batch_size = 64
            self.train_op = None
            self.update_step_op = None
            self.tune_op = None
            self.sess = tf.Session()


    def classifier(self, input,num_classes, scope='classifier', getter=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, custom_getter=getter):
            layer1 = tf.layers.conv2d(input, filters=10, kernel_size=3)
            layer2 = tf.layers.batch_normalization(tf.nn.leaky_relu(layer1))
            layer3 = tf.layers.conv2d(layer2, filters=10, kernel_size=3)
            layer4 = tf.layers.batch_normalization(tf.nn.leaky_relu(layer3))
            layer5 = tf.layers.dense(layer4, 10)

        return layer5

    def model(self, sample_num, num_classes,consistency_weight=50, lr=0.002, wd=0.02, warmup_pos=0.4, ema=0.999, decay=0.999):
        #写到init里
        wd *= lr
        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (sample_num << 10)), 0, 1)
        with tf.name_scope('student'):
            logits_x = self.classifier(self.x_labeled, num_classes)
            post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
            logits_student = self.classifier(self.x_unlabeled, num_classes)

        with tf.name_scope('teacher'):
            trainable_vars = utils.model_vars('classifier')
            ema = tf.train.ExponentialMovingAverage(ema)
            ema_op = ema.apply(trainable_vars)
            ema_getter = functools.partial(utils.getter_ema, ema)
            logits_teacher = self.classifier(self.x_unlabeled, num_classes, getter=ema_getter)
            logits_teacher = tf.stop_gradient(logits_teacher)

        loss_mt = tf.reduce_mean((tf.nn.softmax(logits_teacher) - tf.nn.softmax(logits_student)) ** 2, -1)
        loss_mt = tf.reduce_mean(loss_mt)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=logits_x)
        loss = tf.reduce_mean(loss)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classifier') if 'kernel' in v.name])
        train_op = tf.train.AdamOptimizer(lr).minimize(loss + loss_mt * warmup * consistency_weight,
                                                       colocate_gradients_with_ops=True)



        with tf.control_dependencies([train_op]):
            self.train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.classifier(self.x_labeled, num_classes)
        self.tune_op = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

    def train(self, training_batches, evaluation_batches_fn):

        train_writer = tf.summary.FileWriter('log', self.sess.graph)
        self.update_step_op = tf.assign_add(self.step, self.batch_size)

        self.best_loss = 1
        self.patience = 0

        self.sess.run(tf.global_variables_initializer())
        LOG.info("Model variables initialized")
        for batch in training_batches:
            merge, i, results, _ = self.sess.run([],
                                            feed_dict={self.x_labeled:batch['x'],
                                                       self.x_unlabeled:batch['x2'],
                                                       self.label:batch['y']})
            self.writer.add_summary(merge, i)
            step_control = self.get_training_control()
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
                if if_stop:
                    LOG.info('Stop Training')
                    break
        _ = self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        '''early stoping'''
        loss = results['eval/class_cost/1']
        if loss <= self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == 20:
            stop_training = True
        else:
            stop_training = False

        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

        return stop_training


    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)





model = Model()

