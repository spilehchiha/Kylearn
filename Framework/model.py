from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

class Model():
    def __init__(self, x_shape, ckpt_path, tsboard_path):

        self.checkpoint_path = ckpt_path
        self.tensorboard_path = tsboard_path

        with tf.name_scope('inputs'):
            self.features = tf.placeholder(dtype=tf.float32, shape=x_shape, name= 'features')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.graph = tf.Graph()
        self.session = None
        with self.graph.as_default():
            self.step = tf.train.get_or_create_global_step()
            tf.add_to_collection('global_variables', self.step)


    def initialize_variables(self):
        with tf.get_collection("global_variables"):
            pass

    def load_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def test(self):
        pass


