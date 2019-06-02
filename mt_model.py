from Framework.model import Model
import tensorflow as tf
class Mt_model(Model):

    def __init__(self, Network, ckpt_path, tsboard_path, x_shape, num_classes):
        super().__init__(Network, ckpt_path, tsboard_path)

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
        self.num_classes = num_classes
    def initialize_variables(self):
        # with tf.get_collection("global_variables"):
        pass

    def loss(self):
        logits_labeled = self.network(input=self.features,
                                      num_classes= self.num_classes,
                                      reuse = True,
                                      scope='res_43',
                                      is_training=True)

    def load_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def test(self):
        pass