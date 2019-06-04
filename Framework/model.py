from absl import flags
import tensorflow as tf
from abc import abstractmethod
FLAGS = flags.FLAGS

class Model():
    def __init__(self, ckpt_path, tsboard_path, **kwargs):
        self.checkpoint_path = ckpt_path
        self.tensorboard_path = tsboard_path
        self.session = tf.Session()
        self.global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver(max_to_keep=11)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)


    @abstractmethod
    def classifier(self, **kwargs):
        pass

    @abstractmethod
    def initialize_variables(self, **kwargs):
        # with tf.get_collection("global_variables"):
        pass

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self, **kwargs):
        pass

    def save_tensorboard_graph(self):
        self.writer.add_graph(self.session.graph)
        return self.writer.get_logdir()




