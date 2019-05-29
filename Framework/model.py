from absl import flags
import tensorflow as tf
from abc import abstractmethod
FLAGS = flags.FLAGS

class Model():
    def __init__(self, Network, ckpt_path, tsboard_path):
        self.checkpoint_path = ckpt_path
        self.tensorboard_path = tsboard_path
        self.Network = Network

    def network(self, **kwargs):
        return self.Network.network(**kwargs)

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
    def test(self, **kwargs):
        pass





