from abc import abstractmethod
import numpy as np
class Dataset:
    def __init__(self, shape, x_path, y_path):
        self.train_x = np.load(x_path)
        self.train_y = np.load(y_path)
        self.train_x = self.train_x.reshape(shape=shape)
        self.train_set = np.zeros(self.train_x.shape[0], dtype=[
            ('x', np.float32, (shape[1:])),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        self.train_set['x'] = self.train_x
        self.train_set['y'] = self.train_y
        del self.train_x, self.train_y
        self.test_set = None
    @abstractmethod
    def process_labels(self, **kwargs):
        pass

    @abstractmethod
    def split_dataset(self, **kwargs):
        pass

    def return_dataset(self):
        return self.train_set, self.test_set