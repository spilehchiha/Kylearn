from abc import abstractmethod
import numpy as np
from framework.mini_batch import random_index

class Dataset:
    def __init__(self,  **kwargs):
        self.train_set = None
        self.eval_set = None
        self.test_set = None

    def evaluation_generator(self, batch_size=100):
        print(len(self.eval_set))

        def generate():
            for idx in range(0, len(self.eval_set), batch_size):
                print(idx)
                yield self.eval_set[idx:(idx + batch_size)]

        return generate

    def training_generator(self, batch_size=100, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        for batch_idxs in random_index(len(self.train_set), batch_size, random):
            yield self.train_set[batch_idxs]




