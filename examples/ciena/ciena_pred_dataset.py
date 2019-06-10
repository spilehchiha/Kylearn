import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.resampling import random_upsampling
import collections

class pred_Dataset(Dataset):
    def __init__(self, x_path, y_path):
        super().__init__()

        X = np.load(x_path)
        y = np.load(y_path).flatten()
        print(collections.Counter(y))
        self.shape = X.shape
        X, y = random_upsampling(X, y)
        X = X.reshape([-1]+list(self.shape[1:]))
        print(collections.Counter(y))

        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('y', np.int32, (1))
        ])
        self.train_set['x'] = X
        self.train_set['y'] = y


        # pos_sample = self.train_set[self.train_set['y'] == 1]
        # neg_sample = self.train_set[self.train_set['y'] == 0]

        # _, neg_sample = train_test_split(neg_sample, test_size=pos_sample.shape[0], random_state=42)
        _, self.train_set= train_test_split(self.train_set, test_size=20000, random_state= 42)
        self.train_set, self.test_set = train_test_split(self.train_set, test_size=0.2, random_state=41)
        self.train_set, self.eval_set = train_test_split(self.train_set, test_size=0.1, random_state=40)
