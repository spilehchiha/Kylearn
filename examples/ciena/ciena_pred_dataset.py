import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset


class pred_Dataset(Dataset):
    def __init__(self, x_path, y_path):
        super().__init__()

        x = np.load(x_path)
        self.shape = x.shape

        self.train_set = np.zeros(self.shape[0], dtype=[
            ('x', np.float32, (self.shape[1:])),
            ('y', np.int32, (1))
        ])
        self.train_set['x'] = x
        self.train_set['y'] = np.load(y_path).flatten()


        pos_sample = self.train_set[self.train_set['y'] == 1]
        neg_sample = self.train_set[self.train_set['y'] == 0]

        _, neg_sample = train_test_split(neg_sample, test_size=pos_sample.shape[0], random_state=42)
        self.train_set, self.test_set = train_test_split(np.concatenate([pos_sample, neg_sample]), test_size=0.2,
                                                         random_state=41)
        self.train_set, self.eval_set = train_test_split(self.train_set, test_size=0.1, random_state=40)
