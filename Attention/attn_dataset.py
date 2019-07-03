import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.resampling import multi_inputs_random_upsampling
import collections


class attn_dataset(Dataset):
    def __init__(self, x_path, y_path, dev_path):
        super().__init__()

        # train set
        X = np.load(x_path + '_train')
        dev = np.load(dev_path + '_train')
        y = np.load(y_path + '_train')
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        print(collections.Counter(y.flatten()))
        # X = np.nan_to_num(X)
        X, dev, y = multi_inputs_random_upsampling(X, dev, y, 12)
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, ([dev.shape[1]])),
            ('y', np.int32, ([1]))
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # test_set
        X2 = np.load(x_path + '_test')
        dev2 = np.load(dev_path + '_test')
        y2 = np.load(y_path + '_test')
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]
        print(collections.Counter(y2.flatten()))

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, ([dev2.shape[1]])),
            ('y', np.int32, ([1]))
        ])
        self.train_set['x'] = X2
        self.train_set['dev'] = dev2
        self.train_set['y'] = y2

        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.1, random_state=22)
