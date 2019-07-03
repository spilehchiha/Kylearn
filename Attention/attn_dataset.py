import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.resampling import multi_inputs_random_upsampling
import collections


class Attn_dataset(Dataset):
    def __init__(self, feature_path, dev_path, label_path):
        super().__init__()

        # train set
        X = np.load(feature_path + '_train.npy')
        dev = np.load(dev_path + '_train.npy')
        y = np.load(label_path + '_train.npy')
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        print(collections.Counter(np.argmax(y, 1).flatten()))
        X, dev, y = multi_inputs_random_upsampling(X, dev, y, 12)
        print(collections.Counter(np.argmax(y, 1).flatten()))
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, (dev.shape[1])),
            ('y', np.int32, (y.shape[1]))
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy')
        print(collections.Counter(np.argmax(y2, 1).flatten()))
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, (dev2.shape[1])),
            ('y', np.int32, (y2.shape[1]))
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.1, random_state=22)
