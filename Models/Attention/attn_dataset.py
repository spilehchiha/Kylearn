import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.mini_batch import random_index
import collections
class Attn_dataset_1d(Dataset):
    def __init__(self, feature_path, dev_path, label_path, out_num):
        super().__init__()

        # train set
        X = np.load(feature_path + '_train.npy')
        dev = np.load(dev_path + '_train.npy')
        # y = np.load(label_path + '_train.npy').reshape(-1)
        # y = y-1
        # y = np.eye(out_num)[y].reshape([-1, out_num])
        y = np.load(label_path + '_train.npy')
        y = y.reshape([-1, out_num])
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, (dev.shape[1])),
            ('y', np.int32, ([out_num]))
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy').reshape([-1,out_num])
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, (dev2.shape[1])),
            ('y', np.int32, ([out_num]))
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.1, random_state=22)

class Attn_dataset_2d(Dataset):
    def __init__(self, feature_path, dev_path, label_path, out_num):
        super().__init__()

        # train set
        X = np.expand_dims(np.load(feature_path), axis=-1)
        dev = np.load(dev_path)
        y = np.load(label_path)
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        dataset = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, (dev.shape[1])),
            ('y', np.int32, ([out_num]))
        ])
        dataset['x'] = X
        dataset['dev'] = dev
        dataset['y'] = y
        # normal,_ = train_test_split(normal, train_size= anomaly.shape[0], random_state=22)
        self.train_set,self.test_set = train_test_split(dataset, test_size=0.1, random_state=23
        )
        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.02, random_state=24)
        del dataset

        self.anomaly = self.train_set[self.train_set['y'].flatten() != 0]
        self.normal = self.train_set[self.train_set['y'].flatten() == 0]

    def negative_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.normal) > 0
        for batch_idxs in random_index(len(self.normal), batch_size, random):
            yield self.normal[batch_idxs]

    def positive_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.anomaly) > 0
        for batch_idxs in random_index(len(self.anomaly), batch_size, random):
            yield self.anomaly[batch_idxs]

    def training_generator(self, batch_size=100, portion = 0.5):
        pos = self.positive_generator(batch_size=batch_size - int(batch_size * portion))
        neg = self.negative_generator(batch_size = int(batch_size * portion))
        while True:
            l = pos.__next__()
            u = neg.__next__()
            yield np.concatenate([l, u], axis=0)