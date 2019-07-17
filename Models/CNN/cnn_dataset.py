import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.resampling import multi_inputs_random_upsampling
import collections


class CNN_dataset(Dataset):
    def __init__(self, feature_path, label_path, out_num):
        super().__init__()

        # train set
        X = np.load(feature_path + '_train.npy')
        y = np.load(label_path + '_train.npy').reshape([-1,out_num])
        assert X.shape[0] == y.shape[0]
        # X, dev, y = multi_inputs_random_upsampling(X, dev, y, 12)
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('y', np.int32, ([out_num]))
        ])
        self.train_set['x'] = X
        self.train_set['y'] = y

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy').reshape([-1,out_num])
        assert X2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('y', np.int32, ([out_num]))
        ])
        self.test_set['x'] = X2
        self.test_set['y'] = y2

        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.1, random_state=22)

class Concat_dataset(Dataset):
    def __init__(self, feature_path, dev_path, label_path, out_num):
        super().__init__()
        # train set
        X = np.load(feature_path)
        dev = np.load(dev_path)
        y = np.load(label_path)
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        dev = dev.repeat(X.shape[1], axis=0)
        dev = dev.reshape([-1, X.shape[1], dev.shape[1]])
        X = np.concatenate([dev, X], axis=2)
        X = np.expand_dims(X, -1)
        dataset = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('y', np.int32, ([out_num]))
        ])
        dataset['x'] = X
        dataset['y'] = y

        anomaly = dataset[dataset['y'].flatten() != 0]
        normal = dataset[dataset['y'].flatten() == 0]
        del dataset
        normal,_ = train_test_split(normal, train_size= anomaly.shape[0], random_state=22)
        self.train_set,self.test_set = train_test_split(
            np.concatenate([anomaly, normal], axis=0), test_size=0.2, random_state=23
        )
        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.1, random_state=24)
