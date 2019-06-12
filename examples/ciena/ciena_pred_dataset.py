import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.resampling import random_upsampling, random_downsampling
import collections

class pred_Dataset(Dataset):
    def __init__(self, x_path, y_path):
        super().__init__()

        X = np.load(x_path)
        y = np.load(y_path)
        shape = X.shape
        index_num = np.arange(shape[0])
        index = y == 1
        index_pos = np.extract(index, index_num)
        index_neg = np.extract(~index, index_num)

        idx_pos_train, idx_pos_test = train_test_split(index_pos, test_size= 0.3, random_state= 6)
        idx_pos_train, idx_pos_val = train_test_split(idx_pos_train, test_size= 0.2, random_state= 6)
        idx_neg_train, idx_neg_test = train_test_split(index_neg, test_size= 2000, random_state= 6)
        idx_train = idx_pos_train.tolist()+idx_neg_train.tolist()
        idx_test = idx_pos_test.tolist()+idx_neg_test.tolist()

        X_train, y_train = random_upsampling(X[idx_train], y[idx_train])
        X_val, y_val = X[idx_pos_val], y[idx_pos_val]
        X_test, y_test = X[idx_test], y[idx_test]


        self.train_set = np.zeros(X_train.shape[0], dtype=[
            ('x', np.float32, (X_train.shape[1:])),
            ('y', np.int32, ([1]))
        ])
        self.train_set['x'] = X_train
        self.train_set['y'] = y_train

        self.val_set = np.zeros(X_val.shape[0], dtype=[
            ('x', np.float32, (X_val.shape[1:])),
            ('y', np.int32, ([1]))
        ])
        self.val_set['x'] = X_val
        self.val_set['y'] = y_val.reshape([-1, 1])

        self.test_set = np.zeros(X_test.shape[0], dtype=[
            ('x', np.float32, (X_test.shape[1:])),
            ('y', np.int32, ([1]))
        ])

        self.test_set['x'] = X_test
        self.test_set['y'] = y_test.reshape([-1, 1])

        # pos_sample = self.train_set[self.train_set['y'] == 1]
        # neg_sample = self.train_set[self.train_set['y'] == 0]
        # pos_train, pos_test = train_test_split(pos_sample, test_size=0.4, random_state=42)
        # self.train_set, self.test_set = train_test_split(self.train_set, test_size=20000, random_state=41)
        # self.train_set, self.eval_set = train_test_split(self.train_set, test_size=0.1, random_state=40)


class pred_Dataset_2(Dataset):
    def __init__(self, x_path, y_path):
        super().__init__()

        X = np.load(x_path)
        y = np.load(y_path)
        print(collections.Counter(y.flatten()))
        X = np.nan_to_num(X)
        X, y = random_upsampling(X, y)
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('y', np.int32, ([1]))
        ])
        self.train_set['x'] = X
        self.train_set['y'] = y


        # pos_sample = self.train_set[self.train_set['y'] == 1]
        # neg_sample = self.train_set[self.train_set['y'] == 0]
        # pos_train, pos_test = train_test_split(pos_sample, test_size=0.4, random_state=42)
        self.train_set, self.test_set = train_test_split(self.train_set, test_size=5000, random_state=41)
        self.train_set, self.val_set = train_test_split(self.train_set,train_size=100000, test_size=5000, random_state=40)
