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
class AttnDataSet2DMultiClass(Dataset):
    def __init__(self, feature_path, dev_path, label_path, out_num):
        super().__init__()

        # train set
        X = np.expand_dims(np.load(feature_path, allow_pickle=True), axis=-1)
        dev = np.load(dev_path, allow_pickle=True)
        y = np.load(label_path, allow_pickle=True)
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

        class_1 = dataset[dataset['y'][:, 0] == 1]
        class_2 = dataset[dataset['y'][:, 1] == 1]
        class_3 = dataset[dataset['y'][:, 2] == 1]
        class_4 = dataset[dataset['y'][:, 3] == 1]
        class_5 = dataset[dataset['y'][:, 4] == 1]
        class_6 = dataset[dataset['y'][:, 5] == 1]
        class_7 = dataset[dataset['y'][:, 6] == 1]
        class_8 = dataset[dataset['y'][:, 7] == 1]
        class_9 = dataset[dataset['y'][:, 8] == 1]
        class_10 = dataset[dataset['y'][:, 9] == 1]
        class_11 = dataset[dataset['y'][:, 10] == 1]
        class_12 = dataset[dataset['y'][:, 11] == 1]
        del dataset

        class_1, _ = train_test_split(class_1, train_size=0.8, random_state=21)
        class_2, _ = train_test_split(class_2, train_size=0.8, random_state=25)
        class_3, _ = train_test_split(class_3, train_size=0.8, random_state=26)
        class_4, _ = train_test_split(class_4, train_size=0.8, random_state=27)
        class_5, _ = train_test_split(class_5, train_size=0.8, random_state=28)
        class_6, _ = train_test_split(class_6, train_size=0.8, random_state=29)
        class_7, _ = train_test_split(class_7, train_size=0.8, random_state=30)
        class_8, _ = train_test_split(class_8, train_size=0.8, random_state=31)
        class_9, _ = train_test_split(class_9, train_size=0.8, random_state=32)
        class_10, _ = train_test_split(class_10, train_size=0.8, random_state=33)
        class_11, _ = train_test_split(class_11, train_size=0.8, random_state=34)
        class_12, _ = train_test_split(class_12, train_size=0.8, random_state=35)

        self.train_set, self.test_set = train_test_split(
            np.concatenate([class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10,
                            class_11, class_12], axis=0), test_size=0.2, random_state=23)
        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.1, random_state=24)
