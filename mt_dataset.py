import numpy as np
from sklearn.model_selection import train_test_split
from Framework.dataset import Dataset

class MTDataset(Dataset):
    def __init__(self):
        super().__init__()

    def process_labels(self):
        self.labeled = self.train_set[self.train_set['y'] != -1]
        self.unlabeled = self.train_set[self.train_set['y'] == -1]

    def split_dataset(self, testset_size, random_state):
        lb_train, self.test_set = train_test_split(self.labeled, test_size=testset_size, random_state=random_state)
        self.train_set = np.concatenate([lb_train, self.unlabeled], axis=0)

