# import numpy as np
# from sklearn.model_selection import train_test_split
# from Framework.dataset import Dataset
#
# class pred_Dataset(Dataset):
#     def __init__(self, shape, x_path, y_path):
#         super().__init__(x_path=x_path, y_path=y_path)
#         self.train_x = self.train_x.reshape(shape=shape)
#         self.train_set = np.zeros(self.train_x.shape[0], dtype=[
#             ('x', np.float32, (shape[1:])),
#             ('y', np.int32, ())  # We will be using -1 for unlabeled
#         ])
#         self.train_set['x'] = self.train_x
#         self.train_set['y'] = self.train_y
#         del self.train_x, self.train_y
#         self.test_set = None
#
#     def process_labels(self):
#         self.labeled = self.train_set[self.train_set['y'] != -1]
#         self.unlabeled = self.train_set[self.train_set['y'] == -1]
#
#     def split_dataset(self, testset_size, random_state):
#         lb_train, self.test_set = train_test_split(self.labeled, test_size=testset_size, random_state=random_state)
#         self.train_set = np.concatenate([lb_train, self.unlabeled], axis=0)
#
