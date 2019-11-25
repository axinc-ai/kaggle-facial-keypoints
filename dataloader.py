# -*- encoding: utf-8 -*-
import os
import numpy as np
import torch
from pandas.io.parsers import read_csv
import sklearn.utils
import copy
# from matplotlib import pyplot as plt  # for debug purpose

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

# TODO multithreading ? (cf. heritates Thread, Consumer-Producer)
# TODO how to use this class -> comments with example
# TODO cross_validation ? (in this case, shuffle feature would be difficult)
# TODO modify 48 for regularisation of y (generate image also)


class DataLoader:
    def __init__(self, nb_batch, test=False):
        self.next_index = 0
        self.nb_batch = nb_batch
        self.test = test
        self.X, self.y = self.load()
        self.nb_file = self.X.size(0)
        self.next = True  # when true, there remains the unreaded datas.

    def load(self, cols=None):
        """
        :param test: loading data from FTEST when the param is true, 
                     otherwise loading data from FTRAIN
        :param cols: if you give a list to cols, this function returns only 
                     the data corresponding the cols
        :return:
        """

        fname = FTEST if self.test else FTRAIN
        df = read_csv(os.path.expanduser(fname))

        # transform pixel values which are separated by " " to a numpy array
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:
            df = df[list(cols) + ['Image']]

        print(df.count())  # output the number of each column
        df = df.dropna()  # if there is no data, drop it
        # df.fillna(method='ffill', inplace=True)

        # regularisation between 0 and 1
        X = np.vstack(df['Image'].values) / 255.  
        X = X.astype(np.float32)  # add channel information

        if not self.test:  # only FTRAIN has a label
            y = df[df.columns[:-1]].values
            y = (y - 48) / 48  # regularisation between -1 and 1
            X = X.reshape(X.shape[0], 96, 96)

            # data augmentation [flip]
            X, y = self.data_aug_flip(X, y)

            X, y = sklearn.utils.shuffle(X, y, random_state=42)
            y = torch.from_numpy(y.astype(np.float32))
        else:
            y = None
        # TODO be to variable ?
        X = torch.from_numpy(X).reshape(X.shape[0], 1, 96, 96)
        return X, y

    def get_batch(self):
        X = self.X[self.next_index:self.next_index + self.nb_batch]
        if not self.test:
            y = self.y[self.next_index:self.next_index + self.nb_batch]
        else:
            y = None
        self.next_index += self.nb_batch
        if self.next_index >= self.nb_file:
            self.next = False
        return X, y

    def next_is_available(self):
        return self.next

    def restart(self, shuffle=False):
        """

        :param shuffle: when training, it's recommend to set this True
        :return:
        """
        self.next = True
        self.next_index = 0
        if shuffle:
            idx = np.random.permutation(self.nb_file)
            self.X, self.y = self.X[:, idx], self.y[idx]

    def data_aug_flip(self, X, y):
        """
        data augmentation function for trainig dataset [version flip]
        :param X: images
        :param y: anotation data
        :return new_X, new_Y
        """
        flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11),
                        (12, 16), (13, 17), (14, 18), (15, 19), (22, 24),
                        (23, 25)]

        flip_X = X[:, :, ::-1]  # flip images
        flip_y = copy.deepcopy(y)
        for a, b in flip_indices: # flip annotations
            flip_y[a], flip_y[b] = y[b], y[a]
        new_X = np.vstack(np.array([X, flip_X]))
        new_y = np.vstack(np.array([y, flip_y]))
        return new_X, new_y

        
# for debug
if __name__ == "__main__":
    shuffle = True
    # test = True
    test = False
    nb_batch = 32
    train_dataLoader = DataLoader(nb_batch, test=test)
    
    # for epoch in range(2):
    #     print("======= epoch {} =======".format(epoch + 1))
    #     while train_dataLoader.next_is_available():
    #         X, y = train_dataLoader.get_batch()
    #         print(X[0])
    #         if not test:
    #             print(y[0])
    #     train_dataLoader.restart(shuffle=shuffle)
