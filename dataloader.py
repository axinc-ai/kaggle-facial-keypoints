# -*- encoding: utf-8 -*-
import os
import numpy as np
import torch
from pandas.io.parsers import read_csv
import sklearn.utils
# from matplotlib import pyplot as plt  # for debug purpose

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

# TODO make a batch ? load all images on memory once ?
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
        self.next = True  # when true, there remains the unreaded datas.  # TODO maybe modify?

    def load(self, cols=None):
        """
        :param test: loading data from FTEST when the param is true, otherwise loading data from FTRAIN
        :param cols: if you give a list to cols, this function returns only the data corresponding the cols
        :return:
        """

        fname = FTEST if self.test else FTRAIN
        print(fname)  # TODO remove this line
        df = read_csv(os.path.expanduser(fname))

        # transform pixel values which are separated by "space" to a numpy array
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:
            df = df[list(cols) + ['Image']]

        print(df.count())  # output the number of each column
        df = df.dropna()  # if there is no data, drop it  # TODO find a better solution ?

        X = np.vstack(df['Image'].values) / 255.  # regularisation between 0 and 1
        X = X.astype(np.float32)  # add channel information

        if not self.test:  # only FTRAIN has a label
            y = df[df.columns[:-1]].values
            y = (y - 48) / 48  # regularisation between -1 and 1
            X, y = sklearn.utils.shuffle(X, y, random_state=42)  # shuffle the data (fixed seed)
            y = torch.from_numpy(y.astype(np.float32))
        else:
            y = None
        X = torch.from_numpy(X).reshape(X.shape[0], 1, 96, 96)  # TODO be to variable ?
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


# for test
if __name__ == "__main__":
    shuffle = True
    test = False
    nb_batch = 32
    train_dataLoader = DataLoader(nb_batch, test=test)
    for epoch in range(2):
        print("===================================== epoch {} ======================================".format(epoch + 1))
        while train_dataLoader.next_is_available():
            X, y = train_dataLoader.get_batch()
            print(X[0])
            if not test:
                print(y[0])
        train_dataLoader.restart(shuffle=shuffle)
