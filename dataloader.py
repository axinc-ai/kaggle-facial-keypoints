# -*- encoding: utf-8 -*-
import os
import numpy as np
import torch
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split
import copy
# from matplotlib import pyplot as plt  # for debug purpose

# FTRAIN = 'data/training.csv'
FTRAIN = 'data/resized226.csv'
FTRANSFORMED = 'data/resized226_rotate_30.csv'
FTEST = 'data/test.csv'
IMG_SIZE = 226

# TODO how to use this class -> comments with example
# TODO loading another dataset (transformed version etc.)

# TODO refactoring how to use evaluiation data
# For now, create train_dataloader, then make eval_dataloader with arguments
# eval_X, eval_y = train_loader.get_eval_data()
# eval_loader = dataloader.Dataloader(
#                   nb_batch,
#                   test=False,
#                   X=eval_X,
#                   y=eval_y)


class DataLoader:
    def __init__(self, nb_batch, test=False, X=None, y=None):
        self.next_index = 0
        self.nb_batch = nb_batch
        self.test = test
        if X is None and y is None:
            self.X, self.y, self.eval_X, self.eval_y = self.load()
        else:
            self.X, self.y = X, y
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

        # TODO add argument or something else
        if not self.test:
            df_transformed = read_csv(os.path.expanduser(FTRANSFORMED))
            df = pd.concat([df, df_transformed])

        # transform pixel values which are separated by " " to a numpy array
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:
            df = df[list(cols) + ['Image']]

        print(df.count())  # output the number of each column
        df = df.dropna()  # if there is no data, drop it
        # df.fillna(method='ffill', inplace=True)
        print(df.info())
        # regularisation between 0 and 1
        X = np.vstack(df['Image'].values) / 255.
        X = X.astype(np.float32)  # add channel information

        if not self.test:  # only FTRAIN has a label -> eval_X, eval_y
            y = df[df.columns[:-1]].values
            y = (y - IMG_SIZE//2) / (IMG_SIZE // 2)  # regularisation between -1 and 1

            X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE)

            # data augmentation [flip]
            X, y = self.data_aug_flip(X, y)

            # X, y = sklearn.utils.shuffle(X, y, random_state=42)
            X, eval_X, y, eval_y = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # numpy array to torch tensor
            X = torch.from_numpy(X).reshape(X.shape[0], 1, IMG_SIZE, IMG_SIZE)
            eval_X = torch.from_numpy(eval_X).reshape(
                eval_X.shape[0], 1, IMG_SIZE, IMG_SIZE
            )
            y = torch.from_numpy(y.astype(np.float32))
            eval_y = torch.from_numpy(eval_y.astype(np.float32))
        else:
            X = torch.from_numpy(X).reshape(X.shape[0], 1, IMG_SIZE, IMG_SIZE)
            y, eval_X, eval_y = None, None, None

        return X, y, eval_X, eval_y

    # TODO modify to adapt evaluation mode
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

    def get_eval_data(self):
        X, y = self.eval_X, self.eval_y
        self.eval_X = None  # release memory
        self.eval_y = None
        return X, y

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
        for a, b in flip_indices:  # flip annotations
            flip_y[:, a], flip_y[:, b] = y[:, b], y[:, a]
        flip_y[:, ::2] = -1 * flip_y[:, ::2]
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
    print('done')
    # for epoch in range(2):
    #     print("======= epoch {} =======".format(epoch + 1))
    #     while train_dataLoader.next_is_available():
    #         X, y = train_dataLoader.get_batch()
    #         print(X[0])
    #         if not test:
    #             print(y[0])
    #     train_dataLoader.restart(shuffle=shuffle)
