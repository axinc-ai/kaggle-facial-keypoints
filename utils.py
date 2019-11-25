import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import sys
import dataloader


def save_figures(X, y, savename):
    X_, y_ = X.squeeze(dim=1).detach().cpu().numpy(), y.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05
    )
    for i in range(16):
        axis = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        axis.imshow(X_[i])
        points = np.vstack(np.split(y_[i], 15)).T * 48 + 48
        axis.plot(points[0], points[1], 'o', color='red')
    fig.savefig(savename)


def generate_csv(output, filepath='data/IdLookupTable.csv'):
    lookid_data = pd.read_csv(filepath)
    lookid_list = list(lookid_data['FeatureName'])
    imageID = list(lookid_data['ImageId'] - 1)
    pre_list = list(output.cpu().numpy())
    rowid = list(lookid_data['RowId'])
    feature = []
    for f in list(lookid_data['FeatureName']):
        feature.append(lookid_list.index(f))
    preded = []
    for x, y in zip(imageID, feature):
        preded.append(pre_list[x][y])
    rowid = pd.Series(rowid, name='RowId')
    loc = pd.Series(preded, name='Location')
    submission = pd.concat([rowid, loc * 48 + 48], axis=1)
    submission.to_csv('submission.csv', index=False)


