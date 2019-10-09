import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt


def save_figures(X, y, savename):
    X_, y_ = X.squeeze(dim=1).detach().cpu().numpy(), y.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(16):
        axis = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        axis.imshow(X_[i])
        points = np.vstack(np.split(y_[i], 15)).T * 48 + 48
        axis.plot(points[0], points[1], 'o', color='red')
    fig.savefig(savename)
