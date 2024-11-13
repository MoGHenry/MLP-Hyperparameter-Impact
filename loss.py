import numpy as np


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return np.squeeze(loss)


def one_hot(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)].T