import numpy as np


def cross_entropy_loss(y_true, y_pred):
    # Ensure numerical stability by clipping predictions
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    # Compute cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]  # Use shape[1] if y_true is (num_classes, num_samples)
    return np.squeeze(loss)


def one_hot(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)].T


def get_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / y_true.size
