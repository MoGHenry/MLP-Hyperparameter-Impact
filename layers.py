import numpy as np


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


class Layer:
    def __init__(self, neurons, activation_function):
        self.head = False
        self.tail = False
        self.neurons = neurons
        self.activation_function = activation_function

    def forward_propagation(self, w, b):
        z1 = np.dot(w, x) + b
        a1 = self.activation_function(z1)
        return a1, z1

    # softmax function

    # one-hot encoding
    # num_classes is the number of unique labels in the output
    def one_hot(y, num_classes):
        return np.eye(num_classes)[y.reshape(-1)]

    # backward propagation
    # calculate dw1, db1, dw2, db2
    def backward_propagation(x, y, z1, a1, z2, a2, w2, derivative_activation_function: str):
        one_hot_y = one_hot(y, a2.shape[0])
        dz2 = a2 - one_hot_y
        dw2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis=1, keepdims=True)
        da1 = np.dot(w2.T, dz2)
        dz1 = da1 * activation_function_derivative_relu(z1)
        dw1 = np.dot(dz1, x.T)
        db1 = np.sum(dz1, axis=1, keepdims=True)
        backward_cache = {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2
        }
        return backward_cache

