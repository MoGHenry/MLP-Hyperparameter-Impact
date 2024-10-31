import numpy as np


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def get_activation_function(self, activation="relu"):
    activation_function, derivative_activation_function = get_activation_functions(activation)
    return activation_function, derivative_activation_function


# one-hot encoding
# num_classes is the number of unique labels in the output
def one_hot(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)]


class Layer:
    def __init__(self, neurons, activation_function):
        self.head = False
        self.tail = False
        self.neurons = neurons
        self.activation_function, self.derivative_activation_function = get_activation_function(activation_function)

    def forward_propagation(self, x, w1, b1, a0=None):
        if self.head:
            z1 = np.dot(w1, x) + b1
            a1 = self.activation_function(z1)
            return a1, z1
        elif self.tail:
            z1 = np.dot(w1, a0) + b1
            y = softmax(z1)
            return y, z1
        else:
            z1 = np.dot(w1, a0) + b1
            a1 = self.activation_function(z1)
            return a1, z1

    # backward propagation
    # calculate dw1, db1, dw2, db2
    def backward_propagation(self, x, a0, a1, y=None, dz1=None, z1=None, w2=None, dz2=None):
        if self.tail:
            true_labels = one_hot(y, a1.shape[0])
            dz1 = a1 - true_labels
            dw1 = np.dot(dz1, a0.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)
            return dw1, db1
        elif self.head:
            dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
            dw1 = np.dot(dz1, x.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)
            return dw1, db1
        else:
            dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
            dw1 = np.dot(dz1, a0.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)
            return dw1, db1
