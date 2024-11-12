import numpy as np
from activations import get_activation_functions
from loss import one_hot


def softmax(z):
    z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return z_exp / np.sum(z_exp, axis=0, keepdims=True)


def get_activation_function(self, activation="relu"):
    activation_function, derivative_activation_function = get_activation_functions(activation)
    return activation_function, derivative_activation_function


# one-hot encoding
# num_classes is the number of unique labels in the output



class Layer:
    def __init__(self, neurons, activation_function):
        self.input_layer = False
        self.output_layer = False
        self.neurons = neurons
        self.activation_function, self.derivative_activation_function = get_activation_function(activation_function)

    def forward_propagation(self, w1, b1, x=None, a0=None):
        # breakpoint()
        if self.input_layer:
            # Only the head layer should use `x` as input directly
            z1 = np.dot(w1, x) + b1
            a1 = self.activation_function(z1)
            return a1, z1
        elif self.output_layer or a0 is not None:
            # breakpoint()
            z1 = np.dot(w1, a0) + b1
            y = softmax(z1) if self.output_layer else self.activation_function(z1)
            return y, z1
        else:
            raise ValueError("Invalid input: a0 is None.")

    # backward propagation
    # calculate dw1, db1, dw2, db2
    def backward_propagation(self, x, a0, a1, y=None, dz1=None, z1=None, w2=None, dz2=None):
        if self.output_layer:
            true_labels = one_hot(y, self.neurons)
            dz1 = a1 - true_labels
            dw1 = np.dot(dz1, a0.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)
            return dw1, db1, dz1
        elif self.input_layer:
            dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
            dw1 = np.dot(dz1, x.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)
            return dw1, db1, dz1
        else:
            # breakpoint()
            dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
            dw1 = np.dot(dz1, a0.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)
            return dw1, db1, dz1

    def print_activation_function(self):
        return self.activation_function.__name__
