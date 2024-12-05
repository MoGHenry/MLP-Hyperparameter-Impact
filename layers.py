import numpy as np
from activations import get_activation_functions
from loss import one_hot


def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    # breakpoint()
    return a


def get_activation_function(activation="relu"):
    activation_function, derivative_activation_function = get_activation_functions(activation)
    return activation_function, derivative_activation_function


class Layer:
    def __init__(self, layer_name, neurons, activation_function):
        self.layer_name = layer_name
        self.input_layer = False
        self.output_layer = False
        self.neurons = neurons
        self.activation_function, self.derivative_activation_function = get_activation_function(activation_function)

    def forward_propagation(self, w1, b1, x=None, a0=None):
        # breakpoint()
        if self.layer_name == "first_hidden_layer":
            # Only the head layer should use `x` as input directly
            z1 = np.dot(w1, x) + b1
            a1 = self.activation_function(z1)
            # print("z1, a1:", self.layer_name, sum(sum(z1)), sum(sum(a1)))
            return a1, z1
        elif self.output_layer or a0 is not None:
            # breakpoint()
            z1 = np.dot(w1, a0) + b1
            y = softmax(z1) if self.output_layer else self.activation_function(z1)
            # print("z1, a1:", self.layer_name, sum(sum(z1)), sum(sum(y)))
            return y, z1
        else:
            raise ValueError("Invalid input: a0 is None.")

    def backward_propagation(self, x, a0, a1, y=None, dz1=None, z1=None, w2=None, dz2=None):
        m = a0.shape[1]  # Number of examples
        # breakpoint()
        if self.output_layer:
            true_labels = one_hot(y, self.neurons)
            dz1 = a1 - true_labels
            dw1 = (1 / m) * np.dot(dz1, a0.T)
            db1 = (1 / m) * np.sum(dz1)
            return dw1, db1, dz1
        # elif self.input_layer:
        elif self.layer_name == "first_hidden_layer":
            dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
            dw1 = (1 / m) * np.dot(dz1, x.T)
            db1 = (1 / m) * np.sum(dz1)
            return dw1, db1, dz1
        else:
            dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
            dw1 = (1 / m) * np.dot(dz1, a0.T)
            db1 = (1 / m) * np.sum(dz1)
            return dw1, db1, dz1

    def print_activation_function(self):
        return self.activation_function.__name__
