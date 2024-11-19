import numpy as np


# activation function
# relu, sigmoid, tanh, softmax
def activation_function_relu(z):
    return np.maximum(0, z)


def activation_function_sigmoid(z):
    return 1. / (1. + np.exp(-z))


def activation_function_tanh(z):
    return np.tanh(z)


# derivation of activation function
# relu, sigmoid, tanh, softmax
def activation_function_derivative_relu(z):
    # return np.greater(z, 0).astype(int)
    return z > 0


def activation_function_derivative_sigmoid(z):
    return activation_function_sigmoid(z) * (1. - activation_function_sigmoid(z))


def activation_function_derivative_tanh(z):
    return 1. - np.power(activation_function_tanh(z), 2)


def get_activation_functions(activation="relu"):
    if activation == "relu":
        return activation_function_relu, activation_function_derivative_relu
    elif activation == "sigmoid":
        return activation_function_sigmoid, activation_function_derivative_sigmoid
    elif activation == "tanh":
        return activation_function_tanh, activation_function_derivative_tanh
    else:
        raise ValueError("Invalid activation function")