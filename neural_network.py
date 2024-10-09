import numpy as np

# https://youtu.be/w8yWXqWQYmU?si=MXhI9EgsfYXMdshP&t=917


# initialize w1,b1,w2,b2
# w1 shape (input_size, hidden_size)
# b1 shape (1, hidden_size)
# w2 shape (hidden_size, output_size)
# b2 shape (1, output_size)
def initialize_parameters(input_size, hidden_size, output_size):
    return


# forward propagation
# calculate z1, a1, z2, a2
# return
def forward_propagation(x, w1, b1, w2, b2):
    return


# activation function
# relu, sigmoid, tanh, softmax
def activation_function_xxx(z):
    return


# derivation of activation function
# relu, sigmoid, tanh, softmax
def activation_function_derivative_xxx(z):
    return


# softmax function
def softmax(z):
    return


# one-hot encoding
def one_hot(y):
    return


# backward propagation
# calculate dw1, db1, dw2, db2
def backward_propagation(x, y, z1, a1, z2, a2, w2):
    return


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    return


def get_predictions(a2):
    return


def get_accuracy(y_pred, y_true):

# in each iteration
# forward propagation
# backward propagation
# update parameters
# print the accuracy and loss
def gradient_descent(X, y, w1, b1, w2, b2, learning_rate, num_iterations):
    return


def make_prediction(X, w1, b1, w2, b2):
    return

def test_model(X, y, w1, b1, w2, b2):
    return