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
    z1 = w1*x + b1
    a1 = Sigmoid(z1)
    z2 = w2*x + b2
    a2 = SoftMax(z2)
    return z1, a1, z2, a2

# activation function
# relu, sigmoid, tanh, softmax
def activation_function_sigmoid(z):
    return 1/(1-exp(z))

def activation_function_relu(z):
    return maximum(0,z)

def activation_function_tanh(z):
    return (exp(2x)-1)/(exp(2x)+1)

def activation_function_softmax(z):
    return exp(z)/sum(exp(z))

# derivation of activation function
# relu, sigmoid, tanh, softmax
def activation_function_derivative_sigmoid(z):
    return exp(z)*(sigmoid(z)**2)

def activation_function_derivative_relu(z):
    return z > 0

def activation_function_derivative_tanh(z):
    return 4/(exp(z)+exp(-z))**2
    
def activation_function_derivative_softmax(z):
    return softmax(z)*(1-softmax(z))

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
