import numpy as np

# https://youtu.be/w8yWXqWQYmU?si=MXhI9EgsfYXMdshP&t=917


# initialize w1,b1,w2,b2
# w1 shape (input_size, hidden_size)
# b1 shape (1, hidden_size)
# w2 shape (hidden_size, output_size)
# b2 shape (1, output_size)
def std_initialize_parameters(input_size, hidden_size, output_size):
    w1 = np.random.randn(hidden_size, input_size)
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size)
    b2 = np.zeros((output_size, 1))
    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }
    return parameters


def he_initialize_parameters(input_size, hidden_size, output_size):
    w1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((output_size, 1))
    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }
    return parameters


# forward propagation
# calculate z1, a1, z2, a2
# return
def forward_propagation(x, w1, b1, w2, b2, activation_function: str):
    activation_functions = {
        "relu": activation_function_relu,
        "sigmoid": activation_function_sigmoid,
        "tanh": activation_function_tanh
    }
    z1 = np.dot(w1, x) + b1
    a1 = activation_functions[activation_function](z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    forward_cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2
    }
    return forward_cache


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
    return np.greater(z, 0).astype(int)


def activation_function_derivative_sigmoid(z):
    return activation_function_sigmoid(z) * (1. - activation_function_sigmoid(z))


def activation_function_derivative_tanh(z):
    return 1. - np.power(activation_function_tanh(z), 2)


# softmax function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


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


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    updated_parameters = {
        "w1": w1 - learning_rate * dw1,
        "b1": b1 - learning_rate * db1,
        "w2": w2 - learning_rate * dw2,
        "b2": b2 - learning_rate * db2
    }
    return updated_parameters


def get_predictions(a2):
    return np.argmax(a2, axis=0)


def get_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# in each iteration
# forward propagation
# backward propagation
# update parameters
# print the accuracy and loss
# TODO change parameters to dictionary
def gradient_descent(X, y, w1, b1, w2, b2, learning_rate, num_iterations, activation_function
                     , derivative_activation_function="relu"):
    for i in range(num_iterations):
        forward_cache = forward_propagation(X, w1, b1, w2, b2, activation_function)
        backward_cache = backward_propagation(X, y, forward_cache["z1"], forward_cache["a1"],
                                              forward_cache["z2"], forward_cache["a2"], w2,
                                              derivative_activation_function)
        updated_parameters = update_parameters(w1, b1, w2, b2, backward_cache["dw1"], backward_cache["db1"],
                                               backward_cache["dw2"], backward_cache["db2"], learning_rate)
        w1 = updated_parameters["w1"]
        b1 = updated_parameters["b1"]
        w2 = updated_parameters["w2"]
        b2 = updated_parameters["b2"]
        if i % 100 == 0:
            y_pred = get_predictions(forward_cache["a2"])
            accuracy = get_accuracy(y_pred, y)
            print(f"Iteration {i}, accuracy: {accuracy}")
    return w1, b1, w2, b2


# TODO
def make_prediction(X, w1, b1, w2, b2):
    return

def test_model(X, y, w1, b1, w2, b2):
    return