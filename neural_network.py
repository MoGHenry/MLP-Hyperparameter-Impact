import numpy as np
from activations import get_activation_functions
from layers import Layer
from init_parameters import std_initialize_parameters, he_initialize_parameters

# https://youtu.be/w8yWXqWQYmU?si=MXhI9EgsfYXMdshP&t=917


def get_init_method(init_method="std"):
    if init_method == "std":
        return std_initialize_parameters
    elif init_method == "he":
        return he_initialize_parameters


class MLP:
    def __init__(self, learning_rate, num_iterations, shuffle=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.layers_sizes = []
        self.layers = []
        self.w = [0]
        self.b = [0]
        self.a = [0]
        self.z = [0]
        self.dw = [0]
        self.db = [0]

    # TODO
    def fit(self, x, y):
        for i in range(self.num_iterations):
            current_layer = 1
            while current_layer < len(self.layers_sizes):
                a, z = self.layers[current_layer].forward_propagation(x, self.w[current_layer], self.b[current_layer])
                self.a.append(a)
                self.z.append(z)
                current_layer += 1

            while current_layer > 1:
                current_layer -= 1
                # TODO
                self.layers[current_layer].backward_propagation(self.a[current_layer], self.a[current_layer - 1],
                                                                self.z[current_layer], self.z[current_layer - 1])

        pass
        # self.parameters = self.std_initialize_parameters(input_size, hidden_size, output_size)
        # self.parameters = gradient_descent(x, y, self.parameters["w1"], self.parameters["b1"], self.parameters["w2"],
        #                                    self.parameters["b2"], self.learning_rate, self.num_iterations,
        #                                    self.activation_function, self.derivative_activation_function)

    # TODO
    def predict(self, x):
        pass
        # y_pred = make_prediction(x, self.parameters["w1"], self.parameters["b1"], self.parameters["w2"],
        #                          self.parameters["b2"], self.activation_function)
        # return y_pred

    def add_layer(self, neurons, activation_function: str = "relu", init_method="std"):
        # check if self.hidden_layers is empty
        if len(self.layers) == 0:
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].head = True
            self.layers_sizes.append(neurons)
        else:
            self.layers[-1].tail = False
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].tail = True
            self.layers_sizes.append(neurons)
            init_function = get_init_method(init_method)
            w, b = std_initialize_parameters(input_size=self.layers_sizes[-2], output_size=self.layers_sizes[-1])
            self.w.append(w)
            self.b.append(b)


# TODO
def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    updated_parameters = {
        "w1": w1 - learning_rate * dw1,
        "b1": b1 - learning_rate * db1,
        "w2": w2 - learning_rate * dw2,
        "b2": b2 - learning_rate * db2
    }
    return updated_parameters

# TODO
def get_predictions(a2):
    return np.argmax(a2, axis=0)

# TODO
def get_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# in each iteration
# forward propagation
# backward propagation
# update parameters
# print the accuracy and loss
# TODO change parameters to dictionary
def gradient_descent(x, y, w1, b1, w2, b2, learning_rate, num_iterations, activation_function
                     , derivative_activation_function="relu"):
    for i in range(num_iterations):
        forward_cache = forward_propagation(x, w1, b1, w2, b2, activation_function)
        backward_cache = backward_propagation(x, y, forward_cache["z1"], forward_cache["a1"],
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


def make_prediction(x, w1, b1, w2, b2, activation_function):
    _, _, _, a2 = forward_propagation(x, w1, b1, w2, b2, activation_function)
    y_pred = get_predictions(a2)
    return y_pred


# TODO
def test_model(X, y, w1, b1, w2, b2):
    return



# layers.py file

# import numpy as np
#
#
# def softmax(z):
#     return np.exp(z) / np.sum(np.exp(z), axis=0)
#
#
# def get_activation_function(self, activation="relu"):
#     activation_function, derivative_activation_function = get_activation_functions(activation)
#     return activation_function, derivative_activation_function
#
#
# # one-hot encoding
# # num_classes is the number of unique labels in the output
# def one_hot(y, num_classes):
#     return np.eye(num_classes)[y.reshape(-1)]
#
#
# class Layer:
#     def __init__(self, neurons, activation_function):
#         self.head = False
#         self.tail = False
#         self.neurons = neurons
#         self.activation_function, self.derivative_activation_function = get_activation_function(activation_function)
#
#     def forward_propagation(self, x, w1, b1, a0=None):
#         if self.head:
#             z1 = np.dot(w1, x) + b1
#             a1 = self.activation_function(z1)
#             return a1, z1
#         elif self.tail:
#             z1 = np.dot(w1, a0) + b1
#             y = softmax(z1)
#             return y, z1
#         else:
#             z1 = np.dot(w1, a0) + b1
#             a1 = self.activation_function(z1)
#             return a1, z1
#
#     # backward propagation
#     # calculate dw1, db1, dw2, db2
#     def backward_propagation(self, x, a0, a1, y=None, dz1=None, z1=None, w2=None, dz2=None):
#         if self.tail:
#             true_labels = one_hot(y, a1.shape[0])
#             dz1 = a1 - true_labels
#             dw1 = np.dot(dz1, a0.T)
#             db1 = np.sum(dz1, axis=1, keepdims=True)
#             return dw1, db1
#         elif self.head:
#             dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
#             dw1 = np.dot(dz1, x.T)
#             db1 = np.sum(dz1, axis=1, keepdims=True)
#             return dw1, db1
#         else:
#             dz1 = np.dot(w2.T, dz2) * self.derivative_activation_function(z1)
#             dw1 = np.dot(dz1, a0.T)
#             db1 = np.sum(dz1, axis=1, keepdims=True)
#             return dw1, db1
