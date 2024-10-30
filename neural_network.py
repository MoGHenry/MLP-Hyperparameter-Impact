import numpy as np
from activations import get_activation_functions
from layers import Layer
from init_parameters import std_initialize_parameters, he_initialize_parameters

# https://youtu.be/w8yWXqWQYmU?si=MXhI9EgsfYXMdshP&t=917


class MLP:
    def __init__(self, learning_rate, num_iterations, shuffle=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.hidden_layers_sizes = []
        self.layers = []
        self.w = []
        self.b = []

    def fit(self, x, y):
        pass
        # self.parameters = self.std_initialize_parameters(input_size, hidden_size, output_size)
        # self.parameters = gradient_descent(x, y, self.parameters["w1"], self.parameters["b1"], self.parameters["w2"],
        #                                    self.parameters["b2"], self.learning_rate, self.num_iterations,
        #                                    self.activation_function, self.derivative_activation_function)

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
            w, b = std_initialize_parameters(self.layers[-1].input_size, self.layers[-1].output_size)
            self.w.append(w)
            self.b.append(b)
        else:
            self.layers[-1].tail = False
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].tail = True
            self.hidden_layers_sizes.append(neurons)

    def get_init_method(self, init_method="std"):
        if init_method == "std":
            return self.std_initialize_parameters
        elif init_method == "he":
            return self.he_initialize_parameters





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
