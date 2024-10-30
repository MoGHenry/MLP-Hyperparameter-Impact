import numpy as np
from activations import get_activation_functions
from layers import Layer


# https://youtu.be/w8yWXqWQYmU?si=MXhI9EgsfYXMdshP&t=917


class MLP:
    def __init__(self, hidden_layer_sizes: list, learning_rate, num_iterations,
                 activation_function="relu", shuffle=True, init_method="std"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.init_method = self.get_init_method(init_method)
        self.activation_function, self.derivative_activation_function = self.get_activation_function(
            activation_function)
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

    def add_layer(self, neurons, activation_function):
        # check if self.hidden_layers is empty
        if len(self.layers) == 0:
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].head = True
        else:
            self.layers[-1].tail = False
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].tail = True

    def get_init_method(self, init_method="std"):
        if init_method == "std":
            return self.std_initialize_parameters
        elif init_method == "he":
            return self.he_initialize_parameters

    def get_activation_function(self, activation="relu"):
        activation_function, derivative_activation_function = get_activation_functions(activation)
        return activation_function, derivative_activation_function

    # initialize w1,b1,w2,b2
    # w1 shape (input_size, hidden_size)
    # b1 shape (1, hidden_size)
    # w2 shape (hidden_size, output_size)
    # b2 shape (1, output_size)
    def std_initialize_parameters(self, input_size, hidden_size, output_size):
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

    def he_initialize_parameters(self, input_size, hidden_size, output_size):
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
