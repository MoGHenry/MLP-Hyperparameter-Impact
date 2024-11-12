import numpy as np
from activations import get_activation_functions
from layers import Layer
from init_parameters import std_initialize_parameters, he_initialize_parameters
from loss import cross_entropy_loss, one_hot

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
        self.dz = [0]

    # TODO
    def fit(self, x, y):
        num_layers = len(self.layers)
        self.a = [0] * num_layers  # Initialize activations list
        self.z = [0] * num_layers  # Initialize z values list
        self.dw = [0] * num_layers  # Initialize gradients for weights
        self.db = [0] * num_layers  # Initialize gradients for biases
        self.dz = [0] * num_layers  # Initialize gradients for dz values

        for i in range(self.num_iterations):
            # Forward propagation through each layer
            current_input = x  # Initial input for the first layer
            self.a[0] = current_input  # Store the activation for the input layer

            for current_layer in range(1, num_layers):
                if current_layer == 1:  # If it's the first hidden layer
                    a, z = self.layers[current_layer - 1].forward_propagation(
                        self.w[current_layer], self.b[current_layer], x=current_input
                    )
                else:  # For subsequent layers
                    a, z = self.layers[current_layer - 1].forward_propagation(
                        self.w[current_layer], self.b[current_layer], a0=current_input
                    )

                # Store activations and linear transformations
                self.a[current_layer] = a
                self.z[current_layer] = z
                current_input = a  # Update input for the next layer

            # Backward propagation through each layer in reverse order
            for current_layer in reversed(range(1, num_layers)):
                if current_layer == num_layers - 1:
                    dw, db, dz = self.layers[current_layer].backward_propagation(
                        x=None, a0=self.a[current_layer - 1], a1=self.a[current_layer], y=y
                    )
                else:
                    dw, db, dz = self.layers[current_layer].backward_propagation(
                        x=None,
                        a0=self.a[current_layer - 1],
                        a1=self.a[current_layer],
                        z1=self.z[current_layer],
                        w2=self.w[current_layer + 1],
                        dz2=self.dz[current_layer + 1]
                    )

                self.dw[current_layer] = dw
                self.db[current_layer] = db
                self.dz[current_layer] = dz

            # Update weights and biases
            for current_layer in range(1, num_layers):
                self.w[current_layer] -= self.learning_rate * self.dw[current_layer]
                self.b[current_layer] -= self.learning_rate * self.db[current_layer]

            # Optionally print progress
            if i % 2 == 0 or i == self.num_iterations - 1:
                y_pred = self.predict(x)
                loss = cross_entropy_loss(one_hot(y, self.layers_sizes[-1]), y_pred)
                accuracy = get_accuracy(y_pred, y)
                print(f"Iteration {i + 1}/{self.num_iterations}, Accuracy: {accuracy:.4f}")

    # TODO
    def predict(self, x):
        current_input = x
        num_layers = len(self.layers)

        for current_layer in range(num_layers):
            # Adjust indexing logic if self.w and self.b start at index 1 or 0
            if current_layer == 0:  # First layer (input layer)
                a, _ = self.layers[current_layer].forward_propagation(
                    self.w[current_layer], self.b[current_layer], x=current_input
                )
            else:  # Subsequent layers
                a, _ = self.layers[current_layer].forward_propagation(
                    self.w[current_layer], self.b[current_layer], a0=current_input
                )
            current_input = a  # Update input for the next layer

        # Get predictions from the output of the last layer (softmax probabilities)
        y_pred = np.argmax(current_input, axis=0)  # Returns the index of the max value along columns (samples)
        return y_pred

    def add_layer(self, neurons, activation_function: str = "relu", init_method: str = "std"):
        # Check if no layers have been added yet
        if len(self.layers) == 0:
            # Add the first layer and set it as the head
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].input_layer = True  # Set the head attribute to True for the first layer
            self.layers_sizes.append(neurons)
        else:
            # Add subsequent layers
            self.layers[-1].output_layer = False  # Ensure the previous layer is not marked as the tail
            self.layers.append(Layer(neurons=neurons, activation_function=activation_function))
            self.layers[-1].output_layer = True  # Mark the current layer as the tail
            self.layers_sizes.append(neurons)

            # Initialize weights and biases for the new layer
            init_function = get_init_method(init_method)
            w, b = init_function(input_size=self.layers_sizes[-2], output_size=self.layers_sizes[-1])
            self.w.append(w)
            self.b.append(b)

    def update_parameters(self):
        for current_layer in range(1, len(self.layers)):
            self.w[current_layer] -= self.learning_rate * self.dw[current_layer]
            self.b[current_layer] -= self.learning_rate * self.db[current_layer]

    def get_predictions(self, x):
        current_input = x
        for current_layer in range(1, len(self.layers)):
            a, _ = self.layers[current_layer].forward_propagation(
                self.w[current_layer], self.b[current_layer], current_input
            )
            current_input = a  # Update input for the next layer

        # Get predictions from the output of the last layer (softmax probabilities)
        y_pred = np.argmax(current_input, axis=0)  # Returns the index of the max value along columns (samples)
        return y_pred

    def print_layers(self):
        for i in range(len(self.layers)):
            print(f"Layer {i+1}: {self.layers[i].neurons} neurons, {self.layers[i].print_activation_function()}")


# TODO
def get_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# # in each iteration
# # forward propagation
# # backward propagation
# # update parameters
# # print the accuracy and loss
# # TODO change parameters to dictionary
# def gradient_descent(x, y, w1, b1, w2, b2, learning_rate, num_iterations, activation_function
#                      , derivative_activation_function="relu"):
#     for i in range(num_iterations):
#         forward_cache = forward_propagation(x, w1, b1, w2, b2, activation_function)
#         backward_cache = backward_propagation(x, y, forward_cache["z1"], forward_cache["a1"],
#                                               forward_cache["z2"], forward_cache["a2"], w2,
#                                               derivative_activation_function)
#         updated_parameters = update_parameters(w1, b1, w2, b2, backward_cache["dw1"], backward_cache["db1"],
#                                                backward_cache["dw2"], backward_cache["db2"], learning_rate)
#         w1 = updated_parameters["w1"]
#         b1 = updated_parameters["b1"]
#         w2 = updated_parameters["w2"]
#         b2 = updated_parameters["b2"]
#         if i % 100 == 0:
#             y_pred = get_predictions(forward_cache["a2"])
#             accuracy = get_accuracy(y_pred, y)
#             print(f"Iteration {i}, accuracy: {accuracy}")
#     return w1, b1, w2, b2
#
#
# def make_prediction(x, w1, b1, w2, b2, activation_function):
#     _, _, _, a2 = forward_propagation(x, w1, b1, w2, b2, activation_function)
#     y_pred = get_predictions(a2)
#     return y_pred
#
#
# # TODO
# def test_model(X, y, w1, b1, w2, b2):
#     return



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
