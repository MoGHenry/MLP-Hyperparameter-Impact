import numpy as np
from activations import get_activation_functions
from layers import Layer
from init_parameters import std_initialize_parameters, he_initialize_parameters
from loss import cross_entropy_loss, one_hot, get_accuracy
from layers import softmax
from plotting import plotting, plot_accuracy_loss, plot_all

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
        self.accuracy_train = {}
        self.loss_train = {}
        self.accuracy_valid = {}
        self.loss_valid = {}

    # TODO
    def fit(self, x_train, y_train, x_valid, y_valid, logging=1):
        # breakpoint()
        num_layers = len(self.layers)
        self.a = [0] * num_layers  # Initialize activations list
        self.z = [0] * num_layers  # Initialize z values list
        self.dw = [0] * num_layers  # Initialize gradients for weights
        self.db = [0] * num_layers  # Initialize gradients for biases
        self.dz = [0] * num_layers  # Initialize gradients for dz values

        for i in range(self.num_iterations):
            # Forward propagation through each layer
            current_input = x_train  # Initial input for the first layer
            self.a[0] = current_input  # Store the activation for the input layer
            # breakpoint()
            for current_layer in range(1, num_layers):
                # breakpoint()
                if current_layer == 1:  # If it's the first hidden layer
                    a, z = self.layers[current_layer].forward_propagation(
                        self.w[current_layer], self.b[current_layer], x=current_input
                    )
                else:  # For subsequent layers
                    a, z = self.layers[current_layer].forward_propagation(
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
                        x=x_train, a0=self.a[current_layer - 1], a1=self.a[current_layer], y=y_train
                    )
                else:
                    dw, db, dz = self.layers[current_layer].backward_propagation(
                        x=x_train,
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
                # breakpoint()
                self.w[current_layer] -= self.learning_rate * self.dw[current_layer]
                self.b[current_layer] -= self.learning_rate * self.db[current_layer]
                # print(sum(sum(self.w[current_layer])))
                # breakpoint()

            # print progress every 2 iterations or at the end of the loop
            if i % logging == 0 or i == self.num_iterations - 1:
                y_pred_train = self.predict(x_train)
                y_pred_valid = self.predict(x_valid)
                accuracy_train = get_accuracy(y_pred_train, y_train)
                accuracy_valid = get_accuracy(y_pred_valid, y_valid)
                loss_train = cross_entropy_loss(one_hot(y_train, self.layers_sizes[-1]), y_pred_train)
                loss_valid = cross_entropy_loss(one_hot(y_valid, self.layers_sizes[-1]), y_pred_valid)
                print(f"Iteration {i + 1}/{self.num_iterations}, Accuracy(train): {accuracy_train:.4f}"
                      f", Loss(train): {loss_train:.4f}")
                print(f"Iteration {i + 1}/{self.num_iterations}, Accuracy(valid): {accuracy_valid:.4f}"
                      f", Loss(valid): {loss_valid:.4f}")
                self.accuracy_train[i + 1] = accuracy_train
                self.loss_train[i + 1] = loss_train
                self.accuracy_valid[i + 1] = accuracy_valid
                self.loss_valid[i + 1] = loss_valid

    def add_layer(self, layer_name: str, neurons, activation_function: str = "relu", init_method: str = "he"):
        # Check if no layers have been added yet
        if len(self.layers) == 0:
            # Add the first layer and set it as the head
            self.layers.append(Layer(layer_name=layer_name, neurons=neurons, activation_function=activation_function))
            self.layers[-1].input_layer = True
            self.layers_sizes.append(neurons)
        else:
            # Add subsequent layers
            self.layers[-1].output_layer = False
            self.layers.append(Layer(layer_name=layer_name, neurons=neurons, activation_function=activation_function))
            self.layers[-1].output_layer = True
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

    def predict(self, x):
        current_input = x
        num_layers = len(self.layers)
        for current_layer in range(1, num_layers):
            # breakpoint()
            if current_layer == 1:
                a, z = self.layers[current_layer].forward_propagation(
                    self.w[current_layer], self.b[current_layer], x=current_input
                )
            else:  # For subsequent layers
                a, z = self.layers[current_layer].forward_propagation(
                    self.w[current_layer], self.b[current_layer], a0=current_input
                )

            # Store activations and linear transformations
            current_input = a  # Update input for the next layer
        # Get predictions from the output of the last layer (softmax probabilities)
        y_pred = np.argmax(current_input, axis=0)  # Returns the index of the max value along columns (samples)
        return y_pred

    def plot_accuracy(self):
        plotting(data=[self.accuracy_train, self.accuracy_valid], x_label="Iteration", y_label="Accuracy",
                 title="Accuracy vs Iteration", legend_labels=["Training", "Validation"])

    def plot_loss(self):
        plotting(data=[self.loss_train, self.loss_valid], x_label="Iteration", y_label="Loss",
                 title="Loss vs Iteration", legend_labels=["Training", "Validation"])

    def plot_accuracy_loss(self):
        plot_accuracy_loss(accuracy=self.accuracy_train, loss=self.loss_train, x_label="Iteration", y_label="Value")

    def plots(self):
        plot_all(accuracy=[self.accuracy_train, self.accuracy_valid],
                 loss=[self.loss_train, self.loss_valid], x_label="Iteration", y_label="Value"
                 , legend_labels=["Training", "Validation"])

    def print_layers(self):
        for i in range(len(self.layers)):
            print(f"Layer {i + 1}: {self.layers[i].neurons} neurons, {self.layers[i].print_activation_function()}")





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
