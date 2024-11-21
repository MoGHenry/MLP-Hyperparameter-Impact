import numpy as np
from neural_network import MLP
import dataset
from loss import get_accuracy

train_data = dataset.load_data("./data/digit_recognizer/train.csv/train.csv")
train_data = dataset.preprocess_data(train_data)
train_data = dataset.split_data(train_data)
# split the data into training and testing sets
X_train, y_train, X_valid, y_valid = train_data.values()

# create a neural network with 2 hidden layers and 1 output layer
mlp = MLP(learning_rate=0.1, num_iterations=200)

# add layers to the neural network
# need to specify the name of the layer for input, first hidden, and output layers
# others layers' name can be arbitrary
# relu - he
# sigmoid, tanh - std
# breakpoint()
mlp.add_layer("input_layer", len(X_train), activation_function="sigmoid", init_method="std")
mlp.add_layer("first_hidden_layer", 800, activation_function="sigmoid", init_method="std")
mlp.add_layer("second_hidden_layer", 800, activation_function="sigmoid", init_method="std")
# mlp.add_layer("third_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("second_hidden_layer", 200, activation_function="sigmoid", init_method="std")
mlp.add_layer("output_layer", len(np.unique(y_train)), activation_function="sigmoid", init_method="std")

mlp.print_layers()
# breakpoint()
# logging is the iteration number to print the loss and accuracy
mlp.fit(X_train, y_train, X_valid, y_valid, logging=10)

# either plot the specific plot
# mlp.plot_accuracy()
# mlp.plot_loss()
# mlp.plot_accuracy_vs_loss()

# or plot all the plots
mlp.plots()

# TODO
# If the training accuracy is significantly higher than both your validation, it suggests overfitting

# Overfitting is usually indicated by a training loss that continues to decrease while the validation loss plateaus
# or starts increasing
