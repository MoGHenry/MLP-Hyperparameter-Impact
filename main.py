import numpy as np
from neural_network import MLP
import dataset

data = dataset.load_data("./data/digit_recognizer/train.csv/train.csv")
data = dataset.preprocess_data(data)
data = dataset.split_data(data)
# split the data into training and testing sets
X_train, y_train, X_test, y_test = data.values()


# create a neural network with 2 hidden layers and 1 output layer
mlp = MLP(learning_rate=0.1, num_iterations=200)

# add layers to the neural network
mlp.add_layer("input_layer", len(X_train), activation_function="relu", init_method="he")
mlp.add_layer("first_hidden_layer", 10, init_method="he")
# mlp.add_layer("second_hidden_layer", 200)
mlp.add_layer("output_layer", len(np.unique(y_train)))

mlp.print_layers()
# breakpoint()
mlp.fit(X_train, y_train)
