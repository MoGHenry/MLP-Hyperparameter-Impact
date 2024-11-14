import numpy as np
from neural_network import MLP
import dataset

train_data = dataset.load_data("./data/digit_recognizer/train.csv/train.csv")
train_data = dataset.preprocess_data(train_data)
train_data = dataset.split_data(train_data)
# split the data into training and testing sets
X_train, y_train, X_valid, y_valid = train_data.values()


test_data = dataset.load_data("./data/digit_recognizer/test.csv/test.csv")
test_data = dataset.preprocess_data(test_data)
X_test, y_test = dataset.test_dataset(test_data)

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

# TODO
# evaluate the model on the validation set
# evaluate the model on the test set
# calculate the overfitting
# plot the learning curve, accuracy vs iteration, loss vs iteration
