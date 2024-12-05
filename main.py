import numpy as np
from neural_network import MLP
import dataset
from loss import get_accuracy

# select the dataset to use

# digit recognizer dataset
# you can download it from Kaggle or other sources
# https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv
file_path = "./data/digit-recognizer/train/train.csv"

# fashion-mnist dataset
# this dataset won't be avaible in the Github Repository
# you can download it from Kaggle or other sources
# https://www.kaggle.com/datasets/zalando-research/fashionmnist
# file_path = "./data/fashion-mnist/train/fashion-mnist_train.csv"

# iris species dataset
# iris dataset is the modified version to fit the data preprocessing requirements
# the original iris dataset is not suitable for this task
# https://www.kaggle.com/datasets/uciml/iris
# file_path = "data/iris-species/Iris.csv"

train_data = dataset.load_data(file_path)
train_data = dataset.preprocess_data(train_data)
train_data = dataset.split_data(train_data)
# split the data into training and testing sets
X_train, y_train, X_valid, y_valid = train_data.values()

# Create the neural network and specify the learning rate and number of iterations
mlp = MLP(learning_rate=0.01, num_iterations=200)

# regular pairs for activation function and initialization method
# relu - he
# sigmoid, tanh - std

# add layers to the neural network
# need to specify the name of the layer for input, first hidden, and output layers
# others layers' name can be arbitrary

# the code only produce one figure at a time,
# so to reproduce all the figures in the report,
# you need to run the code multiple times and change the hyperparameters accordingly
mlp.add_layer("input_layer", len(X_train), activation_function="tanh", init_method="std")
mlp.add_layer("first_hidden_layer", 100, activation_function="tanh", init_method="std")
mlp.add_layer("second_hidden_layer", 100, activation_function="tanh", init_method="he")
# mlp.add_layer("third_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("forth_hidden_layer", 800, activation_function="relu", init_method="he")
# mlp.add_layer("second_hidden_layer", 200, activation_function="sigmoid", init_method="std")
mlp.add_layer("output_layer", len(np.unique(y_train)), activation_function="tanh", init_method="std")

mlp.print_layers()
# logging control the frequency of logging the training process
mlp.fit(X_train, y_train, X_valid, y_valid, logging=10)

# plot all the plots
mlp.plots()
