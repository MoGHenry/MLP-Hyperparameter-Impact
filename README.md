# ICSI536-ML-Project

# Project Overview

This project is focused on implementing and analyzing a custom multi-layer perceptron (MLP) neural network 
using Python's NumPy library. The primary objective is to study how varying neural network configurations—such as the 
number of layers, neurons, activation functions, and parameter initialization techniques—affect 
performance across different datasets. The project explores the interplay between architecture depth, 
activation functions, and dataset complexity to determine optimal configurations.

# How to Run
To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the numpy and matplotlib packages using pip.
3. run the `main.py` file.

# Reproduce Results in Report
To reproduce the results in the report, 
- choose the dataset you want to use (MNIST, Fashion-MNIST, or Iris)
- you will have to modify the hyperparameters in the `main.py` file. 
- To use the Iris dataset, you will need to modify the 'dataset.py' file for x_train and x_valid variables.
  - `data_train[1:n] / 255.` is for MNIST and Fashion-MNIST, but for Iris, you will need to modify it to `data_train[1:n]`.
  - You can also comment out the line 38,45 and then uncomment the line 39, 46 for using the Iris dataset.

# plots
Plots are located in the 'plots' folder
For naming convention, plots are named as follows structure:
- the first few letters of the dataset name is the activation function used
- the next few numbers represent the number of neurons for each hidden layers
- The last number that is 0.1 or 1 represents the learning rate used. If it is not specified, it is set to 0.01 as default.
- The last letter is the initialization technique used for the weights and biases. If it is not specified, it use he for ReLU and std for sigmoid and tanh.

For example
- RRR 200
  - it means the input layer uses ReLU activation function, the hidden layers use ReLU activation function, and the output layer uses ReLU activation function)
- SSSS 700 800 0.1 std
  - it means the input layer uses sigmoid activation function, the hidden layers use sigmoid activation function and has 700 and 800 neurons respectively, and the output layer uses sigmoid activation function with learning rate 0.1 and weights and biases initialized using the standard deviation method.
