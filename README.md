# ICSI536-ML-Project

This project focuses on exploring the performance of Multi-Layer Perceptrons (MLPs) on different datasets with varying configurations. The implementation is entirely custom-built using the Numpy library without reliance on pre-existing neural network frameworks.

## Table of Contents
1. [Project Overview](#project-overview)
  - [Motivation](#motivation)
  - [Objective](#objective)
  - [Datasets](#datasets)
2. [Methodology](#methodology)
3. [Results](#results)
  - [MNIST](#mnist)
  - [Fashion MNIST](#fashion-mnist)
  - [Iris](#iris)
4. [Key Takeaways](#key-takeaways)
5. [Future Work](#future-work)
6. [How to Run the Code](#how-to-run)
7. [Authors](#authors)
8. [License](#license)

## Project Overview

### Motivation
MLPs are versatile neural network architectures capable of solving complex problems such as classification, regression, and time-series prediction. Despite their computational expense and potential for overfitting, they remain a foundational model in machine learning, especially for structured data.

### Objective
The objective of this project is to investigate how varying MLP configurations, including the number of layers, number of neurons, activation functions, and initialization techniques, impact performance on datasets of varying complexity.

### Datasets
Three datasets were used for evaluation:
- **MNIST**: A standard benchmark dataset for digit recognition.
- **Fashion MNIST**: A more complex dataset involving detailed textures and patterns.
- **Iris**: A small dataset with only four features and three labels, used to study the behavior of MLPs on simpler data.

## Methodology
1. **Custom MLP Implementation**: The neural network was built from scratch using Numpy, without pre-existing libraries.
2. **Configuration Variations**:
  - Activation Functions: ReLU, Sigmoid, Tanh.
  - Initialization Techniques: Standard (STD) and He initialization.
  - Number of Layers: 1 to 6.
  - Number of Neurons per Layer: 10 to 1200.
3. **Evaluation Metrics**:
  - Accuracy.
  - Loss.
  - Overfitting behavior.

## Results
### MNIST
- **ReLU**: Performed best overall, showing high accuracy without overfitting.
- **Sigmoid and Tanh**: Marginally lower accuracy; Tanh occasionally exhibited overfitting.

### Fashion MNIST
- **ReLU**: Continued to perform well but required more layers and neurons for better accuracy.
- **Sigmoid and Tanh**: Prone to overfitting, mitigated by He initialization.

### Iris
- **ReLU**: Less effective due to sparse activations on a low-dimensional dataset.
- **Sigmoid and Tanh**: Performed better due to their smooth and bounded behavior, well-suited for small datasets.

## Key Takeaways
1. ReLU activation is highly effective for large datasets and deep architectures but struggles with small datasets.
2. Sigmoid and Tanh are better suited for small datasets, providing smoother convergence.
3. Increasing the number of neurons and layers generally improves accuracy but comes at the cost of computational efficiency and risk of overfitting.
4. He initialization helps mitigate overfitting in deeper architectures.

## Future Work
1. Incorporate convolutional neural networks (CNNs) for image-based datasets.
2. Experiment with advanced regularization techniques to address overfitting.
3. Explore alternative activation functions to enhance performance further.

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

## Authors
- Mamadou A Diallo
- John Kaminski
- Ugochukwu B Okoro
- Henry Qui
- Abhishek Santhakumar

## License
This project is licensed under the MIT License.
