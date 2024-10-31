import numpy as np


# initialize w, b
def std_initialize_parameters(input_size, output_size):
    w = np.random.randn(output_size, input_size)
    b = np.zeros((output_size, 1))
    return w, b


def he_initialize_parameters(input_size, output_size):
    w = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
    b = np.zeros((output_size, 1))
    return w, b
