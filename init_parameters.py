import numpy as np


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
    return w,b


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