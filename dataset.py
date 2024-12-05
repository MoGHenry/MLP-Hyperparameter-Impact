import pandas as pd
import numpy as np


# https://youtu.be/w8yWXqWQYmU?si=MOIwj0waNpEunSJS&t=709


# load the data from csv file as pd.DataFrame
def load_data(path: str) -> pd.DataFrame:
    # read the csv file
    data = pd.read_csv(path)
    return data


# convert to numpy array
def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    # convert to numpy array
    data = np.array(data)
    return data


# split the data into training and testing sets
# first col is the label, rest are features
def split_data(data: np.ndarray, size: float = 0.8) -> dict:
    # Shuffle the data
    np.random.seed(42)
    np.random.shuffle(data)
    # breakpoint()
    m, n = data.shape
    # Determine the index for splitting the data
    split_index = m - int(size * m)

    data_train = data[split_index:, :].T
    # Split the data into training and testing sets
    y_train = data_train[0]
    # For Iris dataset, we need to comments out the following line
    # and then uncomment the next line to normalize the data
    x_train = data_train[1:n] / 255.
    # x_train = data_train[1:n]

    data_test = data[:split_index, :].T
    y_valid = data_test[0]
    # For Iris dataset, we need to comments out the following line
    # and then uncomment the next line to normalize the data
    x_valid = data_test[1:n] / 255.
    # x_valid = data_test[1:n]

    training_data = {
        "train_features": x_train,
        "train_labels": y_train.astype(int),
        "valid_features": x_valid,
        "valid_labels": y_valid.astype(int)
    }

    return training_data


# def test_dataset(data: np.ndarray):
#     m, n = data.shape
#     dataset = data.T
#     x_test = dataset[1:n]
#     y_test = dataset[0]
#     x_test = x_test / 255.
#     return x_test, y_test
