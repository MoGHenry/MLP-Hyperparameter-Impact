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
    np.random.shuffle(data)

    # Determine the index for splitting the data
    split_index = int(size * data.shape[0])

    # Split the data into training and testing sets
    train_data = data[:split_index, :].T
    test_data = data[split_index:, :].T

    # The first row contains the labels, and the rest are features
    train_labels = train_data[0].reshape(1, -1)
    validation_labels = test_data[0].reshape(1, -1)
    train_features = train_data[1:]
    validation_features = test_data[1:]

    training_data = {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": validation_features,
        "test_labels": validation_labels
    }

    return training_data
