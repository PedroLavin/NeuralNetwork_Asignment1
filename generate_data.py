# Robert Power
# s5332419
# Neural Networks and Computational Intelligence
# Assignment 1
# Data Generation

import numpy as np

def artificial_data(P, N):
    """
    Generates an artificial dat set with P randomly generated N-dimensional feature vectors and
    binary labels.

    :param P: Number of data points
    :param N: Dimensionality of feature vectors

    :return: List of tuples (feature vector, binary label)
    """

    data = []

    for _ in range(P):
        # Generate an N-dimensional feature vector with Gaussian component
        feature_vector = np.random.normal(0, 1, N)

        # Generate a binary label corresponding to the generated feature vector
        label = np.random.choice([-1, 1])

        data.append((feature_vector, label))

    return data

