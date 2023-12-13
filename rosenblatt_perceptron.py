# Robert Power
# s5332419
# Neural Networks and Computational Intelligence
# Assignment 1
# Rosenblatt Perceptron

import numpy as np
import generate_data as gd

class Perceptron:
    def __init__(self, P, N):
        """
        Initialize a Perceptron model for a given data P and N
        :param P: Number of data points
        :param N: Dimensionality of feature vectors
        """

        self.P = P
        self.N = N

        # Initialize Perceptron weights (weight vector)
        self.w = np.zeros(self.N)

    def fit(self, data, n_max):
        """
        Sequential perceptron training with the Rosenblatt algorithm
        :param data: List of tuples (feature vector, binary label)
        :param n_max: Maximum number of training epochs
        """

        epoch = 0

        while epoch < n_max:        # Outer loop

            index = 0   # Reset index after each epoch
            localPotentials = np.zeros(self.P)

            for instance in data:   # Inner loop

                feature_vector, label = instance

                # Determine the index of the current training example
                index += 1

                # Compute the local potential for given instance
                E = np.dot(self.w, feature_vector) * label
                localPotentials[index-1] = E

                print(f"Local Potential E({index}) for epoch {epoch+1}:", E)

                # Update the weight vector
                if E <= 0:
                    for i in range(len(self.w)):
                        self.w = self.w + feature_vector*label/self.N

            # Check for stopping condition (Local potential for every instance is greater than zero)
            stop = np.all(localPotentials > 0)
            if stop:
                print(f"The local potential for every instance in the given data set is greater than zero at epoch {epoch+1}")
                break

            # Increment epoch
            epoch += 1


# Test 'generate_data' function:
N = 20
alpha = 0.75
P = int(alpha * N)
epochs = 10

D = gd.artificial_data(P, N)

model = Perceptron(P, N)
model.fit(D, epochs)







