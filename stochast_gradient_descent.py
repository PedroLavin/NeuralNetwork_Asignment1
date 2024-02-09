import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class FFNN:

    def __init__(self, N, K=2, learning_rate=0.05):
        """
        Initialize a shallow feedforward neural network for a regression problem.
        The FFNN therefore only has one output unit.
        :param N: The dimensionality of the input features (feature vectors)
        :param K: The number of units in the hidden layer. Default is 2
        :param learning_rate: The learning rate for SGD
        """
        self.N = N
        self.K = K
        self.learning_rate = learning_rate
        self.w1 = self.initialize_w1(K, N)
        self.w2 = self.initialize_w2(K)

    def initialize_w1(self, N, K):
        """
        Function to initialize the input-to-hidden weights such that the weights are independent random vectors
        with |w1|^2 = 1
        :param N: Dimensionality of the input feature vectors
        :param K: The number of hidden units
        :return: initialized input-hidden weight matrix (N, K)
        """
        weights = np.random.randn(N, K)
        normalized_weights = weights / np.sqrt(np.sum(weights**2, axis=1, keepdims=True))
        return normalized_weights

    def initialize_w2(self, K):
        """
        Function to initialize the hidden-to-output weights such that the weights are independent random vectors
        with |w2|^2 = 1
        :param K: The number of hidden units
        :return: initialized hidden-output weight vector (K)
        """
        weights = np.random.randn(K)
        normalized_weights = weights / np.linalg.norm(weights)
        return normalized_weights

    def forward_pass(self, X):
        """
        Calculates the predicted value of our FFNN for a given input
        :param X: input vector (feature vector)
        :return: Predicted value
        """
        z = np.dot(self.w1, X)
        a = np.tanh(z)
        output = np.dot(self.w2, a)
        return output

    def fit(self, X_train, y_train, X_test, y_test, tmax):
        """
        The model is fit using the stochastic gradient descent algorithm. Only the input-hidden weights are updated
        :param X: matrix of input/feature vectors in our training set
        :param y: actual output for the corresponding input/feature vectors in our training set
        :param tmax: maximum number of iterations for SGD to perform (stopping criterion)
        :param P:
        :return:
        """
        N = X_train.shape[1]  # Number of examples
        E_history = []
        Etest_history = []

        for t in range(tmax):
            # Randomly select one example
            idx = np.random.randint(N)
            X_sample = X_train[idx, :]
            y_sample = y_train[idx]

            # Forward pass
            z = np.dot(self.w1, X_sample)
            a = np.tanh(z)
            output = np.dot(self.w2, a)

            # Compute error
            error = output - y_sample

            # Backpropagation
            grad_w1 = np.outer((1 - np.square(a)) * self.w2 * error, X_sample)

            # Update weights
            self.w1 -= self.learning_rate * grad_w1

            # Compute E (training error)
            E = self.evaluate(X_train, y_train)
            E_history.append(E)

            # Compute Etest (testing error)
            Etest = self.evaluate(X_test, y_test)
            Etest_history.append(Etest)

        return E_history, Etest_history

    def evaluate(self, X, y):
        P = X.shape[1]
        total_error = 0
        for i in range(P):
            output = self.forward_pass(X[i, :])
            total_error += (output - y[i]) ** 2
        return total_error / (2 * P)


def run_simulations(num_runs, P, Q, K, lr):
    # Test
    x = np.array((pd.read_csv("C:/MSc Artificial Intelligence/Year 1 Semester Ib/Neural Networks and Computational Intelligence/Assignment 3/Training_Data/xi.csv", header=None)).transpose())
    y = np.array(pd.read_csv("C:/MSc Artificial Intelligence/Year 1 Semester Ib/Neural Networks and Computational Intelligence/Assignment 3/Training_Data/tau.csv", header=None))[0]

    # Create new training and test sets containing only the first 100 samples
    x_train = x[:P]
    y_train = y[:P]
    x_test = x[P:P+Q]
    y_test = y[P:P+Q]

    N = x_train.shape[1]
    t = x_train.shape[0] * 10

    # Perform simulations over the specified number of independent runs
    E_avg_history = np.zeros(t)
    Etest_avg_history = np.zeros(t)

    for run in range(num_runs):
        # Train the model
        model = FFNN(N, K, lr)
        E_history, Etest_history = model.fit(x_train, y_train, x_test, y_test, t)

        # Accumulate error histories
        E_avg_history += np.array(E_history)
        Etest_avg_history += np.array(Etest_history)

    # Average the error histories
    E_avg_history /= num_runs
    Etest_avg_history /= num_runs

    # Plot averaged E and Etest
    plt.plot(range(0, len(E_avg_history) * 100, 100), E_avg_history, label='Train Error')
    plt.plot(range(0, len(Etest_avg_history) * 100, 100), Etest_avg_history, label='Test Error')
    plt.xlabel('t')
    plt.ylabel('MSE')
    plt.title(f'P={P};  Q={Q};  K={K};  eta={lr}')
    plt.legend()
    plt.show()

    # Display final weight vectors (Note: These will be from the last run)
    print("Final weight vector w1:")
    print(model.w1)
    print(model.w1.shape)
    print("Final weight vector w2:")
    print(model.w2)
    print(model.w2.shape)


# Dataset Hyperparameters
P = 100
Q = 100
runs = 2

# Model Hyperparameters
K = 2
lr = 0.05

run_simulations(runs, P, Q, K, lr)
