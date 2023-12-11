import numpy as np
from scipy import stats
import random

P = 5 # Number of vectors
N = 2 # Dimension of each of the vectors
mu = 0 # Mean of the vectors elements
sigma = 1 # Variance of the vectors elements

# Generate the Dataset, with random labels (+1,-1)

dataset = []

for i in range(P):
  label = random.choice([-1,1])
  vector = list(np.random.normal(mu, sigma, N))
  dataset.append((vector, label))

# Initializing the perceptron vector
w = np.zeros((N,))

# Function to calculate the local potential E
def local_potential(w, vector, label):

  return np.dot(w, vector) * label

# Function that updates the weights according to the Rosenblatts' formula.
def update_weights(w, vector, label):

  for i in range(len(w)):
    w[i] = w[i] + vector[i]*label/5
  
# Function that indicates the number of correctly classified datapoints
def evaluate_perceptron(w, data):
  counter = 0
  for i in range(P):
    if np.sign(np.dot(w, data[i][0])) == data[i][1]:
      counter += 1
  return counter

# Define the Rosenblatt's perceptron
def perceptron(w, dataset):

  for j in range(10000):

    for i in range(P):

      vector = dataset[i][0]
      label = dataset[i][1]

      # If the E (local potential) is below 0, it means the vector is incorrectly classified and the weights need to be updated.
      if local_potential(w, vector, label) <= 0:
        update_weights(w, vector, label)

      # print(evaluate_perceptron(w, dataset))

  return w




perceptron(w, dataset)

evaluate_perceptron(w, dataset)
