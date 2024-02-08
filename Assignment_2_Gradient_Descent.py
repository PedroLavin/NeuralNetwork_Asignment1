import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import pandas as pd

# Example Inputs and Weights

input = [1, 0, 5, 6]
wei = [[0.5, 0.2, 0.1, 0.2], [0.5, 0.1, 0.3, 0.1]]




def network_output(wei, input):
  #Multiply the Weiights with the Inputs, apply the activation function and sum the result
  output = 0
  for i in range(len(wei)):
    output += np.tanh(np.dot(wei[i], input))
  return output


def change(wei, input, label, learning):

  # Calculate the difference between predicted value and real value (label)
  # Then, calculate the derivative of the activation function and update the respective weights

  for i in range(len(wei)):

    error = network_output(wei, input) - label
    derivative = (1 - np.tanh(np.dot(wei[i], input))**2)

    # Update the weights
    for j in range(len(wei[i])):
      wei[i][j] = wei[i][j] - error*derivative*learning*input[j]


label = 1
learning_rate = 0.05
evolution = []
for i in range(20):
  change(wei, input, label, learning_rate)
  evolution.append(network_output(wei, input))

plt.plot(evolution, label ="Predicted Label")
plt.axhline(y = label, color = "r", label = "Correct Label")
plt.legend()
plt.show()

inputs = np.array((pd.read_csv("xi.csv", header = None)).transpose())
labels = np.array(pd.read_csv("tau.csv", header = None))[0]



def model(input, label, weights, epochs, learning_rate, test_input, test_label):
  E = []
  E_test = []

  # Run the model for the number of epochs
  for i in range(epochs):


    # Iterate through all of the inputs and change the weights
    for j in range(len(input)):
      change(weights, input[j], label[j], learning_rate)

    E.append(test(input, label, weights))
    E_test.append(test(test_input, test_label, weights))
  return E, E_test

def test(inputs, labels, weights):
  E = 0
  for i in range(len(inputs)):
    E += 0.5 * (network_output(weights, inputs[i]) - labels[i])**2
  return E / len(inputs)

# Initialize the weights

initial_weights = []
for i in range(2):
  column = []
  for j in range(len(inputs[0])):
    column.append(random.random())
  initial_weights.append(column)

E, E_test = model(inputs[0:100], labels[0:100], initial_weights, 100, 0.9, inputs[100:500], labels[100:500])

plt.plot(E, label = "Training Error")
plt.plot(E_test, label = "Testing Error")
plt.xlabel("Epochs")
plt.ylabel("Error (E)")
plt.legend()
plt.show()
