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

len(inputs[0])

def time_rate(learning_rate, time):
  return 40*learning_rate / (40*learning_rate + time)

def model(input, label, weights, epochs, learning_rate, test_input, test_label):
  E = []
  E_test = []

  # Run the model for the number of epochs
  for i in range(epochs):


    # Iterate through all of the inputs and change the weights


    # for j in range(len(input)):
    random_input = random.randint(0, len(input)-1)

      # change(weights, input[j], label[j], (time_rate(learning_rate, i)))
    change(weights, input[random_input], label[random_input], learning_rate)
    if i % int(epochs/100) == 0:
      E.append(test(input, label, weights))
      E_test.append(test(test_input, test_label, weights))
  return E, E_test

def test(inputs, labels, weights):
  E = 0
  for i in range(len(inputs)):
    E +=  0.5*(network_output(weights, inputs[i]) - labels[i])**2
  return E / (len(inputs))

# Initialize the weights

initial_weights = []
for i in range(2):
  column = []
  for j in range(50):
    column.append(random.random())
  initial_weights.append(column)

learning_rate = 0.01
train_batch = 2000
test_batch = 1000
t_max = 8000
E, E_test = model(inputs[0:train_batch], labels[0:train_batch], initial_weights, t_max, learning_rate, inputs[train_batch:train_batch + test_batch], labels[train_batch:train_batch + test_batch])

initial_weights

plt.plot(np.arange(0, len(E)*t_max/100, t_max/100),E, label = "Training Error")
plt.plot(np.arange(0, len(E)*t_max/100, t_max/100),E_test, label = "Testing Error")
plt.xlabel("Time")
plt.ylabel("Error (E)")
# plt.title("Learning Rate: " + str(learning_rate))
plt.legend()
plt.show()

test(inputs[401:403], labels[401:403], initial_weights)

inputs[2100], labels[2100]
network_output(initial_weights, inputs[2100])

# plt.plot([x for x in labels[150:200]])
plt.plot([x for x in labels[425:450]])
plt.plot(l)
# plt.plot([-x for x in initial_weights[0]])
# plt.plot(initial_weights[1])

test(inputs[425:450], labels[425:450], initial_weights)
erro = []
for i in range(425, 450, 1):
  erro.append(0.5*(network_output(initial_weights, inputs[i]) - labels[i])**2)



erro

l = []
for i in range(425, 450,1):
  l.append(network_output(initial_weights, inputs[i]))

initi
