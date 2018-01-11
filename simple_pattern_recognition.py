"""
This is a example of simple neural network which can predict a output
of pattern without actually knowing it.

Pattern:
0 1 0 = 1
0 1 1 = 1
1 0 0 = 1
We can see if the 2nd input is 1 output is 1. But we will not tell it to
the network.

For input a,b,c there is output = weight1 * a + weight2 * b + weight3 * c...(1)
These weights will be adjusted during the training period
"""
from numpy import random, dot, array, exp  # we will only use the numpy library


class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((3, 1)) - 1  # initializing the weights with random number between 0 & 1

    def __sigmoid(self, x):  # sigmoid function is used to normalize the input (between 0 & 1)
        return 1 / (1 + exp(-x))

    def think(self, inputs):  # returns the output by using the formula 1
        return self.__sigmoid(dot(inputs, self.weights))  # uses the numpy dot to calculate the output

    def train(self, inputs, outputs, num):  # takes input and corresponding outputs and trains the network num of times
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = .01 * dot(inputs.T,
                                   error * output * (1 - output))  # adjustment proportion to the size of error
            self.weights += adjustment


neural_network = NeuralNetwork()
inputs = array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
outputs = array([[1, 1, 0]]).T

neural_network.train(inputs, outputs, 10000)

print(neural_network.think(array([1, 1, 1])))  # output will be greater than .5
