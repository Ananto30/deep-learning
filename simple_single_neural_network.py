"""
This is a example of simple neural network which can predict a output
of mathematical expression without actually knowing it.

For simplifying the idea of neural network we are going to use a linear
math expression 2a+3b

For input a,b there is output = weight1 * a + weight2 * b...(1)
These weights will be adjusted during the training period
"""
from numpy import random, dot, array  # we will only use the numpy library


class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((2, 1)) - 1  # initializing the weights with random number between 0 & 1

    def think(self, inputs):  # returns the output by using the formula 1
        return (dot(inputs, self.weights))  # uses the numpy dot to calculate the output

    def train(self, inputs, outputs, num):  # takes input and corresponding outputs and trains the network num of times
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = .01 * dot(inputs.T, error)  # adjustment proportion to the size of error
            self.weights += adjustment


neural_network = NeuralNetwork()
inputs = array([[2, 3], [1, 1], [5, 2], [12, 3]])
outputs = array([[13, 5, 16, 33]]).T

neural_network.train(inputs, outputs, 10000)

print(neural_network.think(array([3, 4])))  # output will be 2*3+3*4=18
