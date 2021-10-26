import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, eta):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn(1)
        self.eta = eta

    def activation(self, x):
        y = np.dot(x, self.weights) + self.bias
        return 1 if y >= 0 else 0

    def backpropagation(self, x, expected):
        for vec, y in zip(x, expected):
            prediction = self.activation(vec)
            self.weights = self.weights + np.dot(self.eta*(y-prediction), vec)
            self.bias = self.bias + self.eta*(y-prediction)

    def print(self):
        print(self.weights)
        print(self.bias)



x = [[22, 11], [6, 3], [4, 2], [14, 7], [1, 2], [8, 16], [30, 60], [34, 68]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

percep = Perceptron(0.1)
percep.backpropagation(x, y)
percep.print()




    

