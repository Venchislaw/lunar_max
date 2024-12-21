import numpy as np


class Model:
    def __init__(self):
        self.learning_rate = None
        self.weights = None
        self.bias = 0
        self.history = []

    def fit(self, X, y, iterations
