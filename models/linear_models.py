"""
LUNA-R MAX Linear Models models family.
For details and stuff check out Journal.md

KurwAI labs. MIT License. Free to copy, use, share
"""


import numpy as np


# CODE IS NOT TESTED SORRY
class LinearRegression:
    def __init__(self, fit_intercept=True, method="GD"):
        self.learning_rate = 0.001
        self.fit_intercept = fit_intercept
        self.method = method
        self.weights = None
        self.bias = 0
        self.history = []
    
    def fit(self, X, y, iterations=1_500, learning_rate=None, verbose=10):
        # GD stands for Gradient Descent optimization method
        # Read the details in Journal.md
        if learning_rate: self.learning_rate = learning_rate
        m_samples, n_features = X.shape
        if self.method == "GD":
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
            self.weights = np.zeros((n_features, 1))

            for iteration in range(iterations):
                prediction = self.predict(X)
                cost = self.cost_fn(y, prediction)
                self.history.append(cost)
                if iteration % verbose == 0:
                    print(f"Iteration: {iteration} | Cost: {cost}")

                # UPDATE:
                dw = 1 / m_samples * np.dot(X.T, (prediction - y))
                self.weights -= self.learning_rate * dw
                if self.fit_intercept:
                    db = 1 / m_samples * np.sum((prediction - y))
                    self.bias -= self.learning_rate * db
            
            return self.weights, self.bias

        elif self.method == "OLS":
            if self.fit_intercept:
                X = np.append(X, np.ones((m_samples, 1)), 1)
            self.parameters = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
            self.residual = np.sum((y - np.dot(X, self.parameters))**2)
            return self.parameters[1:], self.parameters[0]

    def predict(self, X):
        if self.method == "GD":
            return np.dot(X, self.weights) + self.bias
        elif self.method == "OLS":
            if len(set(X[:, 0])) != 1:
                X = np.append(X, np.ones((X.shape[0], 1)), 1)
            return np.dot(X, self.parameters) + self.residual

    def cost_fn(self, y, y_pred):
        # MSE
        return (1 / y.shape[0]) * np.sum((y_pred - y)**2)
