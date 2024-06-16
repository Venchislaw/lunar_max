import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.weights = None
        self.bias = None
        self.fit_intercept = fit_intercept
        self.loss_history = []

    def fit(self, X, y, learning_rate=1, n_epochs=100):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(n_epochs):
            predictions = self.predict(X)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = 0
            if self.fit_intercept:
                db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= dw * learning_rate
            self.bias -= db * learning_rate

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


model = LinearRegression()
model = model.fit(np.array([[1], [3]]), np.array([2, 6]), learning_rate=0.01)
print(model.weights, model.bias)
