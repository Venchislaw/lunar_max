import numpy as np


class KNeighborsClassifier:
    def __init__(self, k):
        self.k = k

        self.y_t = None
        self.X_t = None

    def fit(self, X, y):
        self.X_t = X
        self.y_t = y

    def predict(self, X):
        predictions = []
        for datapoint in X:
            neighbors = self.get_neighbors(datapoint)
            prior_class = np.argmax(np.bincount(neighbors))
            predictions.append(prior_class)

        return predictions

    def get_neighbors(self, x_test):
        distances = [np.sqrt(np.sum((x_test - x_train) ** 2)) for x_train in self.X_t]
        neighbors_indices = np.argsort(distances)[:self.k]
        neighbors = self.y_t[neighbors_indices]

        return neighbors


class KNeighborsRegressor:
    def __init__(self, k):
        self.k = k

        self.X_t = None
        self.y_t = None

    def fit(self, X, y):
        self.X_t = X
        self.y_t = y

    def predict(self, X_test):
        predictions = []

        for datapoint in X_test:
            neighbors = self.get_neighbors(datapoint)
            mean_val = np.mean(neighbors)
            predictions.append(mean_val)

        return predictions

    def get_neighbors(self, x_test):
        distances = [np.sqrt(np.sum((x_test - x_train) ** 2)) for x_train in self.X_t]
        neighbors_indices = np.argsort(distances)[:self.k]

        return self.y_t[neighbors_indices]
