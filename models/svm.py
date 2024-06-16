import numpy as np


class LinearSVC:
    def __init__(self, lr=0.001, lambda_param=0.001):
        self.lr = lr
        self.lambda_param = lambda_param

        self.w = None
        self.b = None

    def fit(self, X, y, n_iters):
        n_samples, n_features = X.shape

        y = np.where(y >= 1, 1, -1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(n_iters):
            for i, x_i in enumerate(X):
                condition = y[i] * (np.dot(x_i, self.w) - self.b)

                if condition >= 1:
                    self.w -= self.lr * 2 * (self.lambda_param * self.w)
                    # we don't update b

                else:
                    self.w -= self.lr * 2 * (self.lambda_param * self.w - y[i] * x_i)
                    self.b -= self.lr * y[i]

        return self.w, self.b

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
