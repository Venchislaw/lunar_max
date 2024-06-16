class Model:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.loss_fn = None

        self.loss_history = []

    def fit(self, loss):
        self.loss_history.append(loss)

    def predict(self):
        pass

    def evaluate(self, X, y):
        return self.loss_fn(self.predict(X), y)  # don't worry. It works in subclasses


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

