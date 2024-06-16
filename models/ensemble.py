from Main.models.tree import DecisionTreeClassifier, DecisionTreeRegressor
from Main.models.main_classes import Node
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class RandomForestClassifier:
    def __init__(self, n_trees, max_depth=50, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self.trees = []

    def fit(self, X, y):
        for t in range(self.n_trees):
            print(f'fitting tree {t + 1}/{self.n_trees}')
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          n_features=self.n_features)

            X_sampled, y_sampled = self.sample_with_replacement_data(X, y)

            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)

    def sample_with_replacement_data(self, X, y):
        samples = X.shape[0]
        random_indices = np.random.choice(samples, int(samples ** 0.5))  # by default replace is True

        return X[random_indices], y[random_indices]

    def most_frequent_label(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        preds_for_tree = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_frequent_label(pred) for pred in preds_for_tree])
        return predictions


class RandomForestRegressor:
    def __init__(self, n_trees, max_depth=50, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees = []

    def fit(self, X, y):
        for t in range(self.n_trees):
            print(f'fitting tree {t + 1}/{self.n_trees}')
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split)

            X_sampled, y_sampled = self.sample_with_replacement_data(X, y)

            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)

    def sample_with_replacement_data(self, X, y):
        samples = X.shape[0]
        random_indices = np.random.choice(samples, int(samples ** 0.5))  # by default replace is True

        return X[random_indices], y[random_indices]

    def predict(self, X):
        trees_predictions = np.empty((len(X), self.n_trees))

        for i, tree in enumerate(self.trees):
            trees_predictions[:, i] = tree.predict(X)

        predictions = np.mean(trees_predictions, axis=1)

        return predictions


class GradientBoostingClassifier:
    def __init__(self, min_samlpes_split=1, max_depth=3, n_estimators=10, threshold=0.5):

        self.max_depth_ = max_depth
        self.min_samlpes_split = min_samlpes_split
        self.n_estimators = n_estimators
        self.threshold_ = threshold

    def fit(self, X, y, lr=0.8):
        self.estimators = []
        self.earlier_predictions = np.log(list(y).count(1) / list(y).count(0))

        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth_, min_samples_split=self.min_samlpes_split)

            residuals = y - self.earlier_predictions
            tree.fit(X, residuals)

            self.estimators.append(tree)

            self.earlier_predictions += lr * tree.predict(X)

    def predict(self, X):
        trees_predictions = np.empty((len(X), len(self.estimators)))

        for i, tree in enumerate(self.estimators):
            trees_predictions[:, i] = tree.predict(X)

        predictions = np.sum(trees_predictions, axis=1)
        predictions = np.float64(predictions >= self.threshold_)

        return predictions


class GradientBoostingRegressor:
    def __init__(self, n_estimators, max_depth=50, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samlpes_split = min_samples_split

        self.estimators = []

    def fit(self, X, y, lr=0.1):
        self.lr = lr
        self.earlier_predictions = np.ones(len(y)) * np.mean(y)

        for t in range(self.n_estimators):
            print(f'Fitting {t + 1}/{self.n_estimators} tree')

            residuals = y - self.earlier_predictions

            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samlpes_split)
            tree.fit(X, residuals)

            self.estimators.append(tree)

            self.earlier_predictions += self.lr * tree.predict(X)

    def predict(self, X):
        return np.sum(self.lr * tree.predict(X) for tree in self.estimators) + np.mean(y)
