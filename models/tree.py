from Main.models.main_classes import Node
import numpy as np


# DecisionTree Classification:
class DecisionTreeClassifier:
    def __init__(self, max_depth=50, min_samples_split=2, n_features=None):
        self.loss_history = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.grow(X, y)

    def grow(self, X, y, cur_depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # base case
        if (cur_depth >= self.max_depth) or (n_samples < self.min_samples_split) or (n_labels == 1):
            val = self.most_common_category(y)
            return Node(value=val)

        feature_indices = np.random.choice(n_features, self.n_features,
                                           replace=True)  # n_features may not be equal to self.n_features
        split_feature, split_threshold = self.best_split(X, y, feature_indices)

        # left and right:

        left_indices, right_indices = self.split(X[:, split_feature], split_threshold)
        left = self.grow(X[left_indices, :], y[left_indices], cur_depth + 1)
        right = self.grow(X[right_indices, :], y[right_indices], cur_depth + 1)

        return Node(split_feature, split_threshold, left, right)

    def best_split(self, X, y, feature_indices):
        information_gain = -1
        split_feature_index, best_threshold = None, None

        for feature_index in feature_indices:
            X_col = X[:, feature_index]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                ig = self.information_gain(X_col, y, threshold)

                if ig > information_gain:
                    information_gain = ig
                    split_feature_index = feature_index
                    best_threshold = threshold
        return split_feature_index, best_threshold

    def information_gain(self, X_col, y, threshold):
        # parent entropy

        parent_e = self.entropy(y)

        # create_children
        left_indices, right_indices = self.split(X_col, threshold)

        # weighted entropy for children
        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        e_l, e_r = self.entropy(y[left_indices]), self.entropy(y[right_indices])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_e - child_entropy

    def entropy(self, y_values):
        p_s = np.bincount(y_values) / len(y_values)
        return -np.sum([p * np.log(p) for p in p_s if p > 0])

    def split(self, X_col, threshold):
        left_indices = np.argwhere(X_col <= threshold).flatten()
        right_indices = np.argwhere(X_col > threshold).flatten()

        return left_indices, right_indices

    def most_common_category(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        return [self.traverse_tree(x, self.root) for x in X]

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)


class DecisionTreeRegressor:
    def __init__(self, max_depth=50, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self.grow(X, y)

    def grow(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth == self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0]):
            return Node(value=np.mean(y))

        # Find the best split
        split_feature, split_threshold = self._find_best_split(X, y)

        if split_feature is None:
            return Node(value=np.mean(y))

        left_indices = X[:, split_feature] < split_threshold
        right_indices = ~left_indices

        left_subtree = self.grow(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.grow(X[right_indices], y[right_indices], depth + 1)

        return Node(split_feature, split_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_indices = X[:, feature_idx] < threshold
                right_indices = ~left_indices

                if np.sum(left_indices) < 2 or np.sum(right_indices) < 2:
                    continue

                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, left_targets, right_targets):
        left_variance = np.var(left_targets)
        right_variance = np.var(right_targets)
        return (len(left_targets) * left_variance + len(right_targets) * right_variance) / (len(left_targets) + len(right_targets))

    def predict(self, X):
        return np.array([self.traverse(x, self.root) for x in X])

    def traverse(self, x, node):
        if node.is_leaf_node():
            return node.value
        else:
            if x[node.feature] < node.threshold:
                return self.traverse(x, node.left)
            else:
                return self.traverse(x, node.right)


np.random.seed(42)
X = np.random.rand(100, 2) * 10
y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + np.random.randn(100)


regressor = DecisionTreeRegressor()
regressor.fit(X, y)

y_pred = regressor.predict(X)

print(y)
print(y_pred)
