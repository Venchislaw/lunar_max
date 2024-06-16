import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def calculate_distance(datapoint, centroids):
        return np.sqrt(np.sum((datapoint - centroids) ** 2, axis=1))

    def fit(self, X, max_iters=100, threshold=1e-4):
        # randomly initialize centroids
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iters):
            # calculate distances and classify points to particular cluster
            y = []
            for datapoint in X:
                distances = self.calculate_distance(datapoint, self.centroids)
                min_distance = np.argmin(distances)

                y.append(min_distance)
            y = np.array(y)

            # get indices of elements for each cluster
            indices_cluster = []
            for i in range(self.k):
                indices_cluster.append(np.argwhere(y == i))

            # calculate new centers
            cluster_centers = []

            for i, indices in enumerate(indices_cluster):
                # if our cluster has 0 points
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            # check goal achievement (to avoid useless iterations)
            if np.max(abs(self.centroids - np.array(cluster_centers))) < threshold:
                break
            else:
                self.centroids = np.array(cluster_centers)

    def predict(self, data):
        y = []
        for datapoint in data:
            distances = self.calculate_distance(datapoint, self.centroids)
            min_distance = np.argmin(distances)

            y.append(min_distance)
        y = np.array(y)

        return y


"""
a = np.random.randint(0, 100, (100, 2))

clusterizer = KMeans()
clusterizer.fit(a)
labels = clusterizer.predict(a)

plt.scatter(a[:, 0], a[:, 1], c=labels)
plt.scatter(clusterizer.centroids[:, 0], clusterizer.centroids[:, 1],
            c='red', label='centroids')
plt.legend()
plt.show()
"""
