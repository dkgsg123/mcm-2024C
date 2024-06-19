import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from zscore import zscore


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # ndarray
x1 = X[:, 0]
y1 = X[:, 1]


class myKmeans:
    def __init__(self, data):
        # data
        self.data_2d = data
        self.data_2d = zscore(self.data_2d)

    def get_labels(self, k):
        self.k = k
        self.model = KMeans(n_clusters=k, n_init=10)
        self.model.fit(self.data_2d)
        return self.model.labels_

    def plot_find_k(self):
        self.inertias = []
        for i in range(1, 11): # k 从 1 遍历到 10
            kmeans = KMeans(n_clusters=i, n_init=10)
            kmeans.fit(self.data_2d)
            self.inertias.append(kmeans.inertia_)

        plt.plot(np.arange(1, 11), self.inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

if __name__ == '__main__':
    model = myKmeans(X)
    labels = model.get_labels(3)

    plt.figure()
    plt.scatter(x1, y1, c=labels)  # scatter+labels
    plt.show()
