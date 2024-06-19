import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # ndarray
# x1 = X[:, 0] # scatter
# y1 = X[:, 1] # scatter

class myPCA:
    def __init__(self, data):
        # data
        self.data_2d = data

    def get_ratio(self):
        nrow, ncol = self.data_2d.shape
        model = PCA(n_components=ncol)
        model.fit(self.data_2d)
        print(model.explained_variance_ratio_)
        return model.explained_variance_ratio_

    def get_PCA(self, k, output): # 自动标准化了
        model = PCA(n_components=k)
        model.fit(self.data_2d)

        if output == 'mat':
            return model.transform(self.data_2d)
        elif output == 'sv':
            return model.singular_values_
        elif output == 'var':
            return model.explained_variance_

    def get_PCA_mle(self, output):
        model = PCA(n_components="mle")
        model.fit(self.data_2d)

        if output == 'mat':
            return model.transform(self.data_2d)
        elif output == 'sv':
            return model.singular_values_
        elif output == 'var':
            return model.explained_variance_

# model = myPCA(X)
# model.get_ratio()

# plt.figure()
# labels = iris.target
# plt.scatter(PCA_mat[:, 0], PCA_mat[:, 1], c=labels) # scatter+labels
# plt.xlabel('com 1')
# plt.ylabel('com 2')
# plt.show()