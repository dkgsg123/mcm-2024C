import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from zscore import zscore

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # ndarray
y = iris.target # ndarray

# 与model相结合，算是特征工程的一种
# 到底要选择特征，还是要最终结果
class mystepwise:
    def __init__(self, X, y):
        self.data_2d = X
        self.data_2d = zscore(self.data_2d)
        self.target = y

    def get_index(self):
        pass

    def get_new_X(self):
        pass

    def get_evaluation(self):
        pass
