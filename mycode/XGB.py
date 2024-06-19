import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

# visualization
from sklearn.tree import export_text
# import graphviz

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # ndarray
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


# dtree for classification
# 鸢尾花数据集是分类任务
class mydtree:
    def __init__(self, X, y):
        # 还需要传入其他的超参数

        self.data_2d = X
        self.target = y
        self.model = DecisionTreeClassifier()

    def plot_dtree(self, names): # 基于所有的数据集
        self.model = self.model.fit(self.data_2d, self.target)
        tree.plot_tree(self.model, feature_names=names, class_names=target_names, filled=True) # 彩色的
        plt.show()

    def print_text(self, names): # 基于所有的数据集
        self.model = self.model.fit(self.data_2d, self.target)
        r = export_text(self.model, feature_names=names)
        print(r)

    def get_model_evaluation(self, test_size=0.3, random_state=None):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_2d, self.target, test_size=test_size, random_state=random_state
        )

        # Train the model on the training set
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        # Evaluate the model on the test set
        accuracy = accuracy_score(y_pred, y_test)

        return accuracy

    def get_kfold_evaluation(self, n_splits=5, random_state=None):
        # Create a KFold cross-validation splitter
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Perform cross-validation
        scores = cross_val_score(self.model, self.data_2d, self.target, cv=kf)

        # Return the mean accuracy across all folds
        mean_accuracy = scores.mean()

        return mean_accuracy
