import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # ndarray
y = iris.target # ndarray

from zscore import zscore
X = zscore(X)


# Perform stepwise regression
# 因为逐步回归是特征工程的一部分，逐步回归是方法，作用在不同的回归模型上
sfs = SequentialFeatureSelector(linear_model.LinearRegression(),
                                k_features='best',
                                forward=False,
                                scoring='neg_mean_squared_error',
                                cv=10) # 逻辑回归是分类，所以精确率是评估模型性能的标准

sfs.fit(X, y) # model.fit

print(sfs.k_feature_names_)
print(sfs.k_score_)










# Create a dataframe with only the selected features
selected_columns = [0, 1, 2, 3] # df.columns
df_selected = X[:, selected_columns]
print(df_selected.shape)

# Split the data into train and test sets
X_train, X_test, \
y_train, y_test = train_test_split(
    df_selected, y,
    test_size=0.3,
    random_state=42)

# Fit a logistic regression model using the selected features
logreg = linear_model.LinearRegression()
logreg.fit(X_train, y_train)


# Make predictions using the test set
y_pred = logreg.predict(X_test)


# print(accuracy_score(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))





# Create a dataframe with only the selected features
selected_columns = [0, 2, 3] # df.columns
df_selected = X[:, selected_columns]
print(df_selected.shape)

# Split the data into train and test sets
X_train, X_test, \
y_train, y_test = train_test_split(
    df_selected, y,
    test_size=0.3,
    random_state=42)

# Fit a logistic regression model using the selected features
logreg = linear_model.LinearRegression()
logreg.fit(X_train, y_train)


# Make predictions using the test set
y_pred = logreg.predict(X_test)

# Evaluate the model performance 没意义啊
# print(y_pred)
# print(accuracy_score(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))