from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# load iris class
iris = load_iris()

# create X y
X= iris.data # ndarray
# 这是因为y的具体分类值是字符串，返回的是0 1 2
y = iris.target # ndarray
y = y.reshape(-1, 1)
feature_names = iris.feature_names # name list
target_names = iris.target_names # name list

if 0:
    print(X, y)

# 随时写出这种导出数据的代码
# def nd2csv(mat):
#     for row in mat:
#         print(row[0], row[1], row[2], row[3])
#
# nd2csv(X)

X_df = pd.DataFrame(X, columns=feature_names)

if 0:
    print(X_df)
if 0:
    X_df.to_excel('iris.xlsx')

# 两个思路
# 法一 把y当成矩阵，按水平方向相加
X_y = np.concatenate((X, y), axis=1) # 元组+方向
# 法二 X作为矩阵，y作为一维数组，直接添加列
# combined_matrix = np.column_stack((X, y)) # 元组

f_l_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'label']

Xy_df = pd.DataFrame(X_y, columns=f_l_names)

if 0:
    print(Xy_df)

if 1:
    Xy_df.to_excel('iris_labels.xlsx')