from sklearn import linear_model
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np
from zscore import zscore

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # ndarray
y = iris.target # ndarray

class mylinear_reg:
    def __init__(self, X, y):
        self.data_2d = X
        self.data_2d = zscore(self.data_2d)
        self.target = y

    def get_info(self): # 最终的模型数据
        x = sm.add_constant(self.data_2d) # 添加常量
        model = sm.OLS(y, x).fit()
        # predictions = model.predict(x)
        print(model.summary())

    def get_coefs(self, output):
        model = linear_model.LinearRegression()
        model.fit(self.data_2d, self.target)

        if output == 'intercept':
             return model.intercept_
        elif output == 'coefs':
            return model.coef_
        elif output == 'all':
            return [model.intercept_, model.coef_]

    def get_pred(self, X_sample):
        model = linear_model.LinearRegression()
        model.fit(self.data_2d, self.target)

        print('prediction: ', model.predict(X_sample))

    def plot_check_linearity(self, k):
        plt.scatter(self.data_2d[:, k], y, color='red')
        plt.title('check linearity', fontsize=14)
        plt.xlabel(f'X_{k}', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.grid(True)
        plt.show()

    def plot_res_analysis(self):
        pass

model = mylinear_reg(X, y)
model.plot_check_linearity(3)
model.get_info()
print(model.get_coefs('all'))
model.get_pred(np.array([2, 3, 4, 5]).reshape(-1, 4))