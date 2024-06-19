from sklearn import linear_model
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from zscore import zscore
from sklearn.datasets import load_iris

from my_dataset import dataset


class mylinear_reg:
    def __init__(self, X, y):
        self.data_2d = X
        # self.data_2d = zscore(self.data_2d)
        self.target = y

    def get_info(self): # 最终的模型数据
        x = sm.add_constant(self.data_2d) # 添加常量
        model = sm.OLS(self.target, x).fit()
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
        plt.scatter(self.data_2d[:, k], self.target, color='red')
        plt.title('check linearity', fontsize=14)
        plt.xlabel(f'X_{k}', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.grid(True)
        plt.show()

    def plot_res_analysis(self):
        pass

def get_ADF_info(df, k, rolling_length):
    data_diff = df.diff(k)
    data_diff.dropna(inplace=True)
    test_stationarity(data_diff, rolling_length)

def test_stationarity(timeseries, rolling_length):
    # Determing rolling statistics
    rolmean = timeseries.rolling(rolling_length).mean()
    rolstd = timeseries.rolling(rolling_length).std()

    # Plot rolling statistics:
    plt.figure(figsize=(10, 5))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dftest

if __name__ == '__main__':
    dataset = dataset()
    df = dataset.get_match('2023-wimbledon-1701')


    # create series for cumulative points
    temp = df.apply(lambda row: row['p1_points_won'] / (row['p1_points_won'] + row['p2_points_won']),
                                    axis=1)
    final_c_pt = pd.Series(data=temp.values, index=df['elapsed_time'], name='cumulative points')
    final_c_pt = final_c_pt.iloc[1:]
    print(final_c_pt.to_string())
    final_c_pt.plot()
    plt.show()

    demo = pd.Series(data=df['p1_points_won'].values, index=df['elapsed_time'])

    get_ADF_info(demo, 1, 10)