import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.api import qqplot
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import statsmodels.api as sm

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

class myarima:
    def __init__(self, dataframe): # 传入以datetime为index的data
        self.df = dataframe
        self.time = dataframe.index
        self.values = dataframe.iloc[:, 0]

    # data info
    def plot_data(self):
        # sns.set_theme(style='darkgrid')
        sns.set()
        # sns.set_palette('pastel')
        sns.lineplot(x=self.df.index, y=self.df.columns[0], data=self.df)
        plt.show()

    def plot_seasonal(self):
        result = seasonal_decompose(self.df,
                                    model='additive')
        result.plot()
        plt.show()

        # print("Trend:", result.trend)
        # print("Seasonal:", result.seasonal)
        # print("Residual:", result.resid)
        # print("Observed:", result.observed)

    # 确定d
    def plot_diff(self, k):
        data_diff = self.df.diff(k)
        data_diff.dropna(inplace=True)
        sns.set()
        sns.lineplot(x=data_diff.index, y=data_diff.columns[0], data=data_diff)
        plt.ylabel(f'diff {k}')
        plt.show()

    def get_ADF_info(self, k, rolling_length):
        data_diff = self.df.diff(k)
        data_diff.dropna(inplace=True)
        test_stationarity(data_diff, rolling_length)

    # 通过差分确定pq
    def plot_acf_pacf(self, k):
        data_diff = self.df.diff(k)
        data_diff.dropna(inplace=True)

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(data_diff, lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(data_diff, lags=40, ax=ax2)
        plt.show()


    def get_final_model(self, p, d, q, P, D, Q, m):
        # print(self.df.dtypes)
        self.df['Number of  reported results'] = pd.to_numeric(self.df['Number of  reported results'], errors='coerce')
        model = sm.tsa.SARIMAX(self.df, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model = model.fit(disp=False)
        return model

    def get_residuals(self, model):
        return model.resid

    def get_MSE(self, model):
        series = model.resid
        mse_series = series.apply(lambda x: x ** 2)
        return mse_series

    def print_results(self, model):
        print(model.summary())
        model.plot_diagnostics()
        plt.show()

    # 确定超参数后，检验
    def plot_res_acf_pacf(self, model):
        resid = model.resid
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1) # lags 一般是Trian数据的前50%
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
        plt.show()

    def get_DW_test(self, model):
        print('D-W test')
        print(sm.stats.durbin_watson(model.resid.values))

    def plot_res_QQ(self, model):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        fig = qqplot(model.resid, line='q', ax=ax, fit=True)

    def get_L_Box_test_info(self, model):
        r, q, p = sm.tsa.acf(model.resid.values.squeeze(), qstat=True)
        data = np.c_[range(1, 20), r[1:], q, p]
        table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
        print(table.set_index('lag'))

    # 打印结果
    def print_auto_model(self):
        auto_arima(self.df, start_p=0, start_q=0,
                              max_p=8, max_q=8, m=7,
                              start_P=0, start_Q=0, seasonal=True,
                              max_P=8, max_Q=8,
                              max_d=3, max_D=3, trace=True,
                              information_criterion='aic',
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=False)

    def get_pred(self, p, d, q, start_date, end_date):
        model = sm.tsa.ARIMA(self.values, order=(p, d, q))
        model = model.fit()
        return model.predict(start=start_date, end=end_date, dynamic=False).to_frame()

    def get_automodel_pred(self, length):
        model = auto_arima(self.df, start_p=0, start_q=0,
                           max_p=8, max_q=8, m=7,
                           start_P=0, start_Q=0, seasonal=True,
                           max_P=8, max_Q=8,
                           max_d=3, max_D=3, trace=False,
                           information_criterion='aic',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False)

        # model = sm.tsa.SARIMAX(self.df, order=(p, d, q), seasonal_order=(P, D, Q, m))
        # model = model.fit()

        forecast, conf_int = model.predict(n_periods=length, return_conf_int=True)

        forecast_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', end='2023-04-10'),
            'Forecast': forecast,
            'Lower Bound': conf_int[:, 0],
            'Upper Bound': conf_int[:, 1]
        })

        forecast_df.set_index('Date', inplace=True)
        return forecast_df
