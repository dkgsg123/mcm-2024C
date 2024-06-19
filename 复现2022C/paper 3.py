import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

data_gold = pd.read_csv('LBMA-GOLD.csv')
data_gold['Date'] = pd.to_datetime(data_gold['Date'], format='%m/%d/%y')
data_gold.set_index('Date', inplace=True)
# print(data_gold.head())

data_bitc = pd.read_csv('BCHAIN-MKPRU.csv')
data_bitc['Date'] = pd.to_datetime(data_bitc['Date'], format='%m/%d/%y')
data_bitc.set_index('Date', inplace=True)
# print(data_bitc.head())

print('########################################### start')

from ARIMA import myarima

# split the data
data_gold_30 = data_gold.loc['2016-09-11':'2016-10-10']
data_bitc_30 = data_bitc.iloc[:30]

# split the model and split the pred
model_g = myarima(data_gold_30)
model_b = myarima(data_bitc_30)

# model_g.plot_data()
# model_g.get_ADF_info(1, 5)
# model_g.plot_seasonal()
# model_g.plot_acf_pacf(1)
# model_g.print_auto_model()
# print(data_gold_30.tail(7))
# print(model_g.get_pred(1, 1, 0, '2016-10-11', '2016-10-13'))

# model_b.plot_data()
# print(model_b.get_automodel_pred(3))

############################################### straregy
class strategy:
    def __init__(self, c0, g0, b0):
        self.C = c0
        self.G = g0
        self.B = b0
        self.beta_g = 0.001
        self.beta_b = 0.001
        self.gamma_g = 0.2
        self.gamma_b = 0.2
        self.q = 0.2
        self.alpha_g = 0.01
        self.alpha_b = 0.02
        self.delta_g = 0.2
        self.delta_b = 0.0002

    def update(self, data_g, pred_g, data_b, pred_b):

        self.X_g = pred_g['Value'].mean()
        self.Y_g = data_g.iloc[-3:-1, 0].mean()
        self.y_g = data_g.iloc[[-1], 0].values

        self.X_b = pred_b['Value'].mean()
        self.Y_b = data_b.iloc[-3:-1, 0].mean() # .iloc 返回series
        self.y_b = data_b.iloc[[-1], 0].values
        # print(self.y_b, type(self.y_b))

        self.A = self.C + self.G * self.y_g + self.B * self.y_b

        if self.X_g - self.Y_g > self.A * self.beta_g:
            self.C = self.C - self.delta_g * (1 + self.alpha_g) * self.y_g
            self.G = self.G + self.delta_g
        elif self.X_g - self.Y_g < - self.A * self.gamma_g:
            self.C = self.C + self.delta_g * (1 - self.alpha_g) * self.y_g
            self.G = self.G - self.delta_g

        if self.X_b - self.Y_b > self.A * self.beta_b:
            self.C = self.C - self.delta_b * (1 + self.alpha_b) * self.y_b
            self.B = self.B + self.delta_b
        elif self.X_b - self.Y_b < - self.A * self.gamma_b:
            self.C = self.C + self.delta_b * (1 - self.alpha_b) * self.y_b
            self.B = self.B - self.delta_b

        print(self.C, self.G, self.B)

model = strategy(1000, 0, 0)
model.update(data_gold_30, model_g.get_pred(1, 1, 0, '2016-10-11', '2016-10-13'), data_bitc_30, model_b.get_automodel_pred(3))
# 传进 4 个 dataframe