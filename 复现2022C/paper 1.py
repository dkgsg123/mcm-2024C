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


