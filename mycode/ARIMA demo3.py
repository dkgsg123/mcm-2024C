from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

################################## load data
dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]

dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2090'))

################################## data info
dta.plot(figsize=(12,8))
plt.title('dta')
print('dta:',dta)

result = seasonal_decompose(dta,
                            model='multiplicative')

# ETS plot
result.plot()
plt.show()

'''
ARIMA 模型对时间序列的要求是平稳型。因此，当你得到一个非平稳的时间序列时，首先要做的即是做时间序列的差分，
直到得到一个平稳时间序列。如果你对时间序列做d次差分才能得到一个平稳序列，那么可以使用ARIMA(p,d,q)模型，
其中d是差分次数，这里我们尝试做一次差分。
'''
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)

################################## create diff
dta_diff1= dta.diff(1)
################################## plot
dta_diff1.plot(ax=ax1)
plt.title('dta_diff1')
print('dta_diff1:',dta_diff1)
################################## drop row 1
dta_diff1.dropna(inplace=True)


'''
平稳性检验：
其中，p-value是接受原假设的概率，一般情况下按照5%的置信区间去判断就可以。
若p-value小于0.05，而且检验统计量Test Statistic小于任何一个critical value，
则拒绝原假设，即判断为平稳数据。
'''
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
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dftest
StationaryTest = test_stationarity(dta_diff1, 20)


'''
现在我们已经得到一个平稳的时间序列，接来下就是选择合适的ARIMA模型，即ARIMA模型中合适的p,q。
第一步我们要先检查平稳时间序列的自相关图和偏自相关图。
其中lags 表示滞后的阶数，以上分别得到acf图和pacf图
'''
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta_diff1,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta_diff1,lags=40,ax=ax2)

'''
现在有以上这么多可供选择的模型，我们通常采用ARMA模型的AIC法则,
优先考虑的模型应是AIC值最小的那一个
但要注意的是，这些准则不能说明某一个模型的精确度，也即是说，
对于三个模型Ａ，Ｂ，Ｃ，我们能够判断出Ｃ模型是最好的，
但不能保证Ｃ模型能够很好地刻画数据，因为有可能三个模型都是糟糕的。
可以看到ARMA(7,1,1)的aic，bic，hqic均最小，因此是最佳模型。
'''
arma_mod20 = sm.tsa.ARIMA(dta,order=(7,0,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod30 = sm.tsa.ARIMA(dta,order=(0,0,1)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod40 = sm.tsa.ARIMA(dta,order=(7,1,1)).fit()
print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
arma_mod50 = sm.tsa.ARIMA(dta,order=(8,0,0)).fit()
print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

'''
模型检验
在指数平滑模型下，观察ARIMA模型的残差是否是平均值为0且方差为常数
的正态分布（服从零均值、方差不变的正态分布），同时也要观察连续残差
是否（自）相关

对ARMA(7,1,1)模型所产生的残差做自相关图
'''
resid = arma_mod40.resid
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

'''
做D-W检验:
DW值显著的接近于O或４时，则存在自相关性，
而接近于２时，则不存在（一阶）自相关性。
'''
print('D-W test')
print(sm.stats.durbin_watson(arma_mod20.resid.values))

'''
观察是否符合正态分布：
使用QQ图，它用于直观验证一组数据是否来自某个分布，
或者验证某两组数据是否来自同一（族）分布。

QQ图是一种散点图，对应于正态分布的QQ图，就是由标准正态分布的分位数
为横坐标，样本值为纵坐标的散点图（其他版本[2]，有将 (x-m)/std作为
纵坐标，那么正态分布得到的散点图是直线：y=x）。要利用QQ图鉴别样本数
据是否近似于正态分布，只需看QQ图上的点是否近似地在一条直线附近，图形
是直线说明是正态分布，而且该直线的斜率为标准差，截距为均值，用QQ图还
可获得样本偏度和峰度的粗略信息。
'''
resid = arma_mod40.resid#残差
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

'''
Ljung-Box检验(白噪声检验即随机性检验 p值小于5%,序列为非白噪声)

对时间序列是否存在滞后相关的一种统计检验。对于滞后相关的检验，
我们常常采用的方法还包括计算ACF和PCAF并观察其图像，但是无论是
ACF还是PACF都仅仅考虑是否存在某一特定滞后阶数的相关。
LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在。

时间序列中一个最基本的模型就是高斯白噪声序列。而对于ARIMA模型，其残差
被假定为高斯白噪声序列，所以当我们用ARIMA模型去拟合数据时，拟合后我们要
对残差的估计序列进行LB检验，判断其是否是高斯白噪声，如果不是，
那么就说明ARIMA模型也许并不是一个适合样本的模型。
'''
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,20), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
'''
检验的结果就是看最后一列前十二行的检验概率（一般观察滞后1~12阶），如果检验概率小于
给定的显著性水平，比如0.05、0.10等就拒绝原假设，其原假设是相关系数为零。就结果来看，
如果取显著性水平为0.05，那么相关系数与零没有显著差异，即为白噪声序列。
'''

'''
模型预测
模型确定之后，就可以开始进行预测了，我们对未来十年的数据进行预测。
'''
predict_sunspots = arma_mod40.predict('2091', '2100', dynamic=True)
# print(predict_sunspots)

#为绘图的连续性把2090的值添加为PredicValue第一个元素
PredicValue=[]
PredicValue.append(dta.values[-1])
for i in range(len(predict_sunspots.values)):
    PredicValue.append(predict_sunspots.values[i])
PredicValue=pd.Series(PredicValue)
PredicValue.index = pd.Index(sm.tsa.datetools.dates_from_range('2090','2100'))

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.loc['2001':].plot(ax=ax,label='训练数据')
PredicValue.plot(ax=ax, label='预测值')
plt.legend()
plt.show()