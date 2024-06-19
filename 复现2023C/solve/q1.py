import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from data_preprocess import dataset
from ARIMA import myarima


# dataset
dataset = dataset()
dataset.clean()
dataset.clean_outlier()
df = dataset.df



# 提取下降趋势并log化
Values = df[['Number of  reported results']]
temp = Values['2022-02-01':]
temp.columns = ['Number of  reported results']


# temp.plot()
# plt.show()

for i in range(len(temp.values)):
    temp.values[i][0] = np.log(temp.values[i][0])

# temp.plot()
# plt.show()







model = myarima(temp)
# model.plot_data()
# model.plot_seasonal()
# model.plot_diff(1)
# model.get_ADF_info(1, 10)
# model.plot_acf_pacf(1)
# model.print_auto_model()






# 得到模型，作评估
final_model = model.get_final_model(0, 1, 1, 1, 0, 1, 7)
# model.get_MSE(final_model).to_excel(r'./q1 evaluation/our model mse.xlsx')

# get some random model

# model1 = model.get_final_model(0, 1, 0, 0, 0, 1, 7)
# model.get_residuals(model1).to_excel(r'./q1 evaluation/random model1 res.xlsx')
# model.get_MSE(model1).to_excel(r'./q1 evaluation/random model1 mse.xlsx')
# model2 = model.get_final_model(1, 1, 0, 0, 0, 1, 7)
# model.get_residuals(model2).to_excel(r'./q1 evaluation/random model2 res.xlsx')
# model.get_MSE(model2).to_excel(r'./q1 evaluation/random model2 mse.xlsx')
# model3 = model.get_final_model(1, 1, 0, 1, 0, 0, 7)
# model.get_residuals(model3).to_excel(r'./q1 evaluation/random model3 res.xlsx')
# model.get_MSE(model3).to_excel(r'./q1 evaluation/random model3 mse.xlsx')

model.print_results(final_model)
# model.plot_res_acf_pacf(final_model)
# model.get_DW_test(final_model)


# # 得到预测结果
# demo = model.get_automodel_pred(100)
# demo = demo[:'2023-03-02']
#
# # 逆变换
# for i in range(len(demo.values)):
#     demo.values[i][0] = np.round(np.exp(demo.values[i][0])).astype(int)
#     demo.values[i][1] = np.round(np.exp(demo.values[i][1])).astype(int)
#     demo.values[i][2] = np.round(np.exp(demo.values[i][2])).astype(int)

# 导出数据
# demo.to_excel('prediction.xlsx', index=True)