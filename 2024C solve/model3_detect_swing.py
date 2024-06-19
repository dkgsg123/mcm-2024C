import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from zscore import zscore

from my_dataset import dataset
from model1_flow_capture import flow_capture

from sklearn.preprocessing import LabelEncoder

class mydetect:
    def __init__(self, df):
        self.df = df

        # params for methods
        # 1
        self.m1_threshold = 0.001
        # 2
        self.m2_threshold = 0.001
        self.m2_window_size = 5

    def run_label_by_gradient(self, encoding_method='integer'):
        data = self.df['p1_final_momentum']

        gradient = np.gradient(data)
        # print(gradient)

        labels = []

        for grad_value in gradient:
            if grad_value > self.m1_threshold:
                labels.append('increasing')
            elif grad_value < -self.m1_threshold:
                labels.append('decreasing')
            else:
                labels.append('stable')

        for i in range(len(labels)):
            if i >0 and labels[i] != labels[i-1] and labels[i-1] != 'inflection':
                if i+4 < len(labels)-1 and labels[i] == labels[i+1] and labels[i] == labels[i+2] \
                        and labels[i] == labels[i+3] and labels[i] == labels[i+4]: # 连续五个一致才算inflection
                    labels[i] = 'inflection'


        self.df['mylabel_m1'] = labels

        # Encoding
        if encoding_method == 'integer':
            self.df['encoder_m1'] = LabelEncoder().fit_transform(self.df['mylabel_m1']) # new column
        elif encoding_method == 'onehot':
            encoded_df = pd.get_dummies(self.df['mylabel_m1'], prefix='encoder_m1', drop_first=False)
            self.df = pd.concat([self.df, encoded_df], axis=1)

        return self.df

    def run_label_by_gradient_demo(self, encoding_method='integer'):
        data = self.df['p1_final_momentum']

        gradient = np.gradient(data)
        # print(gradient)

        labels = []

        for grad_value in gradient:
            if grad_value > self.m1_threshold:
                labels.append('increasing')
            elif grad_value < -self.m1_threshold:
                labels.append('decreasing')
            else:
                labels.append('stable')

        for i in range(len(labels)):
            if i > 0 and labels[i] != labels[i - 1] and labels[i - 1] != 'inflection':
                if i + 4 < len(labels) - 1 and labels[i] == labels[i + 1] and labels[i] == labels[i + 2]: # 连续2个
                    # and labels[i] == labels[i+3] and labels[i] == labels[i+4]: # 连续五个一致才算inflection
                    labels[i] = 'inflection'

        self.df['mylabel_m1_demo'] = labels

        # Encoding
        if encoding_method == 'integer':
            self.df['encoder_m1_demo'] = LabelEncoder().fit_transform(self.df['mylabel_m1_demo'])  # new column
        elif encoding_method == 'onehot':
            encoded_df = pd.get_dummies(self.df['mylabel_m1_demo'], prefix='encoder_m1_demo', drop_first=False)
            self.df = pd.concat([self.df, encoded_df], axis=1)

        return self.df



    def run_label_by_sliding_window(self, encoding_method='integer'):
        data = self.df['p1_final_momentum']

        labels = []

        for i in range(1, len(data)+1):
            start_idx = max(0, i - self.m2_window_size + 1)
            end_idx = i + 1

            window_data = data[start_idx:end_idx]
            window_gradient = np.gradient(window_data)

            if window_gradient[-1] > self.m2_threshold:
                labels.append('increasing')
            elif window_gradient[-1] < -self.m2_threshold:
                labels.append('decreasing')
            else:
                labels.append('inflection')

        self.df['mylabel_m2'] = labels

        # 编码
        if encoding_method == 'integer':
            self.df['encoder_m2'] = LabelEncoder().fit_transform(self.df['mylabel_m2'])
        elif encoding_method == 'onehot':
            encoded_df = pd.get_dummies(self.df['mylabel_m2'], prefix='encoder_m2', drop_first=False)
            self.df = pd.concat([self.df, encoded_df], axis=1)

        return self.df


if __name__ == '__main__':
    match_name = '1314'
    model = flow_capture(f'2023-wimbledon-{match_name}')
    model.run_prob_flow()
    df = model.get_final_df()  # df no+everything

    model = mydetect(df)
    model.run_label_by_gradient('onehot')
    model.run_label_by_gradient_demo('onehot') # 3 in a row
    print(model.df.to_string())

    ################################# 好的数据存excel #######################################
    # model.run_label_by_gradient('integer').to_excel(r'./results/1701/initial params.xlsx')
    model.df.to_excel(f'./results/{match_name}/{match_name} memo.xlsx')
    # model.run_label_by_sliding_window('integer').to_excel(r'./results/1701/initial params.xlsx')