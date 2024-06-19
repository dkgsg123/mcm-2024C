import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class dataset:
    def __init__(self):
        self.path = 'Problem_C_Data_Wordle.xlsx'
        self.df = pd.read_excel('Problem_C_Data_Wordle.xlsx')

    def clean(self):
        self.df.drop(self.df.columns[0], axis=1, inplace=True)
        self.df.columns = self.df.iloc[0]
        self.df = self.df[1:]
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.df = self.df.sort_index()
        self.df['Word'] = self.df['Word'].apply(lambda x: x.strip())

    def print_info(self):
        # print(self.df.columns)
        # print(self.df.dtypes)
        # print(self.df.info()) # 359行
        pass

    def plot_people_line(self):
        # df的可视化都是用seaborn
        sns.set()
        self.df['Number of  reported results'].plot()
        self.df['Number in hard mode'].plot()
        plt.legend(['total', 'hard'])
        plt.show()

        # sns.set()
        # sns.lineplot(data=self.df, x=self.df.index, y='Number of  reported results')
        # sns.lineplot(data=self.df, x=self.df.index, y='Number in hard mode')
        # plt.legend(['total', 'hard'])
        # plt.show()

    def plot_percent_scatter(self):
        y =self.df['Total percent'].values
        x =self.df.index
        plt.scatter(x, y)
        plt.ylim(80, 120)
        plt.show()

    def plot_people_scatter(self):
        sns.set()
        df = self.df
        y = df['Number of  reported results'].values
        # y = df['Number in hard mode'].values
        x = df.index
        plt.scatter(x, y)
        plt.show()

    def plot_dist_bar(self, date):
        df = self.df.loc[[date]]
        tries = df.columns[-8:-1]
        df[tries] = df[tries].astype(int)
        df_melted = df[tries].melt(var_name='Tries', value_name='Percent')

        sns.set()
        sns.barplot(x='Tries', y='Percent', data=df_melted, palette='viridis')
        plt.xlabel('Tries')
        plt.ylabel('Percent')
        plt.title(f'Percent of Tries when {date}')
        plt.show()
        # print(df.to_string())

    def print_outlier(self):
        df = self.df
        df = df[(df['Total percent'] > 107) | (df['Total percent'] < 93)]

        self.outlier_index = df.index

        print(df.to_string())
        print(df.shape)

    def clean_outlier(self):

        df = self.df
        outlier = df[(df['Total percent'] > 107) | (df['Total percent'] < 93)]

        self.outlier_index = outlier.index

        for index in self.outlier_index:

            if index-pd.DateOffset(days=1) not in self.outlier_index:
                temp = self.df.loc[[index-pd.DateOffset(days=1)]]
                self.df.loc[index] = temp.iloc[0]

            elif index-pd.DateOffset(days=2) not in self.outlier_index:

                temp = self.df.loc[[index - pd.DateOffset(days=2)]]
                self.df.loc[index] = temp.iloc[0]


        self.outlier_index2 = df[df['Word'].apply(lambda x: len(x) != 5)].index
        # print(df.loc[self.outlier_index2])

        for index in self.outlier_index2:

            if index-pd.DateOffset(days=1) not in self.outlier_index2:
                temp = self.df.loc[[index-pd.DateOffset(days=1)]]
                self.df.loc[index] = temp.iloc[0]

            elif index-pd.DateOffset(days=2) not in self.outlier_index2:

                temp = self.df.loc[[index - pd.DateOffset(days=2)]]
                self.df.loc[index] = temp.iloc[0]

        self.df.loc['2022-11-30'] = self.df.loc['2022-11-29']

        print(self.df.head(5).to_string())


    def check(self):
        df = self.df
        print(df.loc[self.outlier_index])
        print(df.loc[self.outlier_index2])

    def plotlog_people_scatter(self):
        sns.set()
        df = self.df
        y = df['Number of  reported results'].values
        # y = df['Number in hard mode'].values
        x = df.index

        print(type(y[0]))

        for i in range(len(y)):
            y[i] = np.log(y[i])

        plt.scatter(x, y)
        plt.show()


dataset = dataset()
dataset.clean()
dataset.clean_outlier()

dataset.plot_people_scatter()