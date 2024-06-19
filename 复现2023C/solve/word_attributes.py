import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class myword:
    def __init__(self, df):
        self.df = df
        self.datetime = self.df.index
        self.values = self.df.values.reshape(-1, 1)

        self.dict = {
                        'e': 64.52,
                        'a': 54.08,
                        'i': 50.39,
                        'ï': 50.39,
                        'r': 50.24,
                        't': 48.05,
                        'o': 44.44,
                        'n': 42.77,
                        's': 36.91,
                        'l': 37.03,
                        'c': 32.44,
                        'u': 26.42,
                        'p': 23.05,
                        'm': 22.82,
                        'd': 22.52,
                        'h': 20.04,
                        'g': 16.47,
                        'b': 15.70,
                        'y': 15.15,
                        'f': 10.22,
                        'v': 8.24,
                        'w': 7.15,
                        'k': 6.37,
                        'x': 2.72,
                        'z': 1.66,
                        'q': 1.85,
                        'j': 1.17
                    }

    def count_repeated(self):
        W = []

        for word_row in self.values:
            word = word_row[0]
            unique_letter_count = len(set(word))
            W.append(unique_letter_count)

        unique_letter_counts = pd.Series(W, index=self.datetime, name='Repeat_cal')

        # print(unique_letter_counts.to_string())
        return unique_letter_counts

    def count_Freq(self):
        W = []

        for word_row in self.values: # 对每一个单词
            word = word_row[0] # str

            # values = 1
            values = 0

            for letter in word:
                value = self.dict.get(letter)

                value = value/100.
                if value is not None:
                    values += value # +*
                else:
                    print(f"'{letter}' is not a valid letter.")

            W.append(values) # 写入

        Freq_counts = pd.Series(W, index=self.datetime, name='Freq_cal')

        # print(Freq_counts.to_string())
        return Freq_counts # 导出series

from scipy.stats import pearsonr

def test_cont(series1, series2):
    data1 = series1.tolist()
    data2 = series2.tolist()

    correlation_coefficient, p_value = pearsonr(data1, data2)

    print("Pearson相关系数:", correlation_coefficient)
    print("p-value:", p_value)

    if p_value < 0.05:
        print("存在显著的相关性")
    else:
        print("不存在显著的相关性")

def CLR(): # 考虑到对数据进行中心对数比变换
    pass

from scipy.stats import f_oneway

def test_anova(series1, series2):
    new = [series1, series2]
    data = pd.concat(new, axis=1)

    f_statistic, p_value = f_oneway(*[group[series2.name] for name, group in data.groupby(series1.name)])

    print("ANOVA F统计量:", f_statistic)
    print("p-value:", p_value)

    if p_value < 0.05:
        print("存在显著的差异")
    else:
        print("不存在显著的差异")

    sns.set()
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=series1.name, y=series2.name, data=data)  # ,order=df_p1_sub.index
    plt.show()