import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class dataset:
    def __init__(self):
        self.data_dict = pd.read_csv(r'./data/data_dictionary.csv')
        self.df = pd.read_csv(r'./data/Wimbledon_featured_matches.csv')


    def print_info(self):
        print(self.df.head().to_string())

        print(self.df['match_id'].value_counts())

    # def to_excel(self):
    #     self.data_dict.to_excel('data_dict.xlsx', index=False)
    #     self.df.to_excel('data_df.xlsx', index=False)

    def match_to_excel(self, name_match):
        df = self.df
        df.set_index('match_id', inplace=True)
        df.loc[name_match, :].to_excel(f'.//match_data//{name_match}.xlsx', index=True)

    def get_match(self, name_match):
        df = self.df
        df.set_index('match_id', inplace=True)
        return df.loc[name_match, :]

    def clean(self):
        pass

    def check(self):
        pass


if __name__ == '__main__':
    dataset = dataset()
    dataset.print_info()
    dataset.match_to_excel('2023-wimbledon-1314')

    # df = dataset.get_match('2023-wimbledon-1701')
    #
    # df = df[(df['set_no'] == 1)]  # & (df['game_no'] == 1)
    # print(df.to_string())