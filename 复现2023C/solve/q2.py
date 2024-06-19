import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_preprocess import dataset


dataset = dataset()
dataset.clean()
dataset.clean_outlier()

df = dataset.df

from word_attributes import myword

word = myword(df['Word'])
word.count_repeated()
word.count_Freq()


new = [df['Word'], word.count_repeated(), word.count_Freq(),
       df['1 try'], df['2 tries'], df['3 tries'], df['4 tries'],df['5 tries'],
       df['6 tries'], df['7 or more tries (X)']]
word_df= pd.concat(new, axis=1)

print(word_df.head().to_string())


# print(word_df[word_df['Repeat_cal'] == 3])

# sns.set()
# sns.countplot(x='Repeat_cal', data=word_df)
#
# plt.title('Frequency of Repeat_cal values')
# plt.xlabel('Repeat_cal values')
# plt.ylabel('Frequency')
# plt.show()

# sns.histplot(word_df['Freq_cal'], bins=20, kde=True, color='skyblue')
#
# plt.title('Distribution of Freq_cal values')
# plt.xlabel('Freq_cal values')
# plt.ylabel('Frequency')
# plt.show()

# sns.set()
# sns.countplot(x='1 try', data=word_df)
#
# plt.title('Frequency of Repeat_cal values')
# plt.xlabel('Repeat_cal values')
# plt.ylabel('Frequency')
# plt.show()


from word_attributes import test_cont

test_cont(word_df['Freq_cal'], word_df['4 tries'])


from word_attributes import test_anova

test_anova(word_df['Repeat_cal'], word_df['7 or more tries (X)'])




