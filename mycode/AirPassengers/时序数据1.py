import seaborn as sns
import pandas as pd

# 使用seaborn加载airline数据集
data = sns.load_dataset('flights')

# Create DataFrame
df = data

# Convert 'year' and 'month' to a datetime column
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%b')

# Set the new datetime column as the index
df.set_index('date', inplace=True)

# Drop unnecessary columns
df.drop(['year', 'month'], axis=1, inplace=True)