import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from my_dataset import dataset
from model2_R2_test import mylinear_reg

dataset = dataset()
df = dataset.get_match('2023-wimbledon-1701')

s = df['p2_points_won']
s = pd.Series(data=np.insert(s.values, 0, 0))
print(s)

model = mylinear_reg(X=s.index, y=s.values.reshape(-1, 1))
model.get_info()