import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from my_dataset import dataset
from model1_flow_capture import flow_capture
from model3_detect_swing import mydetect

# get df with momentum
model = flow_capture('2023-wimbledon-1701')
model.run_prob_flow()
df = model.get_final_df()  # df no+everything

# gei labels
model = mydetect(df)
model.run_label_by_gradient('onehot')
model.run_label_by_gradient_demo('onehot')
print(model.df.to_string())



