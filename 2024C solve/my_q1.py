import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from my_dataset import dataset
from model1_flow_capture import flow_capture

model = flow_capture('2023-wimbledon-1701')
model.run_prob_flow()
model.plot_p1_final_momentum()
data = model.get_p1_final_momentum() # series time+value
df = model.get_final_df() # df no+everything