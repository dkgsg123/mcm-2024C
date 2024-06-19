import numpy as np

def zscore(data):
    mean_value = np.mean(data, axis=0)
    std_deviation = np.std(data, axis=0)

    zscore_data = (data - mean_value) / std_deviation

    return zscore_data
