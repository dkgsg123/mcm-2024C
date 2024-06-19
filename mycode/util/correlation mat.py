import numpy as np

np.random.seed(100)

#create array of 50 random integers between 0 and 10
var1 = np.random.randint(0, 10, 50)

#create a positively correlated array with some random noise
var2 = var1 + np.random.normal(0, 10, 50)

#calculate the correlation between the two arrays
print(np.corrcoef(var1, var2))