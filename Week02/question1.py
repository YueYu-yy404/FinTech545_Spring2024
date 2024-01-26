import pandas as pd
import numpy as np
from scipy.stats import moment
 
 # Read CSV data from the specified file
df = pd.read_csv("problem1.csv")

# Extract values from the data column
data_values = df['x'].values

# Calculate the first four moments using the standardizing formula
mean_value = np.mean(data_values)
variance_value = moment(data_values, moment=2)
skewness_value = moment(data_values, moment=3)
kurtosis_value = moment(data_values, moment=4)

# Print the results
print(f"Mean: {mean_value:.2f}")
print(f"Variance: {variance_value:.2f}")
print(f"Skewness: {skewness_value:.2f}")
print(f"Kurtosis: {kurtosis_value:.2f}")