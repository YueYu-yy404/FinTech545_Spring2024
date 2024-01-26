import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, probplot, norm
import matplotlib.pyplot as plt

# Load data from 'problem2_x.csv'
data = pd.read_csv('problem2_x.csv')

# Assume x1 and x2 are two columns of the data
x1 = data['x1'].values
x2 = data['x2'].values

# Combine x1 and x2 into a matrix X
X = np.column_stack((x1, x2))

# Define the log-likelihood function for a multivariate normal distribution
mean_vector = np.mean(X, axis=0)
covariance_matrix = np.cov(X, rowvar=False)

# Create a multivariate normal distribution object
mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)

# Output the estimated mean vector and covariance matrix
print("Estimated mean vector:", mean_vector)
print("Estimated covariance matrix:", covariance_matrix)

# Known x1 values
observed_data = pd.read_csv('problem2_x1.csv')
observed_x1 =  observed_data['x1'].values

# Calculate the mean and covariance matrix for x2 given x1
mean_conditional_x2 = mean_vector[1:] 
cov_matrix_conditional_x2 = covariance_matrix[1:, 1:]
print("Mean conditional x2:", mean_conditional_x2)
print("Covariance matrix conditional x2:", cov_matrix_conditional_x2)

# Create a multivariate normal distribution object for x2 given x1
conditional_mvn = multivariate_normal(mean=mean_conditional_x2, cov=cov_matrix_conditional_x2)

# Estimate corresponding x2 values
estimated_x2 = conditional_mvn.rvs(size=len(observed_x1))

# Calculate the mean and confidence interval for the estimated x2
mean_estimated_x2 = np.mean(estimated_x2)
std_estimated_x2 = np.std(estimated_x2)
confidence_interval = norm.interval(0.95, loc=mean_estimated_x2, scale=std_estimated_x2)

# Plot histogram of estimated x2 values
plt.hist(estimated_x2, bins=30, density=True, alpha=0.7, color='blue', label='Histogram of Estimated x2')
plt.title('Histogram of Estimated x2 Values')
plt.xlabel('x2 Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot Q-Q plot with the expected value and 95% confidence interval
probplot(estimated_x2, dist='norm', plot=plt)
plt.axhline(mean_estimated_x2, color='red', linestyle='dashed', label='Expected Value')
plt.axhline(confidence_interval[0], color='green', linestyle='dashed', label='95% Confidence Interval')
plt.axhline(confidence_interval[1], color='green', linestyle='dashed')
plt.title('Q-Q Plot of Estimated x2 with Expected Value and 95% Confidence Interval')
plt.legend()
plt.show()
