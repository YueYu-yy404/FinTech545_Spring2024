import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Load data from 'problem2.csv'
data = pd.read_csv('problem2.csv')

# Extract 'x' and 'y' columns from the data
x = data['x']
y = data['y']

# Define the log-likelihood function for a normal distribution
def log_likelihood(params, data):
    # 'params' contains mean and standard deviation
    mean, std = params
    
    # Calculate the negative log-likelihood
    return -np.sum(norm.logpdf(data, loc=mean, scale=std))

# Initial guess for mean and standard deviation
initial_guess = [1, 1]

# Use Scipy's minimize function for Maximum Likelihood Estimation (MLE)
result = minimize(log_likelihood, initial_guess, args=(y,), method='Nelder-Mead')

# Output the MLE results
print("MLE estimated mean:", result.x[0])
print("MLE estimated standard deviation:", result.x[1])

# Calculate Log Likelihood
log_likelihood_value = -result.fun

# Calculate AIC
n_params = len(result.x)
n_observations = len(y)
aic = 2 * n_params - 2 * log_likelihood_value

# Calculate BIC
bic = n_params * np.log(n_observations) - 2 * log_likelihood_value

print("Log Likelihood:", log_likelihood_value)
print("AIC:", aic)
print("BIC:", bic)

def MLE_Norm(params, x, y):
    yhat = params[0] + params[1]*x # predictions
    negLL = -1 * np.sum(norm.logpdf(y, yhat, params[2]))
    return(negLL)
results_norm = minimize(MLE_Norm, x0=(1,1,1), args=(x, y))
print(results_norm)

