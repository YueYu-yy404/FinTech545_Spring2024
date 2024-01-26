import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

# Load data from 'problem2.csv'
data = pd.read_csv('problem2.csv')

# Extract 'x' and 'y' columns from the data
x = data['x']
y = data['y']

def log_likelihood_t(params, data):
    # Define a log-likelihood function for a t-distribution
    mean, std, df = params
    return -np.sum(t.logpdf(data, df, loc=mean, scale=std))

# Initial guess for the parameters
initial_guess = [0, 1, 1]

# Use Scipy's minimize function for Maximum Likelihood Estimation (MLE)
result = minimize(log_likelihood_t, initial_guess, args=(data,), method='Nelder-Mead')

# Calculate Log Likelihood
log_likelihood = -result.fun

# Calculate AIC
n_params = len(result.x)
n_observations = len(y)
aic = 2 * n_params - 2 * log_likelihood

# Calculate BIC
bic = n_params * np.log(n_observations) - 2 * log_likelihood

# Output the results
print("MLE estimate of mean:", result.x[0])
print("MLE estimate of standard deviation:", result.x[1])
print("MLE estimate of degrees of freedom:", result.x[2])
print("Log Likelihood:", log_likelihood)
print("AIC:", aic)
print("BIC:", bic)

