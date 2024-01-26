import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load Data
data = pd.read_csv('problem2.csv')
data = pd.DataFrame(data)

# Data Preparation
# Add a constant column ('constant') to the DataFrame
data['constant'] = 1

# Define Variables
# Identify the dependent variable (y) and independent variable (x)
x = data['x']
y = data['y']

# Prepare Feature Matrix
# Create a feature matrix (X) by combining the constant column and the independent variable
X = data[['constant', 'x']]

# Fit OLS Model
# Use the statsmodels library to fit an OLS regression model
est = sm.OLS(y, X)
model = est.fit()

# Calculate Residuals
# Get the residuals (errors) by subtracting predicted values from actual values
residuals = y - model.predict(X)

# Calculate Standard Deviation of Residuals
# Compute the standard deviation of residuals
std_error = np.std(residuals)

print("Standard Deviation of OLS Error:", std_error)


# Display Model Summary
# Print and display the summary of the fitted OLS regression model
print(model.summary())

