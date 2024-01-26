import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. Import Data
data = pd.read_csv('problem3.csv')
# Assuming the data column is named 'value', please change the column name accordingly
time_series = data['x']

# Fit ARIMA models with different AR and MA orders
# AR(1) Models
for i in range(3):
    # 2. Fit ARIMA Model from AR(1) through AR(3)
    p = i+1  # AR order
    d = 0    # Integrated order
    q = 0    # MA order

    # Fit ARIMA model
    model = ARIMA(time_series, order=(p, d, q))
    fit_model = model.fit()

    # 3. Print Model Summary
    print(fit_model.summary())

    # 4. Plot Original Data and Fitted Values
    plt.plot(time_series, label='Actual')
    plt.plot(fit_model.fittedvalues, color='red', label='Fitted')
    plt.legend()
    plt.title(f'ARIMA({p}, {d}, {q}) Fit')
    plt.show()

    # 5. Calculate MSE
    # Split the dataset
    train_size = int(len(time_series) * 0.8)
    train, test = time_series[:train_size], time_series[train_size:]

    # Make predictions on the test set
    predictions = fit_model.forecast(steps=len(test))

    # Calculate MSE
    mse = mean_squared_error(test, predictions)

    # Print MSE
    print(f'AR({p}) MSE: {mse}')