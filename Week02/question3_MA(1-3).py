import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. 导入数据
data = pd.read_csv('problem3.csv')
# 假设数据列为 'value'，请根据实际情况更改列名
time_series = data['x']
    
for j in range(3):
    # 2. 拟合ARIMA模型 (AR(1))
    p = 0  # AR order
    d = 0  # Integrated order
    q = j+1  # MA order

    # 拟合ARIMA模型
    model = ARIMA(time_series, order=(p, d, q))
    fit_model = model.fit()

    # 3. 打印模型摘要
    print(fit_model.summary())

    # 4. 绘制原始数据和拟合结果
    plt.plot(time_series, label='Actual')
    plt.plot(fit_model.fittedvalues, color='red', label='Fitted')
    plt.legend()
    plt.title(f'ARIMA({p}, {d}, {q}) Fit')
    plt.show()
