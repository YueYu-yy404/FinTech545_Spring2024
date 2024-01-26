import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, t, ttest_1samp, norm
import statsmodels.api as sm
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro
import pylab
# import warnings
# warnings.filterwarnings("ignore")

data = pd.read_csv('problem2.csv')
data['constant'] = 1
x = data['x']
X = data[['constant','x']]
y = data ['y']

# define likelihood function
def MLE_Norm(params, x, y):
    yhat = params[0] + params[1]*x # predictions
    negLL = -1 * np.sum(norm.logpdf(y, yhat, params[2]))
    return(negLL)
results_norm = minimize(MLE_Norm, x0=(1,1,1), args=(x, y))
print(results_norm)

def MLE_mu_normal(X):
    n = len(X)
    return sum(X)/n

def MLE_sigma_normal(X):
    n = len(X)
    mu_hat = MLE_mu_normal(X)
    s = sum([(x-mu_hat)**2 for x in X])
    return s/n

import math
list_y = data['y'].tolist()
mu = MLE_mu_normal(y)
sigma_2 = MLE_sigma_normal(y)
print(mu)
print(sigma_2)



# 定义正态分布的对数似然函数
def log_likelihood(params, data):
    mean, std = params
    return -np.sum(norm.logpdf(data, loc=mean, scale=std))

# 初始猜测值
initial_guess = [1, 1]

# 使用Scipy的minimize函数进行MLE
result = minimize(log_likelihood, initial_guess, args=(y), method='Nelder-Mead')

# 输出结果
print("MLE估计的均值:", result.x[0])
print("MLE估计的标准差:", result.x[1])


def log_likelihood_t(params, data):
    mean, std, df = params
    return -np.sum(t.logpdf(data, df, loc=mean, scale=std))

# 初始猜测值
initial_guess = [0, 1, 1]

# 使用Scipy的minimize函数进行MLE
result = minimize(log_likelihood_t, initial_guess, args=(data,), method='Nelder-Mead')

# 输出结果
print("MLE估计的均值:", result.x[0])
print("MLE估计的标准差:", result.x[1])
print("MLE估计的自由度:", result.x[2])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

data = pd.read_csv('problem2_x.csv')

# Assume x1 and x2 are two columns of the data
x1 = data['x1'].values
x2 = data['x2'].values

# Combine x1 and x2 into a matrix X
X = np.column_stack((x1, x2))

# Define the log-likelihood function for multivariate normal distribution
mean_vector = np.mean(X, axis=0)
covariance_matrix = np.cov(X, rowvar=False)

# 创建多元正态分布对象
mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)

# 输出估计的均值和协方差矩阵
print("估计的均值向量：", mean_vector)
print("估计的协方差矩阵：", covariance_matrix)

data_x1 = pd.read_csv('problem2_x1.csv')

# 提取观测到的 x1 值
x1_observed = data_x1['x1'].values

# 从 'problem2_x.csv' 文件加载完整数据
data_full = pd.read_csv('problem2_x.csv')

# 提取完整数据中的 x2 列
x2_full = data_full['x2'].values

# 参数估计（均值和协方差矩阵）
mean_vector = np.mean(data_full[['x1', 'x2']], axis=0)
covariance_matrix = np.cov(data_full[['x1', 'x2']], rowvar=False)

# 初始化数组以存储模拟的 x2 值
simulated_x2_values = []

# 模拟 x2 的分布
for x1 in x1_observed:
    # 计算条件均值和方差
    conditional_mean_x2 = mean_vector[1] + covariance_matrix[1, 0] / covariance_matrix[0, 0] * (x1 - mean_vector[0])
    conditional_variance_x2 = covariance_matrix[1, 1] - covariance_matrix[1, 0] / covariance_matrix[0, 0] * covariance_matrix[0, 1]

    # 从正态分布中抽样生成 x2
    simulated_x2_value = np.random.normal(loc=conditional_mean_x2, scale=np.sqrt(conditional_variance_x2))
    
    # 将值添加到数组中
    simulated_x2_values.append(simulated_x2_value)

# 绘制模拟的 x2 分布图
plt.hist(simulated_x2_values, bins=30, density=True, alpha=0.5, color='blue', label='Simulated X2 Distribution')
plt.xlabel('x2')
plt.ylabel('Probability Density')
plt.title('Simulated Distribution of X2 given X1')
plt.legend()

# Shapiro-Wilk 检验
statistic, p_value = shapiro(simulated_x2_values)
print(f'Shapiro-Wilk Test Statistic: {statistic}, p-value: {p_value}')

# 绘制正态 Q-Q 图
sm.qqplot(np.array(simulated_x2_values), line='s')
plt.title('Normal Q-Q Plot')
plt.show()