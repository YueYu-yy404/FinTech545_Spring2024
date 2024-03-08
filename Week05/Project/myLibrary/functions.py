import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, kurtosis, spearmanr
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize 
from scipy.integrate import quad

def calculate_cov(data, skipMiss = False):
    if skipMiss == True:
        data = data.dropna()
    cov_matrix = data.cov()
    return cov_matrix

def calculate_corr(data, skipMiss = False):
    if skipMiss == True:
        data = data.dropna()
    corr_matrix = data.corr(method='pearson')
    return corr_matrix

# EW Covariance, lambda=0.97
def ew_cov(returns, lambda_=0.97):
    """
    Perform calculation on the input data set with a given λ for exponentially weighted covariance.
    
    Parameters:
    - data: input data set, a pandas DataFrame
    - lambda_: fraction for unpdate the covariance matri, default 0.97
    
    Returns:
    cov: an exponentially weighted covariance matrix, a numpy array
    """

    # Preprocess the data
    returns = returns.values
    mean_return = np.mean(returns, axis=0)
    normalized_returns = returns - mean_return
    
    # Initializing the covariance matrix
    n_timesteps, n_stocks = normalized_returns.shape
    weights = np.zeros(n_timesteps)
    
    # Compute the weight for each time step
    for t in range(n_timesteps):
        weights[n_timesteps-1-t]  = (1-lambda_)*lambda_**t
    
    # Normalize the weights_matrix
    weights_matrix = np.diag(weights/sum(weights))

    cov = np.transpose(normalized_returns) @ weights_matrix @ normalized_returns
    
    return cov

# EW Correlation, lambd=0.94
def ew_corr(returns, lamda_ = 0.94):
    
    # Calculate the exponentially weighted covariance matrix
    ew_cov_matrix = ew_cov(returns, lamda_)

    # Initialize the correlation matrix with zeros
    ew_corr_matrix = pd.DataFrame(np.zeros((returns.shape[1], returns.shape[1])), index=returns.columns, columns=returns.columns)
    
    for i in range(returns.shape[1]): # Iterate over columns
        for j in range(returns.shape[1]):
            if i == j:
                # Correlation with itself is 1
                ew_corr_matrix.iloc[i, j] = 1.0
            else:
                # Normalize the covariance with the square root of the product of the variances
                ew_corr_matrix.iloc[i, j] = ew_cov_matrix[i, j] / np.sqrt(ew_cov_matrix[i][i]* ew_cov_matrix[j][j])
    
    return ew_corr_matrix

# Covariance with EW Variance (l=0.97), EW Correlation (l=0.94)
def cov_with_ew_var_corr(returns, var_lamda=0.97, corr_lamda=0.94):
    
    ew_cov_matrix = ew_cov(returns, var_lamda)
    ew_correlation = ew_corr(returns, corr_lamda)
    adjusted_cov = pd.DataFrame(np.zeros((returns.shape[1], returns.shape[1])), index=returns.columns, columns=returns.columns)
    for i in range(ew_correlation.shape[0]):
        for j in range(ew_correlation.shape[0]):
            adjusted_cov.iloc[i, j] = ew_correlation.iloc[i,j] * np.sqrt(ew_cov_matrix[i][i] * ew_cov_matrix[j][j])

    return adjusted_cov

# near_psd covariance
def near_psd(matrix, epsilon=0.0):
    """
    Calculates a near positive semi-definite (PSD) matrix from a given non-PSD matrix.

    Parameters:
    - matrix: The input matrix, a 2-dimensional numpy array
    - epsilon: A small non-negative value used to ensure that the resulting matrix is PSD, default value is 0.0

    Returns:
    The output of this function is a 2-dimensional numpy array that represents a near PSD matrix. 
    """
    n = matrix.shape[0]

    invSD = None
    out = matrix.copy()

    # calculate the correlation matrix if we got a covariance
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = np.matmul(np.matmul(invSD, out), invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.matmul(np.square(vecs), vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.matmul(np.matmul(T, vecs), l)
    out = np.matmul(B, np.transpose(B))

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = np.matmul(np.matmul(invSD, out), invSD)

    return out

# Higham covariance
def Pu(matrix):
    """The first projection for Higham method with the assumption that weight martrix is diagonal."""
    result = matrix.copy()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i==j:
                result[i][i]=1
    return result

def Ps(matrix, weight):
    """The second projection for Higham method."""
    matrix = np.sqrt(weight)@ matrix @np.sqrt(weight)
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.array([max(i,0) for i in vals])
    result = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
    return result

def Frobenius_Norm(matrix_1, matrix_2):
    distance = matrix_1 - matrix_2
    result = 0
    for i in range(len(distance)):
        for j in range(len(distance)):
            result += distance[i][j]**2
    return result

def Higham_psd(matrix, weight = None, epsilon = 1e-9, max_iter = 1000, tolerance = 1e-8):
    """
    Calculates a near positive semi-definite (PSD) matrix from a given non-PSD matrix.

    Parameters:
    - matrix: The input covariance matrix, a 2-dimensional numpy array
    - weight: Assume weight is a diagonal matrix, if unweighted, set 𝑊 = 𝐼
    - epsilon: Used to check the smallest eigenvalue from the result
    - max_iter: Restriction on the maximum iteration loops
    - tolerance: A small non-negative value used to restrict the distance for the original matrix, default value is 1e-8

    Returns:
    The output of this function is a 2-dimensional numpy array that represents a nearest PSD matrix. 
    """
    if weight is None:
        weight = np.identity(len(matrix))
        
    norml = np.inf
    Yk = matrix.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != matrix.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = np.matmul(np.matmul(invSD, Yk), invSD)
    
    Y0 = Yk.copy()

    for i in range(max_iter):
        Rk = Yk - Delta_S
        Xk = Ps(Rk, weight)
        Delta_S = Xk - Rk
        Yk = Pu(Xk)
        norm = Frobenius_Norm(Yk, Y0)
        minEigVal = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < tolerance and minEigVal > -epsilon:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = np.matmul(np.matmul(invSD, Yk), invSD)
    return Yk

# chol_psd
def chol_psd(cov_matrix):
    """
    Perform Cholesky decomposition on the input matrix `covariance`.
    
    Parameters:
    - cov_matrix: input matrix, a numpy array with shape (n_samples, n_samples)
    
    Returns:
    The Cholesky decomposition of the input matrix `covariance`.
    """
    n = cov_matrix.shape[0]
    root = np.zeros_like(cov_matrix)
    for j in range(n):
        s = 0.0
        if j > 0:
            # calculate dot product of the preceeding row values
            s = np.dot(root[j, :j], root[j, :j])
        temp = cov_matrix[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            # set the column to 0 if we have an eigenvalue of 0
            root[j + 1:, j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (cov_matrix[i, j] - s) * ir
    return root

def multivariate_normal_simulation(covariance_matrix, n_samples, method='direct', fix_method='chol_psd', mean = 0, explained_variance=1.0):
    """
    A function to simulate multivariate normal distributions with different methods.
    
    Parameters:
    - covariance_matrix (np.array): The covariance matrix for the multivariate normal distribution
    - n_samples (int): The number of samples to generate
    - method (str, optional): The method to use for simulation, either 'direct' or 'pca', default 'direct'
         'direct': simulate directly from the covariance matrix.
         'pca': simulate using principal component analysis (PCA).
    - explained_variance (float, optional): The percentage of explained variance to keep when using PCA, default 1.0
    
    Returns:
     np.array: An array with shape (covariance_matrix.shape[0], n_samples) with the simulated samples.
    """
    
    # If the method is 'direct', simulate directly from the covariance matrix
    if method == 'direct' and fix_method=='chol_psd':
        
        L = chol_psd(covariance_matrix)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))
        
        samples = np.transpose(np.dot(L, normal_samples) + mean)
        
        return samples
    
    # If the method is 'pca', simulate using PCA
    elif method == 'pca' and fix_method=='chol_psd':
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Only consider eigenvalues greater than 0
        idx = eigenvalues > 1e-8
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Sort the eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Update the explained_variance incase the explained_variance is higher than the cumulative sum of the eigenvalue
        if explained_variance == 1.0:
            explained_variance = (np.cumsum(eigenvalues)/np.sum(eigenvalues))[-1]
        
        # Determine the number of components to keep based on the explained variance ratio
        n_components = np.where((np.cumsum(eigenvalues)/np.sum(eigenvalues))>= explained_variance)[0][0] + 1
        eigenvectors = eigenvectors[:,:n_components]
        eigenvalues = eigenvalues[:n_components]

        normal_samples = np.random.normal(size=(n_components, n_samples))
        
        # Simulate the multivariate normal samples by multiplying the eigenvectors with the normal samples
        B = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
        samples = np.transpose(np.dot(B, normal_samples))
        
        return samples

    elif method == 'direct' and fix_method=='near_psd':
        psd = near_psd(covariance_matrix)
        L = chol_psd(psd)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))
        
        samples = np.transpose(np.dot(L, normal_samples) + mean)
        
        return samples

    elif method == 'direct' and fix_method == 'higham':
        psd = Higham_psd(covariance_matrix)
        L = chol_psd(psd)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))
        
        samples = np.transpose(np.dot(L, normal_samples) + mean)
        
        return samples
    
    
def return_calculate(prices, method="DISCRETE", date_column="Date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {list(prices.columns)}")
    
    # Exclude the date column from the calculation
    vars = [col for col in prices.columns if col != date_column]
    
    # Convert prices to a matrix (numpy array) for calculations
    p = prices[vars].values
    n, m = p.shape
    p2 = np.empty((n-1, m))
    
    # Vectorized calculation for performance
    p2 = p[1:, :] / p[:-1, :]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")
    
    # Create a new DataFrame to store the returns along with the corresponding dates
    dates = prices[date_column].iloc[1:].reset_index(drop=True)
    out = pd.DataFrame(data=p2, columns=vars)
    out.insert(0, date_column, dates)
    
    return out

# Fit Normal Distribution
def fit_normal(data, lambda_=0.97, method = 'normal'):
    if method == 'ew':
        mu = np.mean(data)
        sigma_ = np.sqrt(ew_cov(data, lambda_=lambda_))
        sigma = sigma_[0][0]
        return mu, sigma
    else:
        mu, sigma = norm.fit(data)
        return mu, sigma

# Fit T Distribution
def MLE_T(params, returns):
    negLL = -1 * np.sum(t.logpdf(returns, df=params[0], loc=params[1], scale=params[2]))
    return(negLL)
def fit_t(data):
    constraints=({"type":"ineq", "fun":lambda x: x[0]-1}, {"type":"ineq", "fun":lambda x: x[2]})
    returns_t = minimize(MLE_T, x0=[10, np.mean(data), np.std(data, ddof=1).item() ], args=data, constraints=constraints)
    df, loc, scale = returns_t.x[0], returns_t.x[1], returns_t.x[2]
    return df, loc, scale

class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval = eval_func
        self.errors = errors
        self.u = u

def neg_log_likelihood_t(params, x, y):
    mu, s, nu, *b = params
    b = np.array(b)[np.newaxis, :]
    xm = y - np.sum(np.hstack((np.ones((x.shape[0], 1)), x)) * b, axis=1).reshape(x.shape[0], 1)
    log_likelihood = np.sum(t.logpdf(x=xm, df=nu, loc=mu, scale=s))
    return -log_likelihood

def fit_regression_t(y, x):
    n = x.shape[0]
    # Approximate values based on moments and OLS
    b_start = np.linalg.lstsq(np.hstack((np.ones((n, 1)), x)), y, rcond=None)[0]
    e = y - np.sum(np.hstack((np.ones((n, 1)), x)) * b_start.T, axis=1).reshape(n, 1)
    start_m = np.mean(e.flatten())
    start_nu = 6.0 / kurtosis(e.flatten()) + 4
    start_s = np.sqrt(np.var(e.flatten()) * (start_nu - 2) / start_nu)
    print(sum(e))

    # Define the initial guess for the parameters
    initial_params = np.concatenate([[[start_m]], [[start_s]], [[start_nu]], b_start]).flatten()
    # Define bounds for the parameters
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * (x.shape[1] + 1)

    # Minimize the negative log-likelihood function
    result = minimize(neg_log_likelihood_t, initial_params, args=(x, y), bounds=bounds)

    # Retrieve optimized values
    m, s, nu, *beta = result.x

    # Define the fitted error model
    error_model = t(df=nu, loc=m, scale=s)

    # Function to evaluate the model for a given x and u
    def eval_model(x, u):
        n = x.shape[0]
        _temp = np.hstack((np.ones((n, 1)), x))
        return np.dot(_temp, beta) + error_model.ppf(u) # add estimated error

    # Calculate the regression errors and their U values
    errors = y - eval_model(x, np.full((x.shape[0],), 0.5))
    u = error_model.cdf(errors)

    return nu, m, s, FittedModel(beta, error_model, eval_model, errors, u)

def VaR_norm(mu, sigma, alpha=0.05):
    return -norm.ppf(alpha, loc=mu, scale=sigma)

def VaR_t(df, loc, scale, alpha = 0.05):
    t_dist = t(df, loc, scale)
    return -t_dist.ppf(alpha)

def VaR_hist(returns, alpha=0.05):
    """Calculate VaR using historic simulation"""
    var_hist = -np.percentile(returns, alpha*100)
    es_hist = -np.mean(returns[returns <= -var_hist])
    return var_hist, es_hist

def simulation_VaR(simulations, alpha = 0.05):
    return -np.percentile(simulations, alpha*100)

def ES_norm(mu, sigma, alpha=0.05):
    # Calculate the VaR at the given alpha level using the provided VaR function
    v = VaR_norm(mu, sigma, alpha)
    
    # Define the normal distribution with the specified mean and standard deviation
    d = norm(loc=mu, scale=sigma)
    
    # Define the function to integrate
    f = lambda x: x * d.pdf(x)
    
    # Determine the starting point for integration (close to the minimum of the distribution's support)
    st = d.ppf(1e-12)
    print(st)
    
    # Perform the integration from st to -v
    result, _ = quad(f, st, -v) 
    print(result)
    # Note: the integration bounds have been corrected to 'v' instead of '-v'
    
    # Calculate and return the Expected Shortfall
    return -result / alpha


def ES_t(df, loc, scale, alpha=0.05):
    # Calculate the VaR at the given alpha level using the provided VaR function
    v = VaR_t(df, loc, scale, alpha)
    
    # Define the normal distribution with the specified mean and standard deviation
    d = t(df=df, loc = loc, scale = scale)
    
    # Define the function to integrate
    f = lambda x: x * d.pdf(x)
    
    # Determine the starting point for integration (close to the minimum of the distribution's support)
    st = d.ppf(1e-12)
    print(st)
    
    # Perform the integration from st to -v
    result, _ = quad(f, st, -v) 
    print(result)
    # Note: the integration bounds have been corrected to 'v' instead of '-v'
    
    # Calculate and return the Expected Shortfall
    return -result / alpha

def simulation_ES(simulations, alpha = 0.05):
    v = simulation_VaR(simulations, alpha=alpha)
    es = -np.mean(simulations[simulations <= -v])
    return es

