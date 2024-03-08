import unittest
import pandas as pd
import numpy as np
from myLibrary.functions import *

class TestDataFrames(unittest.TestCase):
    
    def assertDataFramesAlmostEqual(self, df1, df2, precision=3):
        """
        比较两个 DataFrame 是否相等，精度为指定的小数位数。
        """
        arr1 = df1.values
        arr2 = df2.values
        
        np.testing.assert_array_almost_equal(arr1, arr2, decimal=precision)
        
    def assertDataFramesAlmostEqualByTen(self, df1, df2):
    
        assert df1.shape == df2.shape, "DataFrame shapes are not the same"
        diff = np.abs(df1.values - df2.values)
        diff_by_ten = np.floor(diff / 10)
        assert np.all(diff_by_ten < 1), "DataFrames are not almost equal by ten"


    def test_cov_listwise(self):
        data1 = pd.read_csv('repo/testfiles/data/test1.csv')
        test1_1 = pd.read_csv('repo/testfiles/data/testout_1.1.csv')
        cov_likewise = calculate_cov(data1, True)
        
        self.assertDataFramesAlmostEqual(cov_likewise, test1_1)
        
    def test_corr_listwise(self):
        data1 = pd.read_csv('repo/testfiles/data/test1.csv')
        test1_2 = pd.read_csv('repo/testfiles/data/testout_1.2.csv')
        corr_listwise = calculate_corr(data1, True)
        
        self.assertDataFramesAlmostEqual(corr_listwise, test1_2)
        
    def test_cov_pairwise(self):
        data1 = pd.read_csv('repo/testfiles/data/test1.csv')
        test1_3 = pd.read_csv('repo/testfiles/data/testout_1.3.csv')
        cov_pairwise = calculate_cov(data1)
        
        self.assertDataFramesAlmostEqual(cov_pairwise, test1_3)
        
    def test_corr_pairwise(self):
        data1 = pd.read_csv('repo/testfiles/data/test1.csv')
        test1_2 = pd.read_csv('repo/testfiles/data/testout_1.4.csv')
        corr_listwise = calculate_corr(data1)
        
        self.assertDataFramesAlmostEqual(corr_listwise, test1_2)
        
    def test_ew_cov(self):
        data = pd.read_csv('repo/testfiles/data/test2.csv')
        test = pd.read_csv('repo/testfiles/data/testout_2.1.csv')
        ew_covariance = pd.DataFrame(ew_cov(data))
        
        self.assertDataFramesAlmostEqual(ew_covariance, test)
        
    def test_ew_corr(self):
        data = pd.read_csv('repo/testfiles/data/test2.csv')
        test = pd.read_csv('repo/testfiles/data/testout_2.2.csv')
        out = pd.DataFrame(ew_corr(data))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_cov_ew_cov_corr(self):
        data = pd.read_csv('repo/testfiles/data/test2.csv')
        test = pd.read_csv('repo/testfiles/data/testout_2.3.csv')
        out = pd.DataFrame(cov_with_ew_var_corr(data))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_near_psd(self):
        data = pd.read_csv('repo/testfiles/data/testout_1.3.csv')
        test = pd.read_csv('repo/testfiles/data/testout_3.1.csv')
        out = pd.DataFrame(near_psd(data))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_near_psd_corr(self):
        data = pd.read_csv('repo/testfiles/data/testout_1.4.csv')
        test = pd.read_csv('repo/testfiles/data/testout_3.2.csv')
        out = pd.DataFrame(near_psd(data))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_higham_psd(self):
        data = pd.read_csv('repo/testfiles/data/testout_1.3.csv')
        test = pd.read_csv('repo/testfiles/data/testout_3.3.csv')
        out = pd.DataFrame(Higham_psd(data))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_higham_psd_corr(self):
        data = pd.read_csv('repo/testfiles/data/testout_1.4.csv')
        test = pd.read_csv('repo/testfiles/data/testout_3.4.csv')
        out = pd.DataFrame(Higham_psd(data))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_chol_psd(self):
        data = pd.read_csv('repo/testfiles/data/testout_3.1.csv')
        test = pd.read_csv('repo/testfiles/data/testout_4.1.csv')
        out = pd.DataFrame(chol_psd(data.to_numpy()))
        
        self.assertDataFramesAlmostEqual(out, test)
    
    def test_5_1(self):
        data = pd.read_csv('repo/testfiles/data/test5_1.csv')
        test = pd.read_csv('repo/testfiles/data/testout_5.1.csv')
        samples = multivariate_normal_simulation(data.to_numpy(),100000)
        out = calculate_cov(pd.DataFrame(samples))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_5_2(self):
        data = pd.read_csv('repo/testfiles/data/test5_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout_5.2.csv')
        samples = multivariate_normal_simulation(data.to_numpy(),100000)
        out = calculate_cov(pd.DataFrame(samples))
        
        self.assertDataFramesAlmostEqual(out, test,precision=2)
        
    def test_5_3(self):
        data = pd.read_csv('repo/testfiles/data/test5_3.csv')
        test = pd.read_csv('repo/testfiles/data/testout_5.3.csv')
        samples = multivariate_normal_simulation(data.to_numpy(),100000,fix_method='near_psd')
        out = calculate_cov(pd.DataFrame(samples))
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_5_4(self):
        data = pd.read_csv('repo/testfiles/data/test5_3.csv')
        test = pd.read_csv('repo/testfiles/data/testout_5.4.csv')
        samples = multivariate_normal_simulation(data.to_numpy(),100000,fix_method='higham')
        out = calculate_cov(pd.DataFrame(samples))
        
        self.assertDataFramesAlmostEqual(out, test,precision=2)

    def test_5_4(self):
        data = pd.read_csv('repo/testfiles/data/test5_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout_5.5.csv')
        samples = multivariate_normal_simulation(data.to_numpy(),100000,method='pca')
        out = calculate_cov(pd.DataFrame(samples))
        
        self.assertDataFramesAlmostEqual(out, test,2)
        
    def test_6_1(self):
        data = pd.read_csv('repo/testfiles/data/test6.csv')
        test = pd.read_csv('repo/testfiles/data/test6_1.csv')
        out = return_calculate(data)
        
        self.assertDataFramesAlmostEqual(out.iloc[:,1:], test.iloc[:,1:])
        
    def test_6_2(self):
        data = pd.read_csv('repo/testfiles/data/test6.csv')
        test = pd.read_csv('repo/testfiles/data/test6_2.csv')
        out = return_calculate(data, method='LOG')
        
        self.assertDataFramesAlmostEqual(out.iloc[:,1:], test.iloc[:,1:])
        
    def test_7_1(self):
        data = pd.read_csv('repo/testfiles/data/test7_1.csv')
        test = pd.read_csv('repo/testfiles/data/testout7_1.csv')
        mu, sigma = fit_normal(data)
        out = pd.DataFrame({'mu': [mu], 'sigma': [sigma]})
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_7_2(self):
        data = pd.read_csv('repo/testfiles/data/test7_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout7_2.csv')
        df, loc, scale = fit_t(data)
        out = pd.DataFrame({'mu': [loc], 'sigma': [scale], 'nu':[df]})
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_7_3(self):
        data = pd.read_csv('repo/testfiles/data/test7_3.csv')
        test = pd.read_csv('repo/testfiles/data/testout7_3.csv')
        y = data[['y']].values
        x = data[['x1', 'x2', 'x3']].values
        nu, m, s, fitted_model = fit_regression_t(y,x)
        
        out = pd.DataFrame({'mu': [m], 'sigma': [s], 'nu':[nu], 
                            'Alpha': fitted_model.beta[0], 'B1':fitted_model.beta[1],
                            'B2':fitted_model.beta[2],'B3':fitted_model.beta[3]})
        
        self.assertDataFramesAlmostEqual(out, test,2)
        
    def test_8_1(self):
        data = pd.read_csv('repo/testfiles/data/test7_1.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_1.csv')
        mu, sigma = fit_normal(data)
        var_normal = VaR_norm(mu, sigma)
        var_diff = VaR_norm(0,sigma)
        out = pd.DataFrame({'var_normal': [var_normal], 'var_diff': [var_diff]})
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_8_1(self):
        data = pd.read_csv('repo/testfiles/data/test7_1.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_1.csv')
        mu, sigma = fit_normal(data)
        var_normal = VaR_norm(mu, sigma)
        var_diff = VaR_norm(0,sigma)
        out = pd.DataFrame({'var_normal': [var_normal], 'var_diff': [var_diff]})
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_8_2(self):
        data = pd.read_csv('repo/testfiles/data/test7_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_2.csv')
        df, loc, scale = fit_t(data)
        var_t = VaR_t(df, loc, scale)
        var_diff = VaR_t(df, 0, scale)
        out = pd.DataFrame({'var_normal': [var_t], 'var_diff': [var_diff]})
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_8_3(self):
        data = pd.read_csv('repo/testfiles/data/test7_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_3.csv')
        df, loc, scale = fit_t(data)
        t_dist = t(df, loc, scale)
        simulations = t_dist.ppf(np.random.rand(10000))
        var_t = simulation_VaR(simulations)
        var_diff = simulation_VaR(simulations-simulations.mean())
        out = pd.DataFrame({'var_normal': [var_t], 'var_diff': [var_diff]})
        
        self.assertDataFramesAlmostEqual(out, test,2)
        
    def test_8_4(self):
        data = pd.read_csv('repo/testfiles/data/test7_1.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_4.csv')
        mu, sigma = fit_normal(data)
        es_norm = ES_norm(mu, sigma, 0.05)
        norm_diff = ES_norm(0, sigma, 0.05)
        out = pd.DataFrame({'var_normal': [es_norm], 'var_diff': [norm_diff]})
        
        self.assertDataFramesAlmostEqual(out, test)
    
    def test_8_5(self):
        data = pd.read_csv('repo/testfiles/data/test7_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_5.csv')
        df, loc, scale = fit_t(data)
        es_t = ES_t(df, loc, scale)
        t_diff = ES_t(df, 0, scale)
        out = pd.DataFrame({'var_normal': [es_t], 'var_diff': [t_diff]})
        
        self.assertDataFramesAlmostEqual(out, test)
        
    def test_8_6(self):
        data = pd.read_csv('repo/testfiles/data/test7_2.csv')
        test = pd.read_csv('repo/testfiles/data/testout8_6.csv')
        df, loc, scale = fit_t(data)
        t_dist = t(df, loc, scale)
        simulations = t_dist.ppf(np.random.rand(10000))
        es_t = simulation_ES(simulations)
        es_diff = simulation_ES(simulations-simulations.mean())
        out = pd.DataFrame({'var_normal': [es_t], 'var_diff': [es_diff]})
        
        self.assertDataFramesAlmostEqual(out, test,2)
        
    def test_9_1(self):
        returns = pd.read_csv('repo/testfiles/data/test9_1_returns.csv')
        portfolios = pd.read_csv('repo/testfiles/data/test9_1_portfolio.csv')
        test = pd.read_csv('repo/testfiles/data/testout9_1.csv')
        portfolios['currentValue'] = portfolios['Holding']*portfolios['Starting Price']
        portfolio = portfolios.loc[:,['Stock','currentValue']]
        a_returns = returns['A']
        b_returns = returns['B']
        df, loc, scale = fit_t(b_returns)
        mu, sigma = fit_normal(a_returns)
        u_b = t.cdf(b_returns, df, loc, scale)
        u_a = norm.cdf(a_returns, mu, sigma)
        U = np.concatenate([u_a,u_b])
        spcor = spearmanr(u_a,u_b, axis = 0)[0]
        nSim = 100000
        cor = np.array([[1,spcor],[spcor, 1]])
        uSim = multivariate_normal_simulation(cor, nSim,method = 'pca')
        uSim = norm.cdf(uSim,loc=0,scale=1)
        a_eval = norm.ppf(uSim[:,0], mu, sigma)
        b_eval = t.ppf(uSim[:,1],df, loc, scale)
        simRet = pd.DataFrame({'A': a_eval, 'B': b_eval})
        iterations = pd.DataFrame({'iteration': [i for i in range(1, nSim + 1)]})
        
        values = pd.merge(portfolio, iterations, how='cross')
        nv = len(values)  # Assuming 'values' is a DataFrame as constructed before
        simulatedValue = [0] * nv  # Initialize a list with zeros
        pnl = [0] * nv  # Initialize a list with zeros

        for i in range(nv):
            iteration_raw = values.iloc[i]['iteration']
            # Reset iteration to 1 after reaching 100000
            iteration = (iteration_raw % 100000) if iteration_raw == 100000 else iteration_raw
            stock = values.iloc[i]['Stock']
            currentValue = values.iloc[i]['currentValue']
            
            # Ensure that 'simRet' is indexed or accessed correctly; this might need adjustment
            # Assuming 'simRet' has a multi-level index of 'iteration' and 'Stock' or a similar structure
            ret = simRet.loc[iteration, stock]
            simulatedValue[i] = currentValue * (1 + ret)
            pnl[i] = simulatedValue[i] - currentValue

        # Assign 'pnl' and 'simulatedValue' to 'values' DataFrame
        values['pnl'] = pnl
        values['simulatedValue'] = simulatedValue
        
        pnl_A = values[values['Stock'] == 'A']['pnl']
        pnl_B = values[values['Stock'] == 'B']['pnl']
        # for function in [VaR_norm,ES_norm]:

        mu, sigma = fit_normal(pnl_A)
        df, loc, scale = fit_t(pnl_B)
        value_temp = values.loc[values['Stock']== 'B', 'currentValue'].reset_index(drop=True)
        value_temp = value_temp[0]

        var_es = {}
        out = pd.DataFrame(columns=['Stock', 'VaR95', 'ES95', 'VaR95_pct', 'ES95_pct'])
        for stock_name in ['A','B']:
            value_temp = values.loc[values['Stock']== stock_name, 'currentValue'].reset_index(drop=True).iloc[0]
            if stock_name == 'A':
                var = VaR_norm(mu,sigma)
                es = ES_norm(mu, sigma)
            elif stock_name == 'B':
                var = VaR_t(df, loc, scale,alpha=0.05)
                es = ES_t(df, loc, scale,alpha=0.05)
            var_pct = var/value_temp
            es_pct = es/value_temp
            new_row = pd.DataFrame({'Stock': [stock_name], 'VaR95': [var], 'ES95': [es], 'VaR95_pct': [var_pct], 'ES95_pct': [es_pct]})
            out = pd.concat([out, new_row], ignore_index=True)
        
        gdf = values.groupby('iteration')
        totalValues = gdf.aggregate({
            'currentValue': 'sum',
            'simulatedValue': 'sum',
            'pnl': 'sum'
        }).reset_index()
        pnl_sum = totalValues['pnl']
        var_total = VaR_t(fit_t(pnl_sum)[0], fit_t(pnl_sum)[1],fit_t(pnl_sum)[2], alpha=0.05)
        es_total = ES_t(fit_t(pnl_sum)[0], fit_t(pnl_sum)[1],fit_t(pnl_sum)[2], alpha=0.05)
        value_temp = totalValues['currentValue'][0]
        new_row = pd.DataFrame({'Stock': ['Total'], 'VaR95': [var_total], 'ES95': [es_total], 'VaR95_pct': [var_total/value_temp], 'ES95_pct': [es_total/value_temp]})
        out = pd.concat([out, new_row], ignore_index=True)
        self.assertDataFramesAlmostEqualByTen(out.iloc[:,1:], test.iloc[:,1:])

if __name__ == '__main__':
    unittest.main()