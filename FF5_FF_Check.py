import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import f
from numpy.linalg import inv


# FF factor returns 
FF = pd.read_csv('FF5_monthly.csv')
FF['date'] = pd.to_datetime(FF['date'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
FF = FF[(FF['date'] >= '1963-07') & (FF['date'] <= '2014-01')]
FF['Rm-rf'] = FF['Mkt-RF'] / 100
FF['SMB'] = FF['SMB'] / 100
FF['HML'] = FF['HML'] / 100
FF['RMW'] = FF['RMW'] / 100
FF['CMA'] = FF['CMA'] / 100
FF['RF'] = FF['RF'] / 100
FF['Rm-rf_cum'] =((1+FF['Rm-rf']).cumprod())
FF['SMB_cum'] = ((1+FF['SMB']).cumprod())
FF['HML_cum'] = ((1+FF['HML']).cumprod())
FF['RMW_cum'] = ((1+FF['RMW']).cumprod())
FF['CMA_cum'] = ((1+FF['CMA']).cumprod())

def run_time_series_regressions(portfolio_df, factor_df):
    merged = portfolio_df.merge(factor_df, on='date', how='inner')
    merged['excess_return'] = merged['portfolio_return'] - merged['RF']

    X = merged[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    y = merged['excess_return']

    model = sm.OLS(y, X).fit()

    result = {
        'portfolio': portfolio_df.columns[1], 
        'alpha': model.params['const'],
        't_alpha': model.tvalues['const'],
        'beta_mkt': model.params['Rm-rf'],
        't_mkt': model.tvalues['Rm-rf'],
        'beta_smb': model.params['SMB'],
        't_smb': model.tvalues['SMB'],
        'beta_hml': model.params['HML'],
        't_hml': model.tvalues['HML'],
        'beta_rmw': model.params['RMW'],
        't_rmw': model.tvalues['RMW'],
        'beta_cma': model.params['CMA'],
        't_cma': model.tvalues['CMA'],
        'r_squared': model.rsquared_adj,
        'std_error_regression': model.mse_resid ** 0.5,
        'n_obs': int(model.nobs)
    }

    residuals_df = pd.DataFrame({
        'date': merged['date'].values,
        'portfolio': [result['portfolio']] * len(merged),
        'residual': model.resid.values
    })

    return pd.DataFrame([result]), residuals_df


Port = pd.read_csv('FF_portfolios.csv')
Port['date'] = pd.to_datetime(Port['date'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
portfolio_columns = [str(i) for i in range(1, 26)]
Port[portfolio_columns] = Port[portfolio_columns] / 100
mean = Port[portfolio_columns].mean()

print(mean)

first_step_results = []
first_step_residuals = []

portfolio_columns = [str(i) for i in range(1, 26)]

for portfolio_name in portfolio_columns:
    
    df_portfolio = Port[['date', portfolio_name]].copy()
    df_portfolio = df_portfolio.rename(columns={portfolio_name: 'portfolio_return'})
    
    
    df_result, df_residuals = run_time_series_regressions(df_portfolio, FF)
    
    
    df_result['Portfolio'] = portfolio_name
    df_residuals['Portfolio'] = portfolio_name

    
    first_step_results.append(df_result)
    first_step_residuals.append(df_residuals)


all_results = pd.concat(first_step_results, ignore_index=True)
all_residuals = pd.concat(first_step_residuals, ignore_index=True)
alpha_vector = all_results['alpha'].values.reshape(-1, 1)
factor_returns = FF[['Rm-rf','SMB','HML', 'RMW', 'CMA']].dropna().values
reisiduals = all_residuals.pivot(index='date', columns='Portfolio', values='residual').dropna()



t, n = reisiduals.shape           
k = factor_returns.shape[1] 


mean_factors = np.mean(factor_returns, axis=0).reshape(-1, 1)
factor_cov_matrix = np.cov(factor_returns, rowvar=False)

residual_cov = reisiduals.cov().values    


inv_factor_cov = np.linalg.inv(factor_cov_matrix)
inv_residual_cov = np.linalg.inv(residual_cov)


first_term = (t / n) * ((t - n - k) / (t - k - 1))
second_term = (alpha_vector.T @ inv_residual_cov @ alpha_vector)
third_term = 1 + (mean_factors.T @ inv_factor_cov @ mean_factors)


GRS_stat = first_term * (second_term / third_term)
p_value = 1 - f.cdf(GRS_stat.item(), n, t - n - k)


print(f"GRS statistic: {GRS_stat.item():.4f}")
print(f"p-value: {p_value:.4f}")
