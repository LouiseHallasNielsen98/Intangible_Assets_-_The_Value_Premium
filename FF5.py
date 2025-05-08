import pandas as pd
import sqlite3
from datetime import datetime
import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from scipy.stats import f
from numpy.linalg import pinv


df = pd.read_csv('Final_dataset_INT.csv')
df['date'] = pd.to_datetime(df['date'])

##ADDING TO BOOK VALUE
df['book_value_adjusted'] = df['book_value'] + df['INT'] - df['goodwill_value']

### Factor values 
df['Rm-rf'] = df['Market_returns'] - df['risk_free_rate']
df['B/M'] = df['book_value'] / df['market_cap']
df['B/M_adj'] = df['book_value_adjusted'] / df['market_cap']
df['Prof'] = df['ROE']
df['Inv'] = df['capex_over_assets']
df['SMB'] = df['market_cap']

# Rebalancing months and flag
rebalance_months = [2,5,8,11]
df['rebalance_flag'] = df['date'].dt.month.isin(rebalance_months)
portfolio_df = df[df['rebalance_flag']].copy()
portfolio_df['rebalance_date'] = portfolio_df['date']

##Sorting portfolios quaretly 
portfolio_df['size_group'] = np.where(
    portfolio_df['SMB'] <= portfolio_df.groupby('date')['SMB'].transform('median'),
    'S', 'B'  
)

portfolio_df['value_portfolio'] = portfolio_df['size_group'] + np.where(
    portfolio_df['B/M'] <= portfolio_df.groupby('date')['B/M'].transform(lambda x: x.quantile(0.30)), 'L',
    np.where(
        portfolio_df['B/M'] >= portfolio_df.groupby('date')['B/M'].transform(lambda x: x.quantile(0.70)), 'H', 'N'
    )
)

portfolio_df['profitability_portfolio'] = portfolio_df['size_group'] + np.where(
    portfolio_df['Prof'] <= portfolio_df.groupby('date')['Prof'].transform(lambda x: x.quantile(0.30)), 'W',
    np.where(
        portfolio_df['Prof'] >= portfolio_df.groupby('date')['Prof'].transform(lambda x: x.quantile(0.70)), 'R', 'N'
    )
)

portfolio_df['investment_portfolio'] = portfolio_df['size_group'] + np.where(
    portfolio_df['Inv'] <= portfolio_df.groupby('date')['Inv'].transform(lambda x: x.quantile(0.30)), 'C',
    np.where(
        portfolio_df['Inv'] >= portfolio_df.groupby('date')['Inv'].transform(lambda x: x.quantile(0.70)), 'A', 'N'
    )
)

# Intangible-adjusted value portfolios
portfolio_df['value_intangible_portfolio'] = portfolio_df['size_group'] + np.where(
    portfolio_df['B/M_adj'] <= portfolio_df.groupby('date')['B/M_adj'].transform(lambda x: x.quantile(0.30)), 'L(I)',
    np.where(
        portfolio_df['B/M_adj'] >= portfolio_df.groupby('date')['B/M_adj'].transform(lambda x: x.quantile(0.70)), 'H(I)', 'N(I)'
    )
)

# Merge
portfolio_df = portfolio_df.sort_values(['sec_id', 'rebalance_date'])
df = df.sort_values(['sec_id', 'date'])


df = pd.merge(
    df,
    portfolio_df[['sec_id', 'rebalance_date', 'value_portfolio', 'size_group',
                  'profitability_portfolio', 'investment_portfolio','value_intangible_portfolio']],
    left_on=['sec_id', 'date'],
    right_on=['sec_id', 'rebalance_date'],
    how='left'
)

df = df.sort_values(['sec_id', 'date'])
cols_to_fill = ['value_portfolio', 'size_group', 'profitability_portfolio', 'investment_portfolio','value_intangible_portfolio']
for col in cols_to_fill:
    df[col] = df.groupby('sec_id')[col].ffill(limit=2)
    df[col] = df.groupby('sec_id')[col].bfill(limit=1)

def weighted_avg(df):
    df = df.sort_values(['sec_id', 'date']).copy()
    df['market_cap_lag'] = df.groupby('sec_id')['market_cap'].shift(1)
    df = df.dropna(subset=['market_cap_lag', 'returns'])
    df['weight'] = df.groupby('date')['market_cap_lag'].transform(lambda x: x / x.sum())
    df['weighted_return'] = df['returns'] * df['weight']
    
    return df.groupby('date')['weighted_return'].sum().rename('portfolio_return')


##Compute factors
# --- SMB ---
def compute_SMB(df):
    # Value portfolios
    val_sl = weighted_avg(df[df['value_portfolio'] == 'SL'])
    val_sn = weighted_avg(df[df['value_portfolio'] == 'SN'])
    val_sh = weighted_avg(df[df['value_portfolio'] == 'SH'])
    val_bl = weighted_avg(df[df['value_portfolio'] == 'BL'])
    val_bn = weighted_avg(df[df['value_portfolio'] == 'BN'])
    val_bh = weighted_avg(df[df['value_portfolio'] == 'BH'])

    smb_val = ((val_sl + val_sn + val_sh) / 3) - ((val_bl + val_bn + val_bh) / 3)

    # Profitability portfolios
    prof_sw = weighted_avg(df[df['profitability_portfolio'] == 'SW'])
    prof_sn = weighted_avg(df[df['profitability_portfolio'] == 'SN'])
    prof_sr = weighted_avg(df[df['profitability_portfolio'] == 'SR'])
    prof_bw = weighted_avg(df[df['profitability_portfolio'] == 'BW'])
    prof_bn = weighted_avg(df[df['profitability_portfolio'] == 'BN'])
    prof_br = weighted_avg(df[df['profitability_portfolio'] == 'BR'])

    smb_prof = ((prof_sw + prof_sn + prof_sr) / 3) - ((prof_bw + prof_bn + prof_br) / 3)

    # Investment portfolios
    inv_sc = weighted_avg(df[df['investment_portfolio'] == 'SC'])
    inv_sn = weighted_avg(df[df['investment_portfolio'] == 'SN'])
    inv_sa = weighted_avg(df[df['investment_portfolio'] == 'SA'])
    inv_bc = weighted_avg(df[df['investment_portfolio'] == 'BC'])
    inv_bn = weighted_avg(df[df['investment_portfolio'] == 'BN'])
    inv_ba = weighted_avg(df[df['investment_portfolio'] == 'BA'])

    smb_inv = ((inv_sc + inv_sn + inv_sa) / 3) - ((inv_bc + inv_bn + inv_ba) / 3)

    # Combine all three
    smb = (smb_val + smb_prof + smb_inv) / 3
    return smb.rename('SMB')

def compute_HML(df):
    sh = weighted_avg(df[df['value_portfolio'] == 'SH'])
    sl = weighted_avg(df[df['value_portfolio'] == 'SL'])
    bh = weighted_avg(df[df['value_portfolio'] == 'BH'])
    bl = weighted_avg(df[df['value_portfolio'] == 'BL'])

    hml = ((sh - sl) + (bh - bl)) / 2
    return hml.rename('HML')

def compute_RMW(df):
    sr = weighted_avg(df[df['profitability_portfolio'] == 'SR'])
    sw = weighted_avg(df[df['profitability_portfolio'] == 'SW'])
    br = weighted_avg(df[df['profitability_portfolio'] == 'BR'])
    bw = weighted_avg(df[df['profitability_portfolio'] == 'BW'])

    rmw = ((sr - sw) + (br - bw)) / 2
    return rmw.rename('RMW')

def compute_CMA(df):
    sc = weighted_avg(df[df['investment_portfolio'] == 'SC'])
    sa = weighted_avg(df[df['investment_portfolio'] == 'SA'])
    bc = weighted_avg(df[df['investment_portfolio'] == 'BC'])
    ba = weighted_avg(df[df['investment_portfolio'] == 'BA'])

    cma = ((sc - sa) + (bc - ba)) / 2
    return cma.rename('CMA')


##Factor returns 
HML = compute_HML(df)
RMW = compute_RMW(df)
CMA = compute_CMA(df)
SMB = compute_SMB(df)
factor_returns = pd.concat([HML, RMW, CMA, SMB], axis=1).reset_index()
rm_rf = df[['date', 'Rm-rf']].drop_duplicates(subset='date').sort_values('date')
factor_returns = factor_returns.merge(rm_rf, on='date', how='left')
rf = df[['date', 'risk_free_rate']].drop_duplicates(subset='date').sort_values('date')
factor_returns = factor_returns.merge(rf, on='date', how='left')
for col in ['HML', 'RMW', 'CMA', 'SMB', 'Rm-rf']:
    factor_returns[f'{col}_cum_your_model'] = ((1+factor_returns[col]).cumprod())


# FF factor returns 
FF = pd.read_csv('FF5_monthly.csv')
FF['date'] = pd.to_datetime(FF['date'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
FF = FF[(FF['date'] >= '1999-01') & (FF['date'] <= '2024-12')]
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


# List of factor names
factors = ['HML', 'RMW', 'CMA', 'SMB', 'Rm-rf']
factors = ['HML', 'RMW', 'CMA', 'SMB', 'Rm-rf']
factor_corr_matrix = factor_returns[factors].corr()
T = factor_returns.shape[0]

# Calculate statistics
factor_stats = pd.DataFrame({
    'Mean': factor_returns[factors].mean(),
    'Std Dev': factor_returns[factors].std(),
})


## Factor statistics
factor_stats['t-Statistic'] = factor_stats['Mean'] / (factor_stats['Std Dev'] / np.sqrt(T))
factor_stats = factor_stats.round(4)
factors_ordered = ['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']
factor_corr_matrix = factor_returns[factors_ordered].corr().round(4)
T = factor_returns.shape[0]
factor_stats = pd.DataFrame({
    'Mean': factor_returns[factors_ordered].mean(),
    'Std Dev': factor_returns[factors_ordered].std(),
})
factor_stats['t-Statistic'] = factor_stats['Mean'] / (factor_stats['Std Dev'] / np.sqrt(T))
factor_stats = factor_stats.round(4)



## Regress my factors on FF
comparison_df = factor_returns.merge(
    FF[['date', 'HML', 'RMW', 'CMA', 'SMB', 'Rm-rf']],
    on='date',
    suffixes=('', '_FF')
)


results = []

for factor in factors:
    x = comparison_df[f'{factor}_FF']
    y = comparison_df[factor]
    
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    
    model_mean = y.mean()
    model_vol = y.std()
    ff_mean = x.mean()
    ff_vol = x.std()
    
    results.append({
        'Factor': factor,
        'Slope': slope,
        'Intercept': intercept,
        'R^2': r_value ** 2,
        'Model Mean Return': model_mean,
        'Model Volatility': model_vol,
        'FF Mean Return': ff_mean,
        'FF Volatility': ff_vol
    })
    

results_df = pd.DataFrame(results)

# Grapfs
cum_cols = ['HML_cum_your_model', 'RMW_cum_your_model', 'CMA_cum_your_model', 
            'SMB_cum_your_model']

plt.figure(figsize=(12, 6))
for col in cum_cols:
    plt.plot(factor_returns['date'], factor_returns[col], label=col.replace('_cum_your_model', ''))

plt.title('Cumulative Returns of the Factors')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

##Construct portfolios for FM regressions 
def construct_portfolios(df, group_cols):
    df = df[df[group_cols].notnull().all(axis=1)].copy()

    results = []

    for keys, group in df.groupby(group_cols):
        weighted_returns = weighted_avg(group).reset_index()
        weighted_returns[group_cols] = keys if isinstance(keys, tuple) else (keys,)
        results.append(weighted_returns)

    portfolio_df = pd.concat(results, ignore_index=True)
    return portfolio_df.rename(columns={'vw_return': 'portfolio_return'})

# 1. Size × Value 
size_value_2x3 = construct_portfolios(df, ['value_portfolio'])

# 2. Size × Profitability 
size_prof_2x3 = construct_portfolios(df, ['profitability_portfolio'])

# 3. Size × Investment 
size_inv_2x3 = construct_portfolios(df, ['investment_portfolio'])

# 4. Size × Value intangible
size_valueint_2x3 = construct_portfolios(df, ['value_intangible_portfolio'])

# 5. Size × Value x Profitability
size_value_prof_2x3x3 = construct_portfolios(df, ['value_portfolio', 'profitability_portfolio'])

# 6. Size × Value × Investment
size_value_inv_2x3x3 = construct_portfolios(df, ['value_portfolio', 'investment_portfolio'])

# 7. Size x Profitability x Investment 
size_prof_inv_2x3x3 = construct_portfolios(df, ['profitability_portfolio', 'investment_portfolio'])

# 8. Size x Value x Intangible Value 
size_value_valueint_2x3x3 = construct_portfolios(df, ['value_portfolio', 'value_intangible_portfolio'])

# 9. Size x Intangible value x Profitability 
size_valueint_prof_2x3x3 = construct_portfolios(df, ['value_intangible_portfolio','profitability_portfolio'])

# 10 Size x Intangible value x Investment 
size_valueint_Inv_2x3x3 = construct_portfolios(df, ['value_intangible_portfolio','investment_portfolio'])


# Count stocks in portfolios 
portfolio_sets = {
    'size_value_2x3': ['value_portfolio'],
    'size_prof_2x3': ['profitability_portfolio'],
    'size_inv_2x3': ['investment_portfolio'],
    'size_valueint_2x3': ['value_intangible_portfolio'],
    'size_value_prof_2x3x3': ['value_portfolio', 'profitability_portfolio'],
    'size_value_inv_2x3x3': ['value_portfolio', 'investment_portfolio'],
    'size_prof_inv_2x3x3': ['profitability_portfolio', 'investment_portfolio'],
    'size_value_valueint_2x3x3': ['value_portfolio', 'value_intangible_portfolio'],
    'size_valueint_prof_2x3x3': ['value_intangible_portfolio', 'profitability_portfolio'],
    'size_valueint_Inv_2x3x3': ['value_intangible_portfolio', 'investment_portfolio'],
}

##Count stocks 
df_filtered = df[~((df['date'].dt.year == 1999) & (df['date'].dt.month == 1))].copy()

monthly_counts = []

for name, cols in portfolio_sets.items():
    subset = df_filtered[df_filtered[cols].notnull().all(axis=1)].copy()

    
    monthly = (
        subset.groupby(['date'] + cols)['sec_id']
        .nunique()
        .groupby(cols)
        .mean()
        .reset_index()
        .rename(columns={'sec_id': 'avg_n_stocks'})
    )

    monthly['portfolio_set'] = name
    monthly_counts.append(monthly)

monthly_avg_stocks_df = pd.concat(monthly_counts, ignore_index=True)
monthly_avg_stocks_df = monthly_avg_stocks_df.sort_values(['portfolio_set'] + monthly_avg_stocks_df.columns[:-2].tolist())


##Regressions 1 step
def run_time_series_regressions(portfolio_df, factor_df):
    results = []
    residuals_list = []

    
    portfolio_keys = list(portfolio_df.columns.difference(['date', 'portfolio_return']))

    for name, group in portfolio_df.groupby(portfolio_keys):
        merged = group.merge(factor_df, on='date', how='inner')
        merged['excess_return'] = merged['portfolio_return'] - merged['risk_free_rate']

        X = merged[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X)
        y = merged['excess_return']

        model = sm.OLS(y, X).fit()

        
        results.append({
    'portfolio': name,
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
})
        
        residuals_df = pd.DataFrame({
            'date': merged['date'].values,
            'portfolio': [name] * len(merged),
            'residual': model.resid.values
        })
        residuals_list.append(residuals_df)

    
    all_residuals_df = pd.concat(residuals_list, ignore_index=True)

    return pd.DataFrame(results), all_residuals_df

# Running regression for each portfolio
portfolio_sets = {
    'Size-Value 2x3': size_value_2x3,
    'Size-Prof 2x3': size_prof_2x3,
    'Size-Inv 2x3': size_inv_2x3,
    'Size-Value-Prof 2x3x3': size_value_prof_2x3x3,
    'Size-Value-Inv 2x3x3': size_value_inv_2x3x3,
    'Size-Prof-Inv 2x3x3': size_prof_inv_2x3x3,
    'Size-Value-Intangible 2x3x3': size_value_valueint_2x3x3,
    'Size-Valueint-Prof 2x3x3': size_valueint_prof_2x3x3,
    'Size-Valueint-Inv 2x3x3': size_valueint_Inv_2x3x3,
    'Size-Valueint 2x3': size_valueint_2x3
    }

# Run first-step regressions and store results
first_step_results = []
first_step_residuals = []

for name, df_portfolio in portfolio_sets.items():
    df_result, df_residuals = run_time_series_regressions(df_portfolio, factor_returns)
    df_result['Portfolio_Set'] = name
    df_residuals['Portfolio_Set'] = name
    first_step_results.append(df_result)
    first_step_residuals.append(df_residuals)

all_results = pd.concat(first_step_results, ignore_index=True)
all_residuals = pd.concat(first_step_residuals, ignore_index=True)
all_results.to_excel('all_factor_regression_resultsFF5.xlsx', index=False)

## Second step regressions
def run_fama_macbeth_full(all_results_df, portfolio_df, factor_returns):
    beta_df = all_results_df[['portfolio', 'beta_mkt', 'beta_smb', 'beta_hml', 'beta_rmw', 'beta_cma']].copy()
    beta_df.columns = ['portfolio', 'Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']

    id_cols = list(portfolio_df.columns.difference(['date', 'portfolio_return']))
    portfolio_df['portfolio'] = portfolio_df[id_cols].apply(lambda row: tuple(row), axis=1)

    merged_df = portfolio_df.merge(beta_df, on='portfolio', how='inner')
    merged_df = merged_df.merge(factor_returns[['date', 'risk_free_rate']], on='date', how='left')
    merged_df['excess_return'] = merged_df['portfolio_return'] - merged_df['risk_free_rate']

    lambda_list = []
    residuals_list = []

    for date, group in merged_df.groupby('date'):
        X = group[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']]
        y = group['excess_return']
        model = sm.OLS(y, X).fit()

        # Store lambdas
        coefs = model.params.to_dict()
        coefs['date'] = date
        lambda_list.append(coefs)

        
        residuals = pd.DataFrame({
            'date': date,
            'portfolio': group['portfolio'].values,
            'residual': model.resid.values
        })
        residuals_list.append(residuals)

    lambda_df = pd.DataFrame(lambda_list).sort_values('date').reset_index(drop=True)
    residuals_df = pd.concat(residuals_list, ignore_index=True)

    factors = ['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']
    avg_lambda = lambda_df[factors].mean()

    
    nw_se = {}
    for factor in factors:
        y = lambda_df[factor]
        x = np.ones(len(y))  
        model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        nw_se[factor] = model.bse[0]  

    se_lambda = pd.Series(nw_se)
    t_stats = avg_lambda / se_lambda

    results_summary = pd.DataFrame({
        'Average Risk Premium': avg_lambda,
        'Standard Error (Newey-West)': se_lambda,
        't-Statistic (NW)': t_stats
    })

    results_summary.index.name = 'Factor'
    results_summary = results_summary.reset_index()

    return results_summary, lambda_df, residuals_df

second_step_results = []
lambda_avg = []

for name, portfolio_df in portfolio_sets.items():
    all_results_df = all_results[all_results['Portfolio_Set'] == name]

    
    summary_df, lambda_df_i, residuals_df = run_fama_macbeth_full(all_results_df, portfolio_df, factor_returns)

    
    summary_df['Portfolio_Set'] = name
    lambda_df_i['Portfolio_Set'] = name

    
    second_step_results.append(summary_df)
    lambda_avg.append(lambda_df_i)


final_second_step_df = pd.concat(second_step_results, ignore_index=True)
lambda_df = pd.concat(lambda_avg, ignore_index=True)


final_second_step_df.to_excel('Risk_premiums_FF5.xlsx', index=False)

## Model performance
def GRS(alpha, resids, mu):
    
    T, N = resids.shape
    L = mu.shape[1]

    
    mu_mean = np.mean(mu, axis=0).reshape(-1, 1)

    
    cov_resids = (resids.T @ resids) / (T - L - 1)

    
    mu_centered = mu - mu_mean.T
    cov_fac = (mu_centered.T @ mu_centered) / T

    
    top = alpha.T @ pinv(cov_resids) @ alpha
    bottom = 1 + mu_mean.T @ np.linalg.inv(cov_fac) @ mu_mean

    GRS_stat = (T / N) * ((T - N - L) / (T - L - 1)) * (top / bottom)
    pVal = 1 - f.cdf(GRS_stat, N, T - N - L)

    return GRS_stat.item(), pVal.item()


grs_results = []

for set_name, df in portfolio_sets.items():
    
    resids = all_residuals[all_residuals['Portfolio_Set'] == set_name].pivot(
        index='date', columns='portfolio', values='residual'
    ).dropna()

   
    alphas = all_results[all_results['Portfolio_Set'] == set_name]['alpha'].values.reshape(-1, 1)

    
    mu = factor_returns[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA']].dropna().values

    
    grs_stat, p_value = GRS(alphas, resids, mu)

    
    grs_results.append({
        'Portfolio_Set': set_name,
        'GRS_Statistic': grs_stat,
        'p_value': p_value
    })


grs_results_df = pd.DataFrame(grs_results)

# Absolute alphas
all_results['abs_alpha'] = all_results['alpha'].abs()
average_abs_alpha = all_results.groupby('Portfolio_Set')['abs_alpha'].mean().reset_index(name='avg_abs_alpha')

# Average absolute ri for each portfolio set
results = []

for set_name, df in portfolio_sets.items():
    df = df.merge(factor_returns[['date', 'risk_free_rate']], on='date', how='left')
    df['excess_return'] = df['portfolio_return'] - df['risk_free_rate']
    avg_returns_per_portfolio = df.groupby('portfolio')['excess_return'].mean().reset_index(name='Ri')
    avg_returns_per_portfolio['ri'] = avg_returns_per_portfolio['Ri'] - avg_returns_per_portfolio['Ri'].mean()
    avg_returns_per_portfolio['abs_ri'] = avg_returns_per_portfolio['ri'].abs()
    avg_abs_ri = avg_returns_per_portfolio['abs_ri'].mean()

    results.append({
        'Portfolio_Set': set_name,
        'avg_abs_ri': avg_abs_ri
    })


average_abs_ri_df = pd.DataFrame(results)


final_df = pd.merge(average_abs_alpha, average_abs_ri_df, on='Portfolio_Set')
final_df['alpha_ri_ratio'] = final_df['avg_abs_alpha'] / final_df['avg_abs_ri']


final_df = pd.merge(final_df, grs_results_df, on='Portfolio_Set')

final_df.to_excel('FF5_performance.xlsx', index=False)
