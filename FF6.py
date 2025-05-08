import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import f
from numpy.linalg import inv


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


portfolio_df['value_intangible_portfolio'] = portfolio_df['size_group'] + np.where(
    portfolio_df['B/M_adj'] <= portfolio_df.groupby('date')['B/M_adj'].transform(lambda x: x.quantile(0.30)), 'L(I)',
    np.where(
        portfolio_df['B/M_adj'] >= portfolio_df.groupby('date')['B/M_adj'].transform(lambda x: x.quantile(0.70)), 'H(I)', 'N(I)'
    )
)


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

def compute_SMB(df):
    # --- Value portfolios ---
    val_sl = weighted_avg(df[df['value_portfolio'] == 'SL'])
    val_sn = weighted_avg(df[df['value_portfolio'] == 'SN'])
    val_sh = weighted_avg(df[df['value_portfolio'] == 'SH'])
    val_bl = weighted_avg(df[df['value_portfolio'] == 'BL'])
    val_bn = weighted_avg(df[df['value_portfolio'] == 'BN'])
    val_bh = weighted_avg(df[df['value_portfolio'] == 'BH'])

    smb_val = ((val_sl + val_sn + val_sh) / 3) - ((val_bl + val_bn + val_bh) / 3)

    # --- Profitability portfolios ---
    prof_sw = weighted_avg(df[df['profitability_portfolio'] == 'SW'])
    prof_sn = weighted_avg(df[df['profitability_portfolio'] == 'SN'])
    prof_sr = weighted_avg(df[df['profitability_portfolio'] == 'SR'])
    prof_bw = weighted_avg(df[df['profitability_portfolio'] == 'BW'])
    prof_bn = weighted_avg(df[df['profitability_portfolio'] == 'BN'])
    prof_br = weighted_avg(df[df['profitability_portfolio'] == 'BR'])

    smb_prof = ((prof_sw + prof_sn + prof_sr) / 3) - ((prof_bw + prof_bn + prof_br) / 3)

    # --- Investment portfolios ---
    inv_sc = weighted_avg(df[df['investment_portfolio'] == 'SC'])
    inv_sn = weighted_avg(df[df['investment_portfolio'] == 'SN'])
    inv_sa = weighted_avg(df[df['investment_portfolio'] == 'SA'])
    inv_bc = weighted_avg(df[df['investment_portfolio'] == 'BC'])
    inv_bn = weighted_avg(df[df['investment_portfolio'] == 'BN'])
    inv_ba = weighted_avg(df[df['investment_portfolio'] == 'BA'])

    smb_inv = ((inv_sc + inv_sn + inv_sa) / 3) - ((inv_bc + inv_bn + inv_ba) / 3)

    # --- Intangible-Adjusted Value Portfolios ---
    valI_sl = weighted_avg(df[df['value_intangible_portfolio'] == 'SL(I)'])
    valI_sn = weighted_avg(df[df['value_intangible_portfolio'] == 'SN(I)'])
    valI_sh = weighted_avg(df[df['value_intangible_portfolio'] == 'SH(I)'])
    valI_bl = weighted_avg(df[df['value_intangible_portfolio'] == 'BL(I)'])
    valI_bn = weighted_avg(df[df['value_intangible_portfolio'] == 'BN(I)'])
    valI_bh = weighted_avg(df[df['value_intangible_portfolio'] == 'BH(I)'])

    smb_val_intangible = ((valI_sl + valI_sn + valI_sh) / 3) - ((valI_bl + valI_bn + valI_bh) / 3)

    
    smb = (smb_val + smb_prof + smb_inv + smb_val_intangible) / 4

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

def compute_HML_INT(df):
    sh_int = weighted_avg(df[df['value_intangible_portfolio'] == 'SH(I)'])
    sl_int = weighted_avg(df[df['value_intangible_portfolio'] == 'SL(I)'])
    bh_int = weighted_avg(df[df['value_intangible_portfolio'] == 'BH(I)'])
    bl_int = weighted_avg(df[df['value_intangible_portfolio'] == 'BL(I)'])

    hml_int = ((sh_int - sl_int) + (bh_int - bl_int)) / 2
    return hml_int.rename('HML_INT')

##Factor returns 
HML = compute_HML(df)
RMW = compute_RMW(df)
CMA = compute_CMA(df)
SMB = compute_SMB(df)
HML_INT = compute_HML_INT(df)

factor_returns = pd.concat([HML, RMW, CMA, SMB,HML_INT], axis=1).reset_index()
rm_rf = df[['date', 'Rm-rf']].drop_duplicates(subset='date').sort_values('date')
factor_returns = factor_returns.merge(rm_rf, on='date', how='left')
rf = df[['date', 'risk_free_rate']].drop_duplicates(subset='date').sort_values('date')
factor_returns = factor_returns.merge(rf, on='date', how='left')
for col in ['HML', 'RMW', 'CMA', 'SMB', 'Rm-rf','HML_INT']:
    factor_returns[f'{col}_cum_your_model'] = ((1+factor_returns[col]).cumprod())



factors = ['HML', 'RMW', 'CMA', 'SMB', 'Rm-rf','HML_INT']
factor_corr_matrix = factor_returns[factors].corr()

T = factor_returns.shape[0]

# Calculate statistics
factor_stats = pd.DataFrame({
    'Mean': factor_returns[factors].mean(),
    'Std Dev': factor_returns[factors].std(),
})

factor_stats['t-Statistic'] = factor_stats['Mean'] / (factor_stats['Std Dev'] / np.sqrt(T))
factor_stats = factor_stats.round(6)
factors_ordered = ['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA','HML_INT']
factor_corr_matrix = factor_returns[factors_ordered].corr().round(4)
T = factor_returns.shape[0]

from scipy import stats

diff = factor_returns['HML'] - factor_returns['HML_INT']
t_stat, p_value = stats.ttest_1samp(diff.dropna(), 0)


factor_stats = pd.DataFrame({
    'Mean': factor_returns[factors_ordered].mean(),
    'Std Dev': factor_returns[factors_ordered].std(),
})
factor_stats['t-Statistic'] = factor_stats['Mean'] / (factor_stats['Std Dev'] / np.sqrt(T))
factor_stats = factor_stats.round(4)

# Graphs
cum_cols = ['HML_cum_your_model', 'RMW_cum_your_model', 'CMA_cum_your_model', 
            'SMB_cum_your_model', 'HML_INT_cum_your_model']

plt.figure(figsize=(12, 6))
for col in cum_cols:
    plt.plot(factor_returns['date'], factor_returns[col], label=col.replace('_cum_your_model', ''))

plt.title('Cumulative Returns of All Factors')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(factor_returns['date'], factor_returns['HML_cum_your_model'], label='HML')
plt.plot(factor_returns['date'], factor_returns['HML_INT_cum_your_model'], label='HML_INT', linestyle='--')

plt.title('Cumulative Returns: HML vs. HML(INT)')
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


##Regressions 1 step
def run_time_series_regressions(portfolio_df, factor_df):
    results = []
    residuals_list = []

    
    portfolio_keys = list(portfolio_df.columns.difference(['date', 'portfolio_return']))

    for name, group in portfolio_df.groupby(portfolio_keys):
        merged = group.merge(factor_df, on='date', how='inner')
        merged['excess_return'] = merged['portfolio_return'] - merged['risk_free_rate']

        X = merged[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA','HML_INT']]
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
    'beta_hml_int': model.params['HML_INT'],
    't_hml_int': model.tvalues['HML_INT'],
    'adj_r_squared': model.rsquared_adj,
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
all_results.to_excel('all_factor_regression_resultsFF6.xlsx', index=False)

## Second step regressions
def run_fama_macbeth_full(all_results_df, portfolio_df, factor_returns):
    beta_df = all_results_df[['portfolio', 'beta_mkt', 'beta_smb', 'beta_hml', 'beta_rmw', 'beta_cma','beta_hml_int']].copy()
    beta_df.columns = ['portfolio', 'Rm-rf', 'SMB', 'HML', 'RMW', 'CMA', 'HML_INT']

    id_cols = list(portfolio_df.columns.difference(['date', 'portfolio_return']))
    portfolio_df['portfolio'] = portfolio_df[id_cols].apply(lambda row: tuple(row), axis=1)

    merged_df = portfolio_df.merge(beta_df, on='portfolio', how='inner')
    merged_df = merged_df.merge(factor_returns[['date', 'risk_free_rate']], on='date', how='left')
    merged_df['excess_return'] = merged_df['portfolio_return'] - merged_df['risk_free_rate']

    lambda_list = []
    residuals_list = []

    for date, group in merged_df.groupby('date'):
        X = group[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA','HML_INT']]  
        y = group['excess_return']
        model = sm.OLS(y, X).fit()

        
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

    factors = ['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA','HML_INT']
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


final_second_step_df.to_excel('Risk_premiums_FF56.xlsx', index=False)
print(final_second_step_df)

## Model performance
def GRS(alpha, resids, mu):
    
    T, N = resids.shape
    L = mu.shape[1]

    
    mu_mean = np.mean(mu, axis=0).reshape(-1, 1)

   
    cov_resids = (resids.T @ resids) / (T - L - 1)

   
    mu_centered = mu - mu_mean.T
    cov_fac = (mu_centered.T @ mu_centered) / T

    
    top = alpha.T @ np.linalg.inv(cov_resids) @ alpha
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

    
    mu = factor_returns[['Rm-rf', 'SMB', 'HML', 'RMW', 'CMA', 'HML_INT']].dropna().values

    
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

# average absolute ri for each portfolio set
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


final_df.to_excel('FF6_performance.xlsx', index=False)
