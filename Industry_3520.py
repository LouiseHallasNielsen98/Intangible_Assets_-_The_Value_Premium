import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('Final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['sector_2'] == 3520].copy()
pd.set_option('display.max_columns', None)


### Intangibles

##Depreciation rate  
δ_RD = 0.10  
δ_SGA = 0.20  

##Growth rate
def average_growth(df, variable, id_col='sec_id', time_col='date'):
    df_sorted = df.sort_values(by=[id_col, time_col])
    df_sorted['growth'] = df_sorted.groupby(id_col)[variable].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    df_clean = df_sorted[np.isfinite(df_sorted['growth'])]
    return df_clean['growth'].mean()


g_RD = average_growth(df,variable='rd_value')
g_SGA = average_growth(df,variable='sga_value')

df[['rd_value', 'sga_value', 'goodwill_value']] = df[['rd_value', 'sga_value', 'goodwill_value']].fillna(0)

## Make Intagibles and add to book-value 
def compute_initial_intangible(group):
    first_sga = group['sga_value'].iloc[0]
    first_rd = group['rd_value'].iloc[0]
    int_i0 = (0.50 * first_sga / (g_SGA + δ_SGA)) + (0.70 * first_rd / (g_RD + δ_RD))
    return pd.Series({'INT_i0': int_i0})

int_initial = df.groupby('sec_id', group_keys=False)[['sga_value', 'rd_value']].apply(compute_initial_intangible)
df = df.merge(int_initial, on='sec_id', how='left')
df['INT'] = 0.0  

for sec_id in df['sec_id'].unique():
    sec_data = df[df['sec_id'] == sec_id].copy()

    # Initialize the first value of INT
    sec_data.loc[sec_data.index[0], 'INT'] = float(sec_data['INT_i0'].iloc[0])

    for t in range(1, len(sec_data)):
        prev_row = sec_data.iloc[t-1]
        curr_row = sec_data.iloc[t]

        # Ensure that a new int is only calculated if the value changes to account for the ffill
        inputs = ['rd_value', 'sga_value']
        has_changed = any(
            not np.isclose(curr_row[var], prev_row[var], atol=1e-8) 
            for var in inputs
        )

        if has_changed:
            prev_INT = prev_row['INT']
            sga_t = curr_row['sga_value']
            rd_t = curr_row['rd_value']
            sga_t_1 = prev_row['sga_value']
            rd_t_1 = prev_row['rd_value']

            new_INT = new_INT = (prev_INT - ( δ_RD * rd_t_1) - (δ_SGA * sga_t_1) + (0.50 * sga_t) + (0.70 * rd_t))
            sec_data.loc[sec_data.index[t], 'INT'] = new_INT
        else:
            
            sec_data.loc[sec_data.index[t], 'INT'] = prev_row['INT']

    
    df.loc[sec_data.index, 'INT'] = sec_data['INT'].values

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['INT'] = df['INT'].fillna(0)
df['INT_to_MV'] = df['INT'] / df['market_cap']
firm_yearly_avg = df.groupby(['sec_id', 'year'])['INT_to_MV'].mean().reset_index()
avg_intangibles_df = firm_yearly_avg.groupby('year')['INT_to_MV'].mean().reset_index(name='avg_INT_to_MV_per_firm')
avg_intangibles_df.to_excel('avg_intangibles_to_mv_per_year_3520.xlsx', index=False)

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

    # --- Combine all four SMB components ---
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

factor_returns.to_excel('3520.xlsx', index=False)

