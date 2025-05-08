import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('Final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['sector_2'] == 1510].copy()
pd.set_option('display.max_columns', None)


### Intangibles

##Depreciation rate  
δ_RD = 0.15  
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
firm_yearly_avg = df.groupby(['sec_id', 'year'])['INT'].mean().reset_index()
df['INT_to_MV'] = df['INT'] / df['market_cap']
firm_yearly_avg = df.groupby(['sec_id', 'year'])['INT_to_MV'].mean().reset_index()
avg_intangibles_df = firm_yearly_avg.groupby('year')['INT_to_MV'].mean().reset_index(name='avg_INT_to_MV_per_firm')
avg_intangibles_df.to_excel('avg_intangibles_to_mv_per_year_1510.xlsx', index=False)

