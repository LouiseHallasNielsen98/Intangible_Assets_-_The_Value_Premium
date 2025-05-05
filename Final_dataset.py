import pandas as pd
import sqlite3
from datetime import datetime
import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

##Get data from database, set datetime and drop variables
def get_stock_data():
    conn = sqlite3.connect(f"database.db")
    
    df = pd.read_sql_query(f"SELECT * from data", conn)
    
    conn.close()
    return df

##Take the last trading day price and set end month
df = get_stock_data()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sec_id', 'date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.groupby(['sec_id', 'year', 'month']).tail(1)
df = df.drop(columns=['year', 'month'])
df = df.reset_index(drop=True)
df['date'] = df['date'].dt.to_period('M').dt.to_timestamp('M')


##Convert prices to returns 
df = df.sort_values(['date', 'sec_id'])
df['returns'] = df.groupby('sec_id')['tot_ret_usd'].pct_change()
df['market_return'] = df['market_return'].str.replace(',', '.').astype(float)
df['market_return'] = pd.to_numeric(df['market_return'], errors='coerce')
df['Market_returns'] = df.groupby('sec_id')['market_return'].pct_change().fillna(0)
df = df.drop(columns=['tot_ret_usd','market_return'])

#Count stocks 
df['year'] = df['date'].dt.year
secid_counts_year_raw = df.groupby('year')['sec_id'].nunique()

# Risk-free rate 
risk_free_rate = yf.Ticker("^IRX")
rf_data = risk_free_rate.history(start="1999-01-01")[['Close']]
rf_data.index = pd.to_datetime(rf_data.index)
rf_data = rf_data.tz_localize(None)
rf_data['rf_monthly'] = (1 + rf_data['Close'] / 100) ** (1/12) - 1
rf_monthly = rf_data.resample('ME').last().reset_index()
rf_monthly.rename(columns={'Date': 'date', 'rf_monthly': 'risk_free_rate'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.merge(rf_monthly[['date', 'risk_free_rate']], on='date', how='left')

## Convert values 
df['goodwill_value'] = df['goodwill_value'] * 100000
df['sga_value'] = df['sga_value'] * 1000000
df['rd_value'] = df['rd_value'] * 1000000

## Drop N/A values 
df = df.dropna(subset=['market_cap','capex_over_assets','ROE','book_value'])


##Discount factor 
df = df.reset_index()
df['year'] = df['date'].dt.year
discount_df = df[['date', 'risk_free_rate']].drop_duplicates().copy()
discount_df = discount_df.sort_values(by='date', ascending=True)
discount_df['discount_factor'] = 1 / (1 + discount_df['risk_free_rate'])
discount_df['cumulative_discount'] = discount_df['discount_factor'][::-1].cumprod()[::-1]
discount_df['discounted_500M'] = 500_000_000 * discount_df['cumulative_discount']
df = df.merge(discount_df[['date', 'discounted_500M']], on='date', how='left')
stocks_below_threshold = df[df['market_cap'] < df['discounted_500M']][['sec_id', 'year']].drop_duplicates()
df = df.merge(stocks_below_threshold, on=['sec_id', 'year'], how='left', indicator=True)
df = df[df['_merge'] == 'left_only'].drop(columns=['_merge'])


## Keeps stocks with valid returns
return_counts = (df.groupby(['sec_id', 'year'])['returns'].apply(lambda x: x.notna().sum()).reset_index(name='return_count'))
valid_pairs = return_counts[((return_counts['return_count'] == 12) & (return_counts['year'] != 1999)) |
                             ((return_counts['year'] == 1999) & (return_counts['return_count'] >= 11))][['sec_id', 'year']]

df = df.merge(valid_pairs, on=['sec_id', 'year'], how='inner')

##Final dataset count sec_ids 
secid_counts_year_final = df.groupby('year')['sec_id'].nunique()
counts_df = pd.DataFrame({
    'Raw': secid_counts_year_raw,
    'Final': secid_counts_year_final
})

df = df.drop(columns=['year'])
ax = counts_df.plot(kind='bar', figsize=(12, 6))
ax.set_title('Unique Stocks per Year: Raw vs Final Dataset')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Unique Stocks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##Number of stocks in each sector 
secid_counts_sector_final = df.groupby('sector_2')['sec_id'].nunique().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
secid_counts_sector_final.plot(kind='bar', color='skyblue')

plt.title('Total Number of Unique Stocks per Sector (1999–2024)')
plt.xlabel('Sector')
plt.ylabel('Number of Unique Stocks')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#SP500 - Real
SP500_real = pd.read_excel("Sp500_total_ret.xlsx")
SP500_real['date'] = pd.to_datetime(SP500_real['date'])
SP500_real = SP500_real[(SP500_real['date'] >= '1999-03-01') & (SP500_real['date'] <= '2024-12-31')].reset_index(drop=True)
SP500_real['month'] = SP500_real['date'].dt.to_period('M').dt.to_timestamp('M') 
SP500_real_monthly = SP500_real.sort_values('date').groupby('month').last().reset_index()
SP500_real_monthly['sp500_real_return'] = SP500_real_monthly['price'].pct_change()
SP500_real_monthly = SP500_real_monthly.dropna(subset=['sp500_real_return']).reset_index(drop=True)
SP500_real_monthly['sp500_real_cumulative_return'] = ((1+SP500_real_monthly['sp500_real_return']).cumprod())

##Value weighted function
def weighted_avg(df):
    df = df.sort_values(['date', 'sec_id']).copy()
    df['market_cap_lag'] = df.groupby('sec_id')['market_cap'].shift(1)
    df['weight'] = df.groupby('date')['market_cap_lag'].transform(lambda x: x / x.sum())
    df['weighted_return'] = df['returns'] * df['weight']

    return df.groupby('date')['weighted_return'].sum().rename('portfolio_return')

##Rebalance
rebalance_months = [3,6,9,12]
df['rebalance_flag'] = df['date'].dt.month.isin(rebalance_months)
portfolio_df = df[df['rebalance_flag']].copy()


portfolio_df['rebalance_date'] = portfolio_df['date']

portfolio_df = (
    portfolio_df
    .sort_values(['date', 'market_cap'], ascending=[True, False])
    .groupby('date')
    .head(500)
    .copy()
)


portfolio_df['Flag'] = True

df = df.merge(
    portfolio_df[['date', 'sec_id', 'Flag']],
    on=['date', 'sec_id'],
    how='left'
)

df['Flag'] = df.groupby('sec_id')['Flag'].ffill(limit=2)
df_flagged = df[df['Flag'] == True]


##Create plot 
portfolio_returns = weighted_avg(df_flagged)
cumulative_returns = (1 + portfolio_returns).cumprod()

comparison_df = pd.merge(
    cumulative_returns.rename('portfolio_cumulative_return').reset_index(),
    SP500_real_monthly[['month', 'sp500_real_cumulative_return']].rename(columns={'month': 'date'}),
    on='date',
    how='inner'
)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['date'], comparison_df['portfolio_cumulative_return'], label='Top 500 Portfolio', linewidth=2)
plt.plot(comparison_df['date'], comparison_df['sp500_real_cumulative_return'], label='S&P 500 (Actual)', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns from 1999-2024: Custom Top 500 vs S&P 500')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##Eksport final dataset
df.to_csv('Final_dataset.csv', index=False)

