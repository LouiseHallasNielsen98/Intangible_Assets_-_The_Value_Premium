import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_1010 = pd.read_excel('1010.xlsx')
df_2010 = pd.read_excel('2010.xlsx')
df_2550 = pd.read_excel('2550.xlsx')
df_3520 = pd.read_excel('3520.xlsx')
df_4020 = pd.read_excel('4020.xlsx')
df_4510 = pd.read_excel('4510.xlsx')


industries = [
    ("Energy (1010)", df_1010, 'black'),
    ("Capital Goods (2010)", df_2010, 'blue'),
    ("Consumer Disc. Dist. & Retail (2550)", df_2550, 'gold'),
    ("Pharma, Biotech & Life Sci (3520)", df_3520, 'red'),
    ("Financial Services (4020)", df_4020, 'grey'),
    ("Software & Services (4510)", df_4510, 'green')
]


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, df, color) in enumerate(industries):
    df['date'] = pd.to_datetime(df['date'])
    long_short_returns = df['HML_INT'] - df['HML']
    cumulative_returns = (1 + long_short_returns).cumprod()

    ax = axes[i]
    ax.plot(df['date'], cumulative_returns, color=color)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('Cumulative Return', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True)

    T = len(df)

    # HML
    hml_mean = df['HML'].mean()
    hml_std = df['HML'].std()
    hml_sharpe = hml_mean / hml_std
    hml_tstat = hml_mean / (hml_std / np.sqrt(T))

    # HML_INT
    hml_int_mean = df['HML_INT'].mean()
    hml_int_std = df['HML_INT'].std()
    hml_int_sharpe = hml_int_mean / hml_int_std
    hml_int_tstat = hml_int_mean / (hml_int_std / np.sqrt(T))

    # Long-short
    long_short_mean = long_short_returns.mean()
    long_short_std = long_short_returns.std()
    long_short_sharpe = long_short_mean / long_short_std
    long_short_tstat = long_short_mean / (long_short_std / np.sqrt(T))

    print(f"=== {name} ===")
    print(f"HML:        Mean = {hml_mean:.6f}, Std = {hml_std:.6f}, Sharpe = {hml_sharpe:.4f}, T-stat = {hml_tstat:.4f}")
    print(f"HML_INT:    Mean = {hml_int_mean:.6f}, Std = {hml_int_std:.6f}, Sharpe = {hml_int_sharpe:.4f}, T-stat = {hml_int_tstat:.4f}")
    print(f"Long/Short: Mean = {long_short_mean:.6f}, Std = {long_short_std:.6f}, Sharpe = {long_short_sharpe:.4f}, T-stat = {long_short_tstat:.4f}")
    print()


plt.suptitle("Cumulative Returns by Industry: Long HML(INT) / Short HML", fontsize=14)
plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.90, bottom=0.07, left=0.06, right=0.98)
plt.show()

