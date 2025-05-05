import pandas as pd
import sqlite3
from datetime import datetime

def init_database():
    conn = sqlite3.connect(f"database.db")
    
    df = pd.read_csv("sec_master.csv")
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    conn.cursor().execute(f"CREATE UNIQUE INDEX idx_stock_sec_id ON stocks (sec_id)")
    
    df = pd.read_csv("gics_data.csv", parse_dates=['FromDate','ToDate'], date_format="mixed", dayfirst=True)
    df.to_sql("gics", conn, if_exists="replace", index=False, dtype={"FromDate": "TEXT", "ToDate":"TEXT"})

    df = pd.read_csv("Index (qblue) - Market return.csv", sep=';', parse_dates=['date'], date_format="mixed", dayfirst=True)
    df.to_sql("qblue", conn, if_exists="replace", index=False, dtype={"date": "TEXT"})

    for n in reversed(range(1999,2025)):
        print(f"{datetime.now()} - inserting Price_data/prices_north_america_{n}.csv")
        df = pd.read_csv(f"Price_data/prices_north_america_{n}.csv", parse_dates=['date'], date_format="mixed", dayfirst=True)
        df.to_sql("Prices", conn, if_exists="append", index=False, dtype={"date": "TEXT"})

        print(f"{datetime.now()} - inserting Factor_values/factor_values_north_america_{n}.csv")
        df = pd.read_csv(f"Factor_values/factor_values_north_america_{n}.csv", parse_dates=['date'], date_format="mixed", dayfirst=True)
        columns = ["date", "sec_id", "Marked Cap USD", "BookRatio", "ROE", "CapEx over Assets", "sector_2", "domicile_country_code"]
        df = df.loc[:,columns]

        df.to_sql("factors", conn, if_exists="append", index=False, dtype={"date": "TEXT"})
      
    df = pd.read_csv("Intagibles/goodwill_ws_item_18280_region_NA.csv")
    df.to_sql("goodwill", conn, if_exists="replace", index=False)

    df = pd.read_csv("Intagibles/rd_ws_item_1201_region_NA.csv")
    df.to_sql("rd", conn, if_exists="replace", index=False)

    df = pd.read_csv("Intagibles/sga_ws_item_1101_region_NA.csv")
    df.to_sql("sga", conn, if_exists="replace", index=False)

    conn.cursor().execute(f"""CREATE TABLE data 
                          AS SELECT * FROM (WITH ranked_factors AS (
  SELECT *, 
         ROW_NUMBER() OVER (PARTITION BY sec_id, substr(date, 1, 7) ORDER BY date DESC) AS rn
  FROM factors
  WHERE domicile_country_code = 'USA'
),
dedup_factors AS (
  SELECT * 
  FROM ranked_factors 
  WHERE rn = 1
)

SELECT 
                          prices.date, 
                            prices.sec_id,
                            substr(Sector, 1, 4) as sector_2, 
                            prices.tot_ret_usd, 
                            dedup_factors.[Marked Cap USD] as market_cap, 
                            qblue.close as market_return, 
                            dedup_factors.[Marked Cap USD] * dedup_factors.BookRatio as book_value, 
                            dedup_factors.ROE, dedup_factors.[CapEx over Assets] as capex_over_assets, 
                            goodwill.value as goodwill_value,
                            rd.value as rd_value,
                            sga.value as sga_value
FROM prices
INNER JOIN dedup_factors
  ON prices.sec_id = dedup_factors.sec_id 
  AND substr(prices.date, 1, 7) = substr(dedup_factors.date, 1, 7)
LEFT JOIN gics ON gics.SecId = prices.sec_id AND prices.date > gics.FromDate AND prices.date < gics.ToDate 
LEFT JOIN qblue ON prices.date = qblue.date 
LEFT JOIN goodwill ON prices.sec_id = goodwill.sec_id AND instr(prices.date, goodwill.fiscal_period)
LEFT JOIN rd ON prices.sec_id = rd.sec_id AND instr(prices.date, rd.fiscal_period)
LEFT JOIN sga ON prices.sec_id = sga.sec_id AND instr(prices.date, sga.fiscal_period)                       
)""")

    conn.cursor().execute(f"DROP TABLE factors")
    conn.cursor().execute(f"DROP TABLE prices")
    conn.cursor().execute(f"DROP TABLE gics")
    conn.cursor().execute(f"DROP TABLE goodwill")
    conn.cursor().execute(f"DROP TABLE rd")
    conn.cursor().execute(f"DROP TABLE sga")
    conn.commit()
    conn.close()

init_database()

def validate_database():
    conn = sqlite3.connect(f"database.db")
    pd.set_option('display.max_columns', 1000) 

    df = pd.read_sql_query("SELECT * FROM data where date", conn)

    print(df)

    conn.close()

validate_database()
