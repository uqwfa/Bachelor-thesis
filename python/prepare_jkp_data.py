# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:52:39 2024

@author: moerk
"""

import dask.dataframe as dd
import pandas as pd
import time
import sys
import os


def char_norm(ddf):
    # Calculate the median for each characteristic within each month,
    # and broadcast the result back to the original shape.
    group_medians = ddf.groupby(lebel="eom").transform("median")

    # Fill the NaNs in the original dataframe with the calculated medians.
    df_medfill = ddf.fillna(group_medians)

    # Calculate the ranks within each group (month).
    ranks = df_medfill.groupby(level="eom").rank()

    # Calculate the counts within each group and broadcast them.
    counts = df_medfill.groupby(level="eom").transform("count")

    # Apply the normalization formula using vectorized operations.
    df_ranknorm = 2 * ranks / counts - 1

    # Fill remaining NaNs with 0
    df_ranknorm = df_ranknorm.fillna(0)

    return df_ranknorm


# %% Read data
dataLoc = "Z:/"

jkp = pd.read_excel(os.path.join(dataLoc, 'Factor Details.xlsx'))['abr_jkp'].dropna().tolist()

# print(jkp)

dtype_spec = {
    'adjfct': 'float64',
    'gvkey': 'float64',
    'iid': 'object',
    'naics': 'float64',
    'sic': 'float64',
    'tvol': 'float64'
}

t = time.time()

# usa_data = pd.read_csv(os.path.join(dataLoc, 'usa.csv'))
usa_data = dd.read_csv(os.path.join(dataLoc, 'usa.csv'), blocksize="16MB", dtype=dtype_spec)

# 2. Now, inspect the Dask DataFrame's properties
# print(f"Columns: {usa_data.columns.tolist()}")
# print(f"Partitions: {usa_data.npartitions}")
# print(f"Divisions: {usa_data.divisions}")

# convert eom to datetime
usa_data["eom"] = dd.to_datetime(usa_data["eom"], format="%Y%m%d")

# %% Clean data

# Keep only common stocks listed in the three prime exchanges and data from CRSP
# source_crsp almost certainly refers to data originating from CRSP (Center for Research in Security Prices).
usa_data = usa_data[(usa_data.source_crsp == 1)]

# crsp_shrcd: This stands for CRSP Share Code. It's a numeric code provided by CRSP to classify the type of security.
# 10: Represents an ordinary common stock that has no special conditions. This is the most common type of equity.
# 11: Represents an ordinary common stock with various special conditions (e.g., when-issued, subject to approval, etc.).
usa_data = usa_data[(usa_data.crsp_shrcd == 10) | (usa_data.crsp_shrcd == 11)]

# crsp_exchcd: This stands for CRSP Exchange Code. It's a numeric code that identifies the primary exchange on which the security is listed.
# 1: Represents the NYSE (New York Stock Exchange).
# 2: Represents the NYSE MKT (formerly the AMEX or American Stock Exchange).
# 3: Represents the NASDAQ.
usa_data = usa_data[(usa_data.crsp_exchcd == 1) | (usa_data.crsp_exchcd == 2) | (usa_data.crsp_exchcd == 3)]

# Remove stocks with price not higher than 5
usa_data = usa_data[usa_data.prc > 5]

# Drop observations before March 1957
usa_data = usa_data[usa_data.eom > "1957-03-01"]

# eom: end of month date; permno: unique stock identifier
index_col = ["eom", "permno"]
# sorts chronologically by end of month date and within each date sorts by permno
usa_data = usa_data.sort_values(index_col)
print("finished sorting")
usa_data = usa_data.set_index("eom", sorted=True)
print("finished setting index")

characteristics = usa_data[jkp].copy()

characteristics_ranknorm = char_norm(characteristics)
# characteristics_ranknorm.to_parquet(os.path.join("./data/", "characteristics_ranknorm.pq"))

print(f"Dask Head after:\n{characteristics.head(10)}")
print(f"took minutes: {(time.time() - t) / 60:.2f}")
sys.exit(1)

# Work on returns: ret_exc_lead1m is the next month return in excess of the risk-free rate
returns = usa_data[['ret_exc_lead1m']]
returns["ret_exc_lead1m_rank"] = returns.groupby("eom").ret_exc_lead1m.rank(ascending=False, method="dense")
returns.to_parquet(os.path.join(dataLoc, "returns_ranked.pq"))