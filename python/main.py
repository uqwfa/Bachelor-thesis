import dask.dataframe as dd
import pandas as pd
import numpy as np
import dask
import time
import sys
import os


def create_pairs(df, feature_cols, target_col):
    pairs = []

    for date, group in df.groupby(level="eom"):

        for i in range(len(group)):
            for j in range(i+1, len(group)):
                row1 = group.iloc[i]
                row2 = group.iloc[j]

                print(f"i: {i}, j: {j}")

                features = row1[feature_cols].values - row2[feature_cols].values

                if row1[target_col] < row2[target_col]:
                    label = 1

                else:
                    label = 0

                pairs.append((features, label))

    X_pairs = np.array([p[0] for p in pairs])
    y_pairs = np.array([p[1] for p in pairs])
    return X_pairs, y_pairs

dataLoc = "Z:/"
storeLoc = "./data/"

# t = time.time()

X_dff = dd.read_parquet(os.path.join(dataLoc, "characteristics_ranknorm.pq"))
# y_dff = dd.read_parquet(os.path.join(dataLoc, "returns_ranked.pq"))

# full_dff = X_dff.merge(y_dff, left_index=True, right_index=True)
# full_dff.reset_index().to_parquet(os.path.join(storeLoc, "full_reset_data.pq"))

# print(f"DataFrames merged and saved. Time taken: {(time.time() - t) / 60:.2f} min.")

t = time.time()

full_dff = dd.read_parquet(os.path.join(storeLoc, "full_reset_data.pq"))

print("Sorting values...")
full_dff = full_dff.sort_values(by=["eom", "permno"])

print("Setting the index to 'eom'...")
full_ddf = full_dff.set_index("eom", sorted=True)
print(f"Index set and sorted in {(time.time() - t) / 60:.2f} min.")

t = time.time()

train_sample_ddf = full_ddf.loc["2010-01-01":"2010-01-31"]

print("Computing training sample into memory...")
train_df = train_sample_ddf.compute()
print(f"Computation complete in {(time.time() - t) / 60:.2f} min.")

print(len(train_df))
print(train_df.shape)
print(train_df.head(10))
print(train_df["permno"].max())
print(train_df["permno"].min())

t = time.time()

feature_cols = X_dff.columns.tolist()
X_train, y_train = create_pairs(train_df, feature_cols, "ret_exc_lead1m_rank")

print(f"\nminutes taken: {(time.time() - t) / 60:.2f}")
sys.exit(1)
