import matplotlib.pyplot as plt
import dask.dataframe as dd
import tensorflow as tf
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import dask
import time
import sys
import os
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence


def create_ranknet_model(num_features):
    model = models.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])

    return model


class DataGenerator(Sequence):
    def __init__(self, df, feature_cols, target_col, batch_size=128, samples_per_stock=10):
        super().__init__()

        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.samples_per_stock = samples_per_stock

        self.grouped = list(df.groupby(level="eom"))
        self.group_indices = np.arange(len(self.grouped))
        self.total_samples = len(df)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.total_samples / self.batch_size))

    def __getitem__(self, idx):
        X_batch = []
        y_batch = []

        while len(X_batch) < self.batch_size:
            group_idx = np.random.choice(self.group_indices)
            _, group = self.grouped[group_idx]

            if len(group) < 2:
                continue

            anchor_idx = np.random.randint(0, len(group))
            row1 = group.iloc[anchor_idx]

            for _ in range(self.samples_per_stock):
                other_idx = np.random.randint(0, len(group))

                while other_idx == anchor_idx:
                    other_idx = np.random.randint(0, len(group))

                row2 = group.iloc[other_idx]

                features = row1[self.feature_cols].values - row2[self.feature_cols].values

                if row1[self.target_col] < row2[self.target_col]:
                    label = 1
                else:
                    label = 0

                X_batch.append(features)
                y_batch.append(label)

                if len(X_batch) >= self.batch_size:
                    break

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        np.random.shuffle(self.group_indices)


dataLoc = "Z:/"
storeLoc = "./data/"
modelLoc = "./model/"

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

train_sample_ddf = full_ddf.loc["2000-01-01":"2000-12-31"]

print("Computing training sample into memory...")
train_df = train_sample_ddf.compute()
print(f"Computation complete in {(time.time() - t) / 60:.2f} min.")

print(len(train_df))
print(train_df.shape)
print(train_df.head(10))
print(train_df["permno"].max())
print(train_df["permno"].min())


all_dates = train_df.index.unique()
split_point = int(len(all_dates) * 0.85)
train_dates = all_dates[:split_point]
test_dates = all_dates[split_point:]

train_data = train_df.loc[train_dates]
val_data = train_df.loc[test_dates]

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

feature_cols = X_dff.columns.tolist()
num_features = len(feature_cols)
batch_size = 128

train_generator = DataGenerator(
    train_data,
    feature_cols=feature_cols,
    target_col="ret_exc_lead1m_rank",
    batch_size=batch_size,
    samples_per_stock=10
)

val_generator = DataGenerator(
    val_data,
    feature_cols=feature_cols,
    target_col="ret_exc_lead1m_rank",
    batch_size=batch_size,
    samples_per_stock=10
)

model = create_ranknet_model(num_features)
model.summary()
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Training...")
t = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopper]
)

print(f"Training complete in {(time.time() - t) / 60:.2f} min.")

# Erstelle eine neue Abbildung
plt.figure(figsize=(10, 6))

# Plotte den Trainingsverlust aus dem history-Objekt
plt.plot(history.history['loss'], label='Trainings-Verlust')

# Plotte den Validierungsverlust aus dem history-Objekt
plt.plot(history.history['val_loss'], label='Validierungs-Verlust')

# Füge Titel und Beschriftungen hinzu, um den Graphen verständlich zu machen
plt.title('Verlustverlauf während des Trainings')
plt.xlabel('Epochen')
plt.ylabel('Verlust (Binary Cross-Entropy)')
plt.legend()
plt.grid(True)

# Zeige den Graphen an
plt.show()
