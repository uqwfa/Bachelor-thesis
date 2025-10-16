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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import Sequence


def create_ranknet_model(num_features):
    model = models.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ], name="scoring_model")

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
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        X_i_batch = []
        X_j_batch = []
        y_batch = []

        while len(X_i_batch) < self.batch_size:
            group_idx = np.random.choice(self.group_indices)
            _, group = self.grouped[group_idx]

            if len(group) < 2:
                continue

            indices = np.random.choice(len(group), 2, replace=False)
            row1 = group.iloc[indices[0]]
            row2 = group.iloc[indices[1]]

            features_i = row1[self.feature_cols].values
            features_j = row2[self.feature_cols].values

            if row1[self.target_col] < row2[self.target_col]:
                label = 1

            else:
                label = 0

            X_i_batch.append(features_i)
            X_j_batch.append(features_j)
            y_batch.append(label)

        return (np.array(X_i_batch), np.array(X_j_batch)), np.array(y_batch)

    def on_epoch_end(self):
        np.random.shuffle(self.group_indices)


dataLoc = "Z:/"
storeLoc = "./data/"
modelLoc = "./model/"

# t = time.time()

# X_dff = dd.read_parquet(os.path.join(dataLoc, "characteristics_ranknorm.pq"))
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

# print(len(train_df))
# print(train_df.shape)
print(train_df.head(10))
# print(train_df["permno"].max())
# print(train_df["permno"].min())


all_dates = train_df.index.unique()
split_point = int(len(all_dates) * 0.85)
train_dates = all_dates[:split_point]
test_dates = all_dates[split_point:]

train_data = train_df.loc[train_dates]
val_data = train_df.loc[test_dates]

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

feature_cols = [col for col in train_df.columns if col not in ["permno", "ret_exc_lead1m", "ret_exc_lead1m_rank"]]
num_features = len(feature_cols)
print(f"Number of features: {num_features}")
batch_size = 1024

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

scoring_model = create_ranknet_model(num_features)

input_i = Input(shape=(num_features,), name="item_i")
input_j = Input(shape=(num_features,), name="item_j")

score_i = scoring_model(input_i)
score_j = scoring_model(input_j)

diff = layers.Subtract()([score_i, score_j])
ranknet_model = models.Model(inputs=[input_i, input_j], outputs=diff, name="RankNet_Model")

ranknet_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

ranknet_model.summary()

early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("Training...")
t = time.time()

history = ranknet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=2,
    callbacks=[early_stopper]
)

print(f"Training complete in {(time.time() - t) / 60:.2f} min.")

scoring_model.save(os.path.join(modelLoc, "scoring_ranknet.keras"))

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
