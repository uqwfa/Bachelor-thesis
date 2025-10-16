import matplotlib.pyplot as plt
import dask.dataframe as dd
import tensorflow as tf
import pandas as pd
import time
import os
from tqdm import tqdm


dataLoc = "Z:/"
storeLoc = "./data/"
modelLoc = "./model/"


def simulate_model(model, data, feature_cols, quintile=0.1):
    montly_results = []

    for date, group in tqdm(data.groupby(level="eom"), total=data.index.nunique()):
        features_batch = group[feature_cols].values
        pred_scores = model.predict(features_batch, verbose=0).flatten()

        results_df = group[['permno', 'ret_exc_lead1m', "ret_exc_lead1m_rank"]].copy()
        results_df['pred_score'] = pred_scores
        results_df["pred_rank"] = results_df["pred_score"].rank(ascending=False, method="first")

        # print("Top 10 Predicted Ranks:")
        # print(results_df.sort_values(by="pred_rank").head(10))
        # print("\nTop 10 True Ranks:")
        # print(results_df.sort_values(by="ret_exc_lead1m_rank").head(10))

        cutoff = int(len(results_df) * quintile)
        top_quintile = results_df.nsmallest(cutoff, 'pred_rank')
        bottom_quintile = results_df.nlargest(cutoff, 'pred_rank')

        # print(cutoff)
        # print(top_quintile.head(2))
        # print(top_quintile.tail(2))
        # print(bottom_quintile.head(2))
        # print(bottom_quintile.tail(2))

        top_avg_return = top_quintile['ret_exc_lead1m'].mean()
        bottom_avg_return = bottom_quintile['ret_exc_lead1m'].mean()
        long_short_return = top_avg_return - bottom_avg_return

        montly_results.append({
            "date": date,
            "top_avg_return": top_avg_return,
            "bottom_avg_return": bottom_avg_return,
            "long_short_return": long_short_return
        })

    return montly_results


def plot_backtest_results(results):
    results_df = pd.DataFrame(results)
    results_df.set_index("date", inplace=True)
    results_df = results_df.sort_index()

    print(results_df.describe())

    results_df['cumulative_performance'] = (1 + results_df['long_short_return']).cumprod() - 1

    plt.figure(figsize=(12, 7))
    plt.plot(results_df.index, results_df['cumulative_performance'], label='Long-Short Strategy')

    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    plt.title('Cumulative Performance of RankNet Long-Short Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    results_df.to_csv(os.path.join(dataLoc, "ranknet_results.csv"))

    plt.show()


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(modelLoc, "scoring_ranknet.h5"))

    t = time.time()
    full_dff = dd.read_parquet(os.path.join(storeLoc, "full_reset_data.pq"))
    full_dff = full_dff.sort_values(by=["eom", "permno"])
    full_ddf = full_dff.set_index("eom", sorted=True)
    eval_df = full_ddf.compute()
    print(f"Data loaded and prepped in {(time.time() - t) / 60:.2f} min.")

    feature_cols = [col for col in eval_df.columns if col not in ["permno", "ret_exc_lead1m", "ret_exc_lead1m_rank"]]

    backtest_results = simulate_model(model, eval_df, feature_cols)
    plot_backtest_results(backtest_results)
