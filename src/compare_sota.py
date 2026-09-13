"""
compare_sota.py 
-------------------------------------------------------
Trains a handful of reference models ON THE SAME DATA/SPLIT as 
proposed LSTM, so the comparison table in the README is a fair,
reproducible, same-dataset comparison rather than numbers copied from a
different paper's different dataset.

Included baselines:
  1. Persistence (naive forecast: predict next hour = current hour) -
     the standard baseline every forecasting paper reports.
  2. Linear Regression on the flattened window.
  3. Dense (MLP) network on the flattened window.
  4. GRU (a natural "SOTA-lite" competitor to LSTM).
  5. Your proposed model (baseline LSTM or attention LSTM) - just read its
     metrics.csv from test.py and append it to this table.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import load_raw, split_scale, create_sequences, TIME_STEPS


def evaluate_regression(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n_params = None
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "Parameters": n_params}


def main(epochs=15, batch_size=128):
    df = load_raw()
    (X_train, X_val, X_test, y_train, y_val, y_test,
     x_scaler, y_scaler, split_info) = split_scale(df)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)
    n_features = X_train_seq.shape[-1]

    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    rows = []

    # 1. Persistence baseline: predict last observed temperature column value
    target_col_idx = None  # temperature was removed from X; use y_test itself shifted by 1
    y_persist_scaled = np.roll(y_test.flatten(), 1)[TIME_STEPS:]
    y_persist = y_scaler.inverse_transform(y_persist_scaled.reshape(-1, 1)).flatten()
    rows.append(evaluate_regression(y_true, y_persist, "Persistence (naive)"))

    # 2. Linear Regression on flattened window
    X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
    X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)
    lr = LinearRegression().fit(X_train_flat, y_train_seq.ravel())
    y_pred_lr = y_scaler.inverse_transform(lr.predict(X_test_flat).reshape(-1, 1)).flatten()
    rows.append(evaluate_regression(y_true, y_pred_lr, "Linear Regression"))

    # 3. Dense (MLP) baseline
    mlp = tf.keras.Sequential([
        layers.Input(shape=(X_train_flat.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    mlp.compile(optimizer="adam", loss="mse", metrics=["mae"])
    mlp.fit(X_train_flat, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_mlp = y_scaler.inverse_transform(mlp.predict(X_test_flat).reshape(-1, 1)).flatten()
    row = evaluate_regression(y_true, y_pred_mlp, "Dense MLP")
    row["Parameters"] = mlp.count_params()
    rows.append(row)

    # 4. GRU baseline
    gru = tf.keras.Sequential([
        layers.Input(shape=(TIME_STEPS, n_features)),
        layers.GRU(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    gru.compile(optimizer="adam", loss="mse", metrics=["mae"])
    gru.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_gru = y_scaler.inverse_transform(gru.predict(X_test_seq).reshape(-1, 1)).flatten()
    row = evaluate_regression(y_true, y_pred_gru, "GRU")
    row["Parameters"] = gru.count_params()
    rows.append(row)

    # 5. Proposed model - pull in the already-computed metrics from test.py if available
    try:
        proposed = pd.read_csv("results/baseline_metrics.csv").iloc[0]
        rows.append({"Model": "Proposed LSTM", "RMSE": proposed["RMSE"],
                      "MAE": proposed["MAE"], "R2": proposed["R2"], "Parameters": None})
    except FileNotFoundError:
        print("Run `python src/test.py --model baseline` first to include the proposed model here.")

    try:
        proposed_att = pd.read_csv("results/attention_metrics.csv").iloc[0]
        rows.append({"Model": "Proposed LSTM + Attention (ours)", "RMSE": proposed_att["RMSE"],
                      "MAE": proposed_att["MAE"], "R2": proposed_att["R2"], "Parameters": None})
    except FileNotFoundError:
        pass

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv("results/comparison.csv", index=False)
    print(comparison_df.to_string(index=False))
    print("\nSaved to results/comparison.csv")


if __name__ == "__main__":
    main()
