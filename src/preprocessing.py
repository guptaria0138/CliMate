"""
preprocessing.py
----------------
Loads the ERA5 hourly weather CSV (Delhi, 2024-2026), cleans it, reports
dataset statistics required by the rubric, scales features,
splits into train/validation/test, and builds LSTM input sequences.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TARGET = "Temp 2m"
DROP_COLS = ["latitude", "longitude"]  # constant columns, no predictive value
TIME_STEPS = 12


def load_raw(csv_path="data/2024-2026.csv"):
    df = pd.read_csv(csv_path)
    df["valid_time"] = pd.to_datetime(df["valid_time"], format="%d-%m-%Y %H:%M")
    df.set_index("valid_time", inplace=True)
    return df


def dataset_report(df):
    """Prints / returns the Section 5A 'Dataset Analysis' tables."""
    report = {}
    report["n_samples"] = len(df)
    report["n_features"] = df.shape[1]
    report["date_range"] = (df.index.min(), df.index.max())
    report["missing_values"] = df.isna().sum().to_dict()
    report["describe"] = df.describe()

    median_temp = df[TARGET].median()
    report["binary_class_balance"] = {
        "below_median (0)": int((df[TARGET] < median_temp).sum()),
        "above_or_equal_median (1)": int((df[TARGET] >= median_temp).sum()),
    }
    return report


def split_scale(df, train_frac=0.7, val_frac=0.15):
    """
    Chronological 70/15/15 train/val/test split (never shuffle time series
    randomly). Returns scaled arrays + the fitted scalers so predictions
    can be inverse-transformed later.
    """
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    X = df.drop(columns=DROP_COLS + [TARGET]).values
    y = df[TARGET].values.reshape(-1, 1)

    X_train_raw, X_val_raw, X_test_raw = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train_raw, y_val_raw, y_test_raw = y[:train_end], y[train_end:val_end], y[val_end:]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    split_info = {
        "train_samples": len(X_train), "train_pct": round(100 * len(X_train) / n, 2),
        "val_samples": len(X_val), "val_pct": round(100 * len(X_val) / n, 2),
        "test_samples": len(X_test), "test_pct": round(100 * len(X_test) / n, 2),
    }

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            x_scaler, y_scaler, split_info)


def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


if __name__ == "__main__":
    df = load_raw()
    report = dataset_report(df)
    print("Samples:", report["n_samples"], "Features:", report["n_features"])
    print("Date range:", report["date_range"])
    print("Missing values per column:\n", report["missing_values"])
    print("Binary class balance (for classification metrics only):", report["binary_class_balance"])

    (X_train, X_val, X_test, y_train, y_val, y_test,
     x_scaler, y_scaler, split_info) = split_scale(df)
    print("Split info:", split_info)
