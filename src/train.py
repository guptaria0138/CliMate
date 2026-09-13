"""
train.py
--------
Runs 5-fold CV for bias/variance estimation, then does the final fit on train/val,
and saves everything test.py needs.
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold

from preprocessing import load_raw, dataset_report, split_scale, create_sequences, TIME_STEPS
from model import build_baseline_lstm, build_attention_lstm


def get_model_fn(name):
    if name == "baseline":
        return lambda n_feat: build_baseline_lstm(n_feat, TIME_STEPS)
    elif name == "attention":
        return lambda n_feat: build_attention_lstm(n_feat, TIME_STEPS, use_attention=True)
    else:
        raise ValueError(f"Unknown model name: {name}")


def main(model_name="baseline", epochs=30, batch_size=128):
    df = load_raw()
    report = dataset_report(df)
    print("Dataset:", report["n_samples"], "rows |", report["n_features"], "features")

    (X_train, X_val, X_test, y_train, y_val, y_test,
     x_scaler, y_scaler, split_info) = split_scale(df)
    print("Split:", split_info)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val)
    n_features = X_train_seq.shape[-1]

    build_fn = get_model_fn(model_name)

    # ---- 5-fold CV for bias/variance (Section 5E) ----
    kf = KFold(n_splits=5, shuffle=False)  # shuffle=False: keep temporal blocks intact
    cv_losses = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        Xtr_seq, ytr_seq = create_sequences(Xtr, ytr)
        Xva_seq, yva_seq = create_sequences(Xva, yva)
        if len(Xtr_seq) == 0 or len(Xva_seq) == 0:
            continue
        m = build_fn(n_features)
        m.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
        loss = m.evaluate(Xva_seq, yva_seq, verbose=0)[0]
        cv_losses.append(loss)
        print(f"Fold {fold + 1} val_loss={loss:.5f}")

    bias = float(np.mean(cv_losses))
    variance = float(np.var(cv_losses))
    print(f"\nCV Bias (avg loss): {bias:.5f} | CV Variance: {variance:.5f}")

    # ---- Final training on train, validated on val (test stays untouched) ----
    model = build_fn(n_features)
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs, batch_size=batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
    )

    # ---- Save everything test.py needs ----
    model.save(f"models/{model_name}_model.keras")
    with open(f"models/{model_name}_scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)
    pd.DataFrame(history.history).to_csv(f"results/{model_name}_history.csv", index=False)
    with open(f"results/{model_name}_cv_bias_variance.json", "w") as f:
        json.dump({"cv_losses": cv_losses, "bias": bias, "variance": variance}, f, indent=2)
    with open("results/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved model to models/{model_name}_model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=["baseline", "attention"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args.model, args.epochs, args.batch_size)
