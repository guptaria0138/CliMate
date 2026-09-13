"""
ablation.py 
------------------------------------------
Quantifies the contribution of each component of the proposed innovation
(temporal attention + optional bidirectional encoder) by training every
combination on the identical train/val/test split and comparing test RMSE/MAE.

This produces exactly the Section 5D table the guideline asks for:

| Experiment | Attention | Bidirectional | RMSE | MAE |
|------------|-----------|----------------|------|-----|
| Baseline   |     x     |       x        |      |     |
| + Attn     |     v     |       x        |      |     |
| + BiLSTM   |     x     |       v        |      |     |
| Proposed   |     v     |       v        |      |     |
"""

import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing import load_raw, split_scale, create_sequences, TIME_STEPS
from model import build_attention_lstm


def run_experiment(name, use_attention, use_bidirectional,
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
                    y_scaler, n_features, epochs=25, batch_size=128):
    model = build_attention_lstm(n_features, TIME_STEPS,
                                  use_attention=use_attention,
                                  use_bidirectional=use_bidirectional)
    model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
              epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)])

    y_pred = y_scaler.inverse_transform(model.predict(X_test_seq).reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"Experiment": name, "Attention": use_attention, "Bidirectional": use_bidirectional,
            "RMSE": rmse, "MAE": mae, "Parameters": model.count_params()}


def main(epochs=25, batch_size=128):
    df = load_raw()
    (X_train, X_val, X_test, y_train, y_val, y_test,
     x_scaler, y_scaler, split_info) = split_scale(df)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)
    n_features = X_train_seq.shape[-1]

    configs = [
        ("Baseline (no attention, no BiLSTM)", False, False),
        ("+ Attention only", True, False),
        ("+ Bidirectional only", False, True),
        ("Proposed (Attention + Bidirectional)", True, True),
    ]

    results = []
    for name, use_att, use_bidir in configs:
        print(f"\nRunning: {name}")
        results.append(run_experiment(
            name, use_att, use_bidir,
            X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
            y_scaler, n_features, epochs, batch_size
        ))
        print(results[-1])

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv("results/ablation_study.csv", index=False)
    print("\n", ablation_df.to_string(index=False))
    print("\nSaved to results/ablation_study.csv")


if __name__ == "__main__":
    main()
