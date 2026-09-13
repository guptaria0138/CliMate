"""
test.py
-------
Loads a trained model, evaluates it on the held-out test set,
and produces every table/figure required:
  - RMSE, MAE, R2 (regression)
  - Accuracy, Precision, Recall, F1, Specificity, ROC-AUC (classification,
    via an above/below-median threshold on the true test values)
  - Confusion matrix
  - Train-vs-test loss/MAE bars
  - Loss/MAE vs epoch curves
  - True-vs-predicted scatter
  - Monthly / yearly bias tables
  - Feature correlation heatmap
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              precision_score, recall_score, f1_score, accuracy_score,
                              confusion_matrix, roc_auc_score, roc_curve)

from preprocessing import load_raw, split_scale, create_sequences, TIME_STEPS


def main(model_name="baseline"):
    df = load_raw()
    (X_train, X_val, X_test, y_train, y_val, y_test,
     x_scaler, y_scaler, split_info) = split_scale(df)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)

    model = tf.keras.models.load_model(
        f"models/{model_name}_model.keras",
        custom_objects={},  # add {"TemporalAttention": TemporalAttention} if loading the attention model
    )
    with open(f"models/{model_name}_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    y_scaler = scalers["y_scaler"]

    # ---- Regression metrics (in original Kelvin units) ----
    y_pred_scaled = model.predict(X_test_seq).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.3f} K | MAE: {mae:.3f} K | R2: {r2:.4f}")

    # ---- Classification metrics (above/below median of TRUE test values) ----
    threshold = np.median(y_true)
    y_true_class = (y_true >= threshold).astype(int)
    y_pred_class = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class)
    recall = recall_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)
    cm = confusion_matrix(y_true_class, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    # ROC-AUC needs a continuous score; use the raw predicted temperature
    auc = roc_auc_score(y_true_class, y_pred)

    metrics = {
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "F1": f1, "Specificity": specificity, "ROC_AUC": auc,
    }
    print(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(f"results/{model_name}_metrics.csv", index=False)

    # ---- Confusion matrix plot ----
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Below median", "Above median"],
                yticklabels=["Below median", "Above median"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout(); plt.savefig(f"figures/{model_name}_confusion_matrix.png"); plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_true_class, y_pred)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})"); plt.legend()
    plt.tight_layout(); plt.savefig(f"figures/{model_name}_roc_curve.png"); plt.close()

    # ---- Train vs test loss/MAE ----
    train_loss, train_mae = model.evaluate(X_train_seq, y_train_seq, verbose=0)
    test_loss, test_mae = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    plt.figure(figsize=(6, 4))
    plt.bar(["Train Loss", "Test Loss"], [train_loss, test_loss], color=["steelblue", "orange"])
    plt.ylabel("Loss (MSE, scaled)"); plt.title(f"Train vs Test Loss ({model_name})")
    plt.tight_layout(); plt.savefig(f"figures/{model_name}_train_test_loss.png"); plt.close()

    # ---- Loss / MAE vs epoch ----
    history = pd.read_csv(f"results/{model_name}_history.csv")
    for col_pair, fname in [(("loss", "val_loss"), "loss_curve"), (("mae", "val_mae"), "mae_curve")]:
        plt.figure(figsize=(7, 4))
        plt.plot(history[col_pair[0]], label=f"Train {col_pair[0]}")
        plt.plot(history[col_pair[1]], label=f"Val {col_pair[1]}")
        plt.xlabel("Epoch"); plt.legend(); plt.grid(True)
        plt.title(f"{col_pair[0].upper()} vs Epoch ({model_name})")
        plt.tight_layout(); plt.savefig(f"figures/{model_name}_{fname}.png"); plt.close()

    # ---- True vs predicted scatter ----
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True Temp (K)"); plt.ylabel("Predicted Temp (K)")
    plt.title(f"True vs Predicted ({model_name})")
    plt.tight_layout(); plt.savefig(f"figures/{model_name}_scatter.png"); plt.close()

    # ---- Monthly / yearly bias ----
    df_test = df.iloc[split_info["train_samples"] + split_info["val_samples"] + TIME_STEPS:].copy()
    df_test = df_test.iloc[:len(y_true)]
    df_test["y_true"], df_test["y_pred"] = y_true, y_pred
    df_test["year"], df_test["month"] = df_test.index.year, df_test.index.month

    yearly_bias = df_test.groupby("year").apply(lambda g: mean_absolute_error(g.y_true, g.y_pred))
    monthly_bias = df_test.groupby("month").apply(lambda g: mean_absolute_error(g.y_true, g.y_pred))
    yearly_bias.to_csv(f"results/{model_name}_yearly_bias.csv")
    monthly_bias.to_csv(f"results/{model_name}_monthly_bias.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(monthly_bias.index, monthly_bias.values, marker="o")
    plt.xlabel("Month"); plt.ylabel("MAE (K)"); plt.title(f"Monthly Bias ({model_name})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"figures/{model_name}_monthly_bias.png"); plt.close()

    # ---- Correlation heatmap (full dataset, run once regardless of model) ----
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.drop(columns=["latitude", "longitude"]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap"); plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png"); plt.close()

    print(f"\nAll figures saved to figures/, metrics saved to results/{model_name}_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=["baseline", "attention"])
    args = parser.parse_args()
    main(args.model)
