"""
generate_placeholder_results.py
--------------------------------
Produces EVERY figure/table likeconfusion
matrix, train-vs-test loss & MAE bars, loss/MAE vs epoch, scatter,
correlation heatmap, monthly bias trend, yearly bias, rain-vs-dry bias,
and a 5-fold CV bias/variance chart and comparison table.

"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              precision_score, recall_score, f1_score, accuracy_score,
                              confusion_matrix, roc_auc_score, roc_curve)

from preprocessing import load_raw, dataset_report, split_scale, create_sequences, TIME_STEPS

RESULTS = "results"
FIGURES = "figures"

# ---------------------------------------------------------------- load & split
df = load_raw("data/2024-2026.csv")
report = dataset_report(df)
(X_train, X_val, X_test, y_train, y_val, y_test,
 x_scaler, y_scaler, split_info) = split_scale(df)

with open(f"{RESULTS}/split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)
with open(f"{RESULTS}/dataset_report.json", "w") as f:
    json.dump({"n_samples": report["n_samples"], "n_features": report["n_features"],
                "missing_values": report["missing_values"],
                "binary_class_balance": report["binary_class_balance"]}, f, indent=2, default=str)

X_train_seq, y_train_seq = create_sequences(X_train, y_train)
X_val_seq, y_val_seq = create_sequences(X_val, y_val)
X_test_seq, y_test_seq = create_sequences(X_test, y_test)

X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
X_val_flat = X_val_seq.reshape(len(X_val_seq), -1)
X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)
y_train_flat, y_val_flat, y_test_flat = y_train_seq.ravel(), y_val_seq.ravel(), y_test_seq.ravel()

# ---------------------------------------------------------------- 5-fold CV bias/variance
kf = KFold(n_splits=5, shuffle=False)
cv_losses = []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    Xtr, Xva = X_train[tr_idx], X_train[va_idx]
    ytr, yva = y_train[tr_idx], y_train[va_idx]
    Xtr_seq, ytr_seq = create_sequences(Xtr, ytr)
    Xva_seq, yva_seq = create_sequences(Xva, yva)
    if len(Xtr_seq) == 0 or len(Xva_seq) == 0:
        continue
    m = MLPRegressor(hidden_layer_sizes=(64, 16), max_iter=60, random_state=42, early_stopping=False)
    m.fit(Xtr_seq.reshape(len(Xtr_seq), -1), ytr_seq.ravel())
    pred = m.predict(Xva_seq.reshape(len(Xva_seq), -1))
    loss = mean_squared_error(yva_seq.ravel(), pred)
    cv_losses.append(loss)
    print(f"Fold {fold+1}: val_loss={loss:.5f}")

bias = float(np.mean(cv_losses))
variance = float(np.var(cv_losses))
with open(f"{RESULTS}/cv_bias_variance.json", "w") as f:
    json.dump({"cv_losses": cv_losses, "bias": bias, "variance": variance}, f, indent=2)
print(f"Bias (avg loss): {bias:.5f} | Variance: {variance:.5f}")

plt.figure(figsize=(6, 4))
plt.bar([f"Fold {i+1}" for i in range(len(cv_losses))], cv_losses, color="teal")
plt.axhline(bias, color="red", linestyle="--", label=f"Mean (bias) = {bias:.4f}")
plt.ylabel("Validation MSE (scaled)")
plt.title(f"5-Fold CV Loss — Bias={bias:.4f}, Variance={variance:.6f}")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIGURES}/cv_bias_variance.png", dpi=150); plt.close()

# ---------------------------------------------------------------- main proxy model with epoch tracking
mlp = MLPRegressor(hidden_layer_sizes=(128, 32), activation="relu", solver="adam",
                    learning_rate_init=0.001, max_iter=1, warm_start=True, random_state=42)

n_epochs = 30
train_loss_hist, val_loss_hist, train_mae_hist, val_mae_hist = [], [], [], []
for epoch in range(n_epochs):
    mlp.fit(X_train_flat, y_train_flat)
    tr_pred, va_pred = mlp.predict(X_train_flat), mlp.predict(X_val_flat)
    train_loss_hist.append(mean_squared_error(y_train_flat, tr_pred))
    val_loss_hist.append(mean_squared_error(y_val_flat, va_pred))
    train_mae_hist.append(mean_absolute_error(y_train_flat, tr_pred))
    val_mae_hist.append(mean_absolute_error(y_val_flat, va_pred))

history_df = pd.DataFrame({"loss": train_loss_hist, "val_loss": val_loss_hist,
                            "mae": train_mae_hist, "val_mae": val_mae_hist})
history_df.to_csv(f"{RESULTS}/baseline_history.csv", index=False)

# Loss vs Epoch
plt.figure(figsize=(7, 4))
plt.plot(history_df["loss"], label="Train Loss")
plt.plot(history_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)"); plt.title("Loss vs Epochs")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f"{FIGURES}/loss_vs_epoch.png", dpi=150); plt.close()

# MAE vs Epoch
plt.figure(figsize=(7, 4))
plt.plot(history_df["mae"], label="Train MAE")
plt.plot(history_df["val_mae"], label="Validation MAE")
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("MAE vs Epochs")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f"{FIGURES}/mae_vs_epoch.png", dpi=150); plt.close()

# Training Loss vs Validation Loss (kept as its own figure to match the original script)
plt.figure(figsize=(8, 5))
plt.plot(history_df["loss"], label="Train Loss", color="blue")
plt.plot(history_df["val_loss"], label="Validation Loss", color="orange")
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f"{FIGURES}/training_vs_validation_loss.png", dpi=150); plt.close()

# ---------------------------------------------------------------- test set evaluation
y_pred_scaled = mlp.predict(X_test_flat)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = y_scaler.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

threshold = np.median(y_true)
y_true_c = (y_true >= threshold).astype(int)
y_pred_c = (y_pred >= threshold).astype(int)
accuracy = accuracy_score(y_true_c, y_pred_c)
precision = precision_score(y_true_c, y_pred_c)
recall = recall_score(y_true_c, y_pred_c)
f1 = f1_score(y_true_c, y_pred_c)
cm = confusion_matrix(y_true_c, y_pred_c)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
auc = roc_auc_score(y_true_c, y_pred)

metrics = {"Model": "sklearn MLP proxy (replace with LSTM)", "RMSE": rmse, "MAE": mae, "R2": r2,
           "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1,
           "Specificity": specificity, "ROC_AUC": auc}
pd.DataFrame([metrics]).to_csv(f"{RESULTS}/results.csv", index=False)
pd.DataFrame([metrics]).to_csv(f"{RESULTS}/baseline_metrics.csv", index=False)
print(json.dumps(metrics, indent=2, default=float))

# Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Below median", "Above median"], yticklabels=["Below median", "Above median"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{FIGURES}/confusion_matrix.png", dpi=150); plt.close()
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Below median", "Above median"], yticklabels=["Below median", "Above median"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{RESULTS}/confusion_matrix.png", dpi=150); plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_true_c, y_pred)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
plt.legend(); plt.tight_layout(); plt.savefig(f"{FIGURES}/roc_curve.png", dpi=150); plt.close()

# Train vs Test Loss / MAE bars
train_loss_final = mean_squared_error(y_train_flat, mlp.predict(X_train_flat))
test_loss_final = mean_squared_error(y_test_flat, mlp.predict(X_test_flat))
train_mae_final = mean_absolute_error(y_train_flat, mlp.predict(X_train_flat))
test_mae_final = mean_absolute_error(y_test_flat, mlp.predict(X_test_flat))

plt.figure(figsize=(6, 4))
plt.bar(["Train Loss", "Test Loss"], [train_loss_final, test_loss_final], color=["blue", "orange"])
plt.title("Training vs Testing Loss"); plt.ylabel("Loss (MSE)"); plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout(); plt.savefig(f"{FIGURES}/train_test_loss_bar.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
plt.bar(["Train MAE", "Test MAE"], [train_mae_final, test_mae_final], color=["green", "red"])
plt.title("Training vs Testing MAE"); plt.ylabel("MAE"); plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout(); plt.savefig(f"{FIGURES}/train_test_mae_bar.png", dpi=150); plt.close()

# Scatter: True vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.4, s=8)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("True Values (K)"); plt.ylabel("Predicted Values (K)"); plt.title("Scatter Plot: True vs Predicted")
plt.tight_layout(); plt.savefig(f"{FIGURES}/scatter_true_vs_pred.png", dpi=150); plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.drop(columns=["latitude", "longitude"]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap"); plt.tight_layout()
plt.savefig(f"{FIGURES}/correlation_heatmap.png", dpi=150); plt.close()

# ---------------------------------------------------------------- bias analysis (yearly/monthly/rain-dry)
df_test = df.iloc[split_info["train_samples"] + split_info["val_samples"] + TIME_STEPS:].copy()
df_test = df_test.iloc[:len(y_true)]
df_test["y_true"], df_test["y_pred"] = y_true, y_pred
df_test["year"], df_test["month"] = df_test.index.year, df_test.index.month

yearly_bias = df_test.groupby("year").apply(lambda g: mean_absolute_error(g.y_true, g.y_pred))
monthly_bias = df_test.groupby("month").apply(lambda g: mean_absolute_error(g.y_true, g.y_pred))
yearly_bias.to_csv(f"{RESULTS}/yearly_bias.csv")
monthly_bias.to_csv(f"{RESULTS}/monthly_bias.csv")
print("Yearly Bias:\n", yearly_bias.round(4))
print("Monthly Bias:\n", monthly_bias.round(4))

plt.figure(figsize=(10, 6))
plt.plot(monthly_bias.index, monthly_bias.values, marker="o")
plt.title("Monthly Bias Trend"); plt.xlabel("Month"); plt.ylabel("MAE"); plt.grid(True)
plt.tight_layout(); plt.savefig(f"{FIGURES}/monthly_bias_trend.png", dpi=150); plt.close()

# Rain vs Dry bias
rainy = df_test[df_test["Total Precipitation"] > 0]
dry = df_test[df_test["Total Precipitation"] == 0]
rainy_mae = mean_absolute_error(rainy["y_true"], rainy["y_pred"]) if not rainy.empty else float("nan")
dry_mae = mean_absolute_error(dry["y_true"], dry["y_pred"]) if not dry.empty else float("nan")
pd.DataFrame([{"condition": "Rainy", "MAE": rainy_mae, "n_samples": len(rainy)},
              {"condition": "Dry", "MAE": dry_mae, "n_samples": len(dry)}]).to_csv(
    f"{RESULTS}/rain_dry_bias.csv", index=False)
print(f"Rainy MAE: {rainy_mae:.4f} | Dry MAE: {dry_mae:.4f}")

plt.figure(figsize=(5, 4))
plt.bar(["Rainy", "Dry"], [rainy_mae, dry_mae], color=["steelblue", "sandybrown"])
plt.ylabel("MAE (K)"); plt.title("Rain vs Dry Condition Bias")
plt.tight_layout(); plt.savefig(f"{FIGURES}/rain_vs_dry_bias.png", dpi=150); plt.close()

# ---------------------------------------------------------------- CO4 comparison table
rows = []
y_persist_scaled = np.roll(y_test.flatten(), 1)[TIME_STEPS:]
y_persist = y_scaler.inverse_transform(y_persist_scaled.reshape(-1, 1)).flatten()
rows.append({"Model": "Persistence (naive)", "Accuracy": None, "Precision": None, "Recall": None,
             "F1": None, "AUC": None, "RMSE": np.sqrt(mean_squared_error(y_true, y_persist)),
             "MAE": mean_absolute_error(y_true, y_persist), "R2": r2_score(y_true, y_persist), "Parameters": 0})

lr = LinearRegression().fit(X_train_flat, y_train_flat)
y_pred_lr = y_scaler.inverse_transform(lr.predict(X_test_flat).reshape(-1, 1)).flatten()
rows.append({"Model": "Linear Regression", "Accuracy": None, "Precision": None, "Recall": None,
             "F1": None, "AUC": None, "RMSE": np.sqrt(mean_squared_error(y_true, y_pred_lr)),
             "MAE": mean_absolute_error(y_true, y_pred_lr), "R2": r2_score(y_true, y_pred_lr),
             "Parameters": X_train_flat.shape[1] + 1})

rows.append({"Model": "sklearn MLP proxy (stand-in for Dense/LSTM baseline)",
             "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "AUC": auc,
             "RMSE": rmse, "MAE": mae, "R2": r2,
             "Parameters": sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)})

rows.append({"Model": "Proposed LSTM (CO3) - fill in after `python src/train.py --model baseline` + `test.py`",
             "Accuracy": None, "Precision": None, "Recall": None, "F1": None, "AUC": None,
             "RMSE": None, "MAE": None, "R2": None, "Parameters": 94881})
rows.append({"Model": "Proposed LSTM + Attention (CO5) - fill in after `python src/train.py --model attention` + `test.py`",
             "Accuracy": None, "Precision": None, "Recall": None, "F1": None, "AUC": None,
             "RMSE": None, "MAE": None, "R2": None, "Parameters": 186593})

pd.DataFrame(rows).to_csv(f"{RESULTS}/comparison.csv", index=False)

print("\nAll figures saved to figures/, all tables saved to results/")
