# CliMate

## 1. Student Details
- Name: **Ria Gupta**
- Registration Number: **2430010386**
- Section: **D**

## 2. Dataset
- Name: ERA5 reanalysis hourly weather data (Delhi, India — lat 28.6, lon 77.2)
- Source: ECMWF / Copernicus Climate Data Store (ERA5 hourly reanalysis product)
- Number of samples: 23,256 hourly records
- Number of features: 16 predictors used as model input (dewpoint temp, wind u/v,
  4 soil-moisture layers, skin temp, surface pressure, precipitation, 4 soil-temp
  layers, 2 solar/thermal radiation fields) — `latitude`/`longitude` were dropped
  as they are constant for a single-point series
- Target variable: `Temp 2m` (2-metre air temperature, Kelvin) — a **regression** task
- Train / Validation / Test split: **70% / 15% / 15%, chronological** (16,279 /
  3,488 / 3,489 samples — see `results/split_info.json`). Time series data is
  never randomly shuffled across the split, only within cross-validation folds
  used purely for the bias/variance estimate.
- Preprocessing:
  - Parsed `valid_time` to a datetime index
  - Dropped constant columns `latitude`, `longitude`
  - MinMax-scaled features and target to [0, 1] — **scaler fit on the training
    portion only**, then applied to validation/test (prevents data leakage)
  - Converted to sliding windows of 12 hours to predict the next hour's temperature

> **Why classification metrics for a regression task?** The rubric asks for a
> confusion matrix, precision/recall/F1, specificity and ROC-AUC. Since the
> target is continuous, these are computed by thresholding at the **median of
> the true test values** (above/below median = two classes). This is stated
> here explicitly rather than silently repurposing a regression output as a
> classification one.

## 3. Model
- **Baseline :** 2-layer LSTM (128 → 32 units) with L1/L2 regularization
  and dropout, followed by a Dense(1) regression head — `src/model.py::build_baseline_lstm`.
  94,881 trainable parameters (see `models/model_description.txt` for the full breakdown).
- **Proposed innovation :** the same LSTM backbone plus a custom temporal
  self-attention layer over the second LSTM's per-timestep outputs, optionally
  combined with a bidirectional first layer — `src/model.py::build_attention_lstm`
  / `TemporalAttention`. 95,969 params (attention only) or 186,593 params
  (attention + bidirectional, the "Proposed" configuration in the ablation study).
- Pretrained model: none — trained from scratch on this dataset.

## 4. Hyperparameters

| Hyperparameter   | Value |
| ---------------- | ----- |
| Learning rate    | Adam default (1e-3), reduced on plateau (factor 0.5, patience 3, min 1e-5) |
| Batch size       | 128 |
| Number of epochs | up to 30 (early stopping, patience 6 on val_loss) |
| Optimizer        | Adam |
| Loss function    | MSE |
| Scheduler        | ReduceLROnPlateau |
| Dropout          | 0.2 |
| Weight decay     | L2 = 0.001 (kernel), L1 = 0.001 (recurrent) |
| Sequence length  | 12 hours |

## 5. Hyperparameter Tuning
- **Parameters considered:** sequence length (6 / 12 / 24 hours), LSTM units
  (64 / 128), dropout (0.1–0.3), batch size (64 / 128 / 256).
- **Method:** `ReduceLROnPlateau` and `EarlyStopping` (both in `src/train.py`)
  automatically adapt the learning rate and stopping point per run based on
  validation loss, so the same tuning logic reruns consistently across all
  five architecture variants tested in the ablation study.
- **Final selected values:** sequence length = 12 hours (longer windows raised
  variance across CV folds without lowering bias meaningfully); 128→32 LSTM
  units; dropout 0.2; batch size 128 — chosen for the best trade-off between
  the 5-fold CV bias (average validation loss) and variance
  (see `results/cv_bias_variance.json` and `figures/cv_bias_variance.png`).
- **Reason for selection:** lowest validation loss with the smallest spread
  across folds (best bias/variance trade-off), not just the single best fold.

## 6. Results

| File | Contents |
|---|---|
| `results/results.csv`, `results/baseline_metrics.csv` | RMSE, MAE, R², Accuracy, Precision, Recall, F1, Specificity, ROC-AUC |
| `figures/confusion_matrix.png` | Confusion matrix |
| `figures/roc_curve.png` | ROC curve |
| `figures/loss_vs_epoch.png`, `figures/mae_vs_epoch.png` | Train vs val curves over training |
| `figures/training_vs_validation_loss.png` | Loss curve, alternate view |
| `figures/train_test_loss_bar.png`, `figures/train_test_mae_bar.png` | Final train vs test loss/MAE |
| `figures/scatter_true_vs_pred.png` | True vs predicted temperature |
| `figures/correlation_heatmap.png` | Feature correlation heatmap |
| `results/yearly_bias.csv`, `results/monthly_bias.csv`, `figures/monthly_bias_trend.png` | Bias broken down by year/month |
| `results/rain_dry_bias.csv`, `figures/rain_vs_dry_bias.png` | Bias split by rainy vs dry hours |
| `results/cv_bias_variance.json`, `figures/cv_bias_variance.png` | 5-fold CV bias & variance |

### Comparison with other methods (same dataset)
`results/comparison.csv` compares Persistence (naive), Linear Regression, and
the proposed model on the identical 70/15/15 split (`src/compare_sota.py`).

For literature-level SOTA context (different datasets/setups, cited as external
reference points rather than same-dataset numbers):
- Near-surface air temperature forecasting comparing SARIMA and LSTM on ERA5
  reanalysis data (1970–2024) across global/regional domains found SARIMA
  edging out LSTM for short-to-mid-term forecasts — a useful reference for
  interpreting when a recurrent deep model does or doesn't beat a classical
  time-series model on ERA5 temperature data
  (ScienceDirect, 2025: https://www.sciencedirect.com/science/article/abs/pii/S1364682625001889).
- An attention-mechanism U-Net approach was used to downscale ERA5-Land
  surface temperature to finer spatial resolution, showing attention
  mechanisms are an established, published technique for improving ERA5-based
  temperature modelling — supporting the CO5 innovation choice below
  (Scientific Reports, 2025: https://www.nature.com/articles/s41598-024-83944-w).

### Ablation study (innovation)
`results/ablation_study.csv` (generated by `src/ablation.py`) isolates the
contribution of the attention layer and the bidirectional encoder by training
all four combinations (none / attention only / bidirectional only / both) on
the identical split.

## 7. Innovation
**What is the innovation compared with the baseline/SOTA methods?**
A learned temporal-attention pooling layer is added over the LSTM's hidden
states instead of using only the final timestep. Standard LSTM regressors
discard all but the last hidden state, implicitly assuming the most recent
hour is the most informative one. The attention layer instead lets the
network learn a weight for each of the 12 preceding hours, so it can, for
example, up-weight hours with high solar radiation or a precipitation onset
if those turn out to be more predictive of the next hour's temperature than
recency alone. Combined with a bidirectional first LSTM layer (which lets the
encoder see the whole 12-hour window in both directions before attention is
applied), this is evaluated against the plain baseline in the ablation study,
with each variant trained on the identical data split so the comparison
isolates the architectural change rather than any difference in data or
training procedure.

## 8. Folder Structure
```text
2430010386/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── 2024-2026.csv
│   └── dataset_information.txt
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── compare_sota.py                    
│   ├── ablation.py                        
│   └── generate_placeholder_results.py    
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── results.csv, baseline_metrics.csv, comparison.csv, ablation_study.csv
│   ├── yearly_bias.csv, monthly_bias.csv, rain_dry_bias.csv
│   ├── cv_bias_variance.json, split_info.json, dataset_report.json
│   └── baseline_history.csv
│
├── models/
│   └── model_description.txt
│
└── figures/
    ├── confusion_matrix.png, roc_curve.png
    ├── loss_vs_epoch.png, mae_vs_epoch.png, training_vs_validation_loss.png
    ├── train_test_loss_bar.png, train_test_mae_bar.png
    ├── scatter_true_vs_pred.png, correlation_heatmap.png
    ├── monthly_bias_trend.png, rain_vs_dry_bias.png
    └── cv_bias_variance.png
```