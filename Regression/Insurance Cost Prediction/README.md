# 💊 Insurance Cost Prediction — Standardized ML Pipeline

Predict medical insurance **charges** (regression) using a clean, leakage-free, 8-stage pipeline.

## 🔁 Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Basic Analysis | shape, info, describe, missing-value check |
| 2 | Train/Test Split | done **first** (80/20) — everything after is fit on train only |
| 3 | EDA | training data only: histograms, countplots, correlation, charges-by-smoker |
| 4 | Feature Engineering | `is_obese`, `high_risk_smoker` (applied to train & test) |
| 5 | Encoding | one-hot for `sex`, `smoker`, `region` |
| 6 | Outlier Handling | compares IQR vs Z-score vs Winsorization vs none → keeps best (train only) |
| 7 | Scaling | compares Standard vs MinMax vs Robust → keeps best (judged with KNN) |
| 8 | Models | Linear, Random Forest, Gradient Boosting, KNN → compared by R²/MAE/RMSE |

> **No leakage:** the split happens before any data-driven step. Encoding, outlier bounds, and the scaler are all fit on the training set only; the test set is transformed, never learned from. Outlier *removal* is train-only by design (you never drop test rows).

## 📊 Dataset

`insurance.csv` — 1,338 records, 7 columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the target `charges` (USD). No missing values. **Smoking** is the dominant cost driver.

## 🤖 Results (from last run)

| Stage | Outcome |
|-------|---------|
| Best outlier method | **Z-score** |
| Best scaler | **Standard** |
| **Best model** | **Gradient Boosting** |
| Test R² | **0.868** |
| Test MAE | 2356.42 |
| Test RMSE | 4447.86 |

(Full ranking: Gradient Boosting ≈ Linear Regression > KNN ≈ Random Forest.)

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## 🚀 Getting Started

Open `Insurance_Cost_Prediction.ipynb` in Google Colab (it mounts Google Drive for the CSV) or locally (update the path in the load cell to `insurance.csv`), then **Run All**.

## 📁 Files

```
Insurance Cost Prediction/
├── Insurance_Cost_Prediction.ipynb
├── insurance.csv
└── README.md
```
