# 🥇 Gold Price Prediction — Standardized ML Pipeline

Predict the gold ETF price **GLD** (regression) from other daily market indicators, using a clean, leakage-free, 8-stage pipeline.

## 🔁 Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Basic Analysis | shape, info, describe, missing-value check |
| 2 | Train/Test Split | done **first** (80/20) — everything after is fit on train only |
| 3 | EDA | training data only: histograms, correlation heatmap |
| 4 | Feature Engineering | drops the `Date` column (plain regression, not time series) |
| 5 | Encoding | no categoricals → safe no-op (all inputs are numeric) |
| 6 | Outlier Handling | compares IQR vs Z-score vs Winsorization vs none → keeps best (train only) |
| 7 | Scaling | compares Standard vs MinMax vs Robust → keeps best (judged with KNN) |
| 8 | Models | Linear, Random Forest, Gradient Boosting, KNN → compared by R²/MAE/RMSE |

> **No leakage:** split first, then fit every data-driven step on the training set only.

## 📊 Dataset

`gold_price_data.csv` — daily rows (2008–2018). Columns: `Date`, `SPX` (S&P 500), `GLD` (🎯 target, gold ETF), `USO` (oil), `SLV` (silver), `EUR/USD`. Gold is most strongly correlated with **silver (SLV ≈ 0.87)**.

## ⏱️ Two versions

- **This notebook** uses a normal random split — good for learning the standard pipeline.
- A separate **time-series notebook** uses a chronological split (train on the past, test on the future) for a realistic forecasting estimate. Use that one if you care about predicting *future* gold prices.

## 🤖 Results

The notebook auto-selects the best outlier method, scaler, and model at runtime. **Run All** to generate the numbers.

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## 🚀 Getting Started

Open `Gold_Price_Prediction.ipynb` in Google Colab (it mounts Google Drive for the CSV) or locally (change the load-cell path to `gold_price_data.csv`), then **Run All**.

## 📁 Files

```
Gold Price Prediction/
├── Gold_Price_Prediction.ipynb
├── gold_price_data.csv
└── README.md
```
