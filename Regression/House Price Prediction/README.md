# 🏠 House Price Prediction — Standardized ML Pipeline

Predict the median California house value **MedHouseVal** (regression) using a clean, leakage-free, 8-stage pipeline.

## 🔁 Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Basic Analysis | shape, info, describe, missing-value check |
| 2 | Train/Test Split | done **first** (80/20) — everything after is fit on train only |
| 3 | EDA | training data only: histograms, correlation heatmap |
| 4 | Feature Engineering | `bedroom_ratio` (= AveBedrms / AveRooms) |
| 5 | Encoding | no categoricals → safe no-op (all inputs are numeric) |
| 6 | Outlier Handling | compares IQR vs Z-score vs Winsorization vs none → keeps best (train only) |
| 7 | Scaling | compares Standard vs MinMax vs Robust → keeps best (judged with KNN) |
| 8 | Models | Linear, Random Forest, Gradient Boosting, KNN → compared by R²/MAE/RMSE |

> **No leakage:** split first, then fit every data-driven step on the training set only.

## 📊 Dataset

The **California Housing** dataset, built into scikit-learn (`fetch_california_housing`). 20,640 rows, 8 numeric features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`; target `MedHouseVal` (in $100,000s). `MedInc` (median income) is the strongest predictor.

> No CSV or Google Drive needed — the data downloads automatically via scikit-learn.

## 🤖 Results

The notebook auto-selects the best outlier method, scaler, and model at runtime. **Run All** to generate the numbers.

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## 🚀 Getting Started

Open `House_Price_Prediction.ipynb` and **Run All** — no data download or path changes required.

## 📁 Files

```
House Price Prediction/
├── House_Price_Prediction.ipynb
└── README.md
```
