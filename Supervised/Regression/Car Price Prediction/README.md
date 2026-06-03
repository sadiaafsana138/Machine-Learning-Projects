# 🚗 Car Price Prediction — Standardized ML Pipeline

Predict a used car's **selling price** (regression) using a clean, leakage-free, 8-stage pipeline.

## 🔁 Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Basic Analysis | shape, info, describe, missing-value check |
| 2 | Train/Test Split | done **first** (80/20) — everything after is fit on train only |
| 3 | EDA | training data only: histograms, categorical countplots, correlation |
| 4 | Feature Engineering | `age` (= 2020 − year), `brand` (from car name); drops `name`, `year` |
| 5 | Encoding | one-hot for `fuel`, `seller_type`, `transmission`, `owner`, `brand` |
| 6 | Outlier Handling | compares IQR vs Z-score vs Winsorization vs none → keeps best (train only) |
| 7 | Scaling | compares Standard vs MinMax vs Robust → keeps best (judged with KNN) |
| 8 | Models | Linear, Random Forest, Gradient Boosting, KNN → compared by R²/MAE/RMSE |

> **No leakage:** split first, then fit every data-driven step on the training set only.

## 📊 Dataset

`car data.csv` — used-car listings. Columns: `name`, `year`, `selling_price` (🎯 target), `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`. The engineered `brand` (extracted from `name`) is a strong price signal that the raw name throws away.

## 🤖 Results

The notebook auto-selects the best outlier method, scaler, and model at runtime, then prints R²/MAE/RMSE for every model. **Run All** to generate the numbers.

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## 🚀 Getting Started

Open `Car_Price_Prediction.ipynb` in Google Colab (it mounts Google Drive for the CSV) or locally (change the load-cell path to `car data.csv`), then **Run All**.

## 📁 Files

```
Car Price Prediction/
├── Car_Price_Prediction.ipynb
├── car data.csv
└── README.md
```
