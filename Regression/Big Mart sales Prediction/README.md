# 🛒 Big Mart Sales Prediction — Standardized ML Pipeline

Predict a product's **Item_Outlet_Sales** at a store (regression) using a clean, leakage-free, 8-stage pipeline.

## 🔁 Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Basic Analysis | load `Train.csv`; shape, info, describe, missing-value check |
| 2 | Train/Test Split | done **first** (80/20) — everything after is fit on train only |
| 3 | EDA | training data only: histograms, categorical countplots, correlation |
| 4 | Feature Engineering | **train-only imputation** + `Item_Category`, `Outlet_Years` |
| 5 | Encoding | one-hot for all nominal categoricals |
| 6 | Outlier Handling | compares IQR vs Z-score vs Winsorization vs none → keeps best (train only) |
| 7 | Scaling | compares Standard vs MinMax vs Robust → keeps best (judged with KNN) |
| 8 | Models | Linear, Random Forest, Gradient Boosting, KNN → compared by R²/MAE/RMSE |

> **No leakage:** the split happens first. Crucially, the **missing-value imputation values are learned from the training set only** (mean of `Item_Weight`, mode of `Outlet_Size`) and then applied to both train and test.

## ⚠️ Which file to use

This dataset comes as two files:
- **`Train.csv`** — 8,523 rows, **includes the target** `Item_Outlet_Sales`. ✅ This is what the notebook uses.
- `Test.csv` — 5,681 rows, **no target**. Only for generating Kaggle submissions; **cannot** be used to train or score a model.

## 📊 Dataset

`Train.csv` — 12 columns. Notable ones:

| Column | Notes |
|--------|-------|
| Item_Weight | numeric, **has missing values** → imputed with train mean |
| Item_Visibility | numeric |
| Item_MRP | numeric (strong predictor) |
| Item_Fat_Content | categorical (labels cleaned: LF/low fat → Low Fat, reg → Regular) |
| Item_Type | categorical (16 types) |
| Outlet_Identifier | categorical (10 stores) |
| Outlet_Size | categorical, **has missing values** → imputed with train mode |
| Outlet_Location_Type, Outlet_Type | categorical |
| Outlet_Establishment_Year | turned into `Outlet_Years` |
| Item_Identifier | turned into `Item_Category` (FD/DR/NC), then dropped |
| **Item_Outlet_Sales** | 🎯 target |

## 🤖 Results

The notebook auto-selects the best outlier method, scaler, and model at runtime. **Run All** to generate the numbers. (Big Mart is a genuinely hard dataset — expect a moderate R², not 0.9+.)

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## 🚀 Getting Started

Open `Big_Mart_Sales_Prediction.ipynb` in Google Colab (it mounts Google Drive for the CSV) or locally (change the load-cell path to `Train.csv`), then **Run All**.

## 📁 Files

```
Big Mart sales Prediction/
├── Big_Mart_Sales_Prediction.ipynb
├── Train.csv          # used for training (has the target)
├── Test.csv           # Kaggle test set (no target) - not used
└── README.md
```
