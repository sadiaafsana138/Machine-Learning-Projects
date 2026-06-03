# 🔥 Calories Burnt Prediction — Standardized ML Pipeline

Predict the **calories burnt** during exercise (regression) using a clean, leakage-free, 8-stage pipeline.

## 🔁 Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Basic Analysis | merge `exercise.csv` + `calories.csv` on `User_ID`; shape, info, describe, missing-value check |
| 2 | Train/Test Split | done **first** (80/20) — everything after is fit on train only |
| 3 | EDA | training data only: histograms, Gender countplot, Calories-vs-Duration scatter, correlation |
| 4 | Feature Engineering | `BMI` (= Weight / height²) |
| 5 | Encoding | one-hot for `Gender` |
| 6 | Outlier Handling | compares IQR vs Z-score vs Winsorization vs none → keeps best (train only) |
| 7 | Scaling | compares Standard vs MinMax vs Robust → keeps best (judged with KNN) |
| 8 | Models | Linear, Random Forest, Gradient Boosting, KNN → compared by R²/MAE/RMSE |

> **No leakage:** split first, then fit every data-driven step on the training set only.

## 📊 Dataset

Two files (15,000 rows), merged on `User_ID`:
- `exercise.csv` — `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- `calories.csv` — `Calories` (🎯 target)

**Duration**, **Heart_Rate**, and **Body_Temp** correlate most strongly with calories burnt.

## 🤖 Results

The notebook auto-selects the best outlier method, scaler, and model at runtime, then prints R²/MAE/RMSE for every model. **Run All** to generate the numbers. (Expect a very high R² — this is an easy, clean dataset.)

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## 🚀 Getting Started

Open `Calories_Burnt_Prediction.ipynb` in Google Colab (it mounts Google Drive for the CSVs) or locally (change the `base` path to this folder), then **Run All**.

## 📁 Files

```
Burn Calories Prediction/
├── Calories_Burnt_Prediction.ipynb
├── exercise.csv
├── calories.csv
└── README.md
```
