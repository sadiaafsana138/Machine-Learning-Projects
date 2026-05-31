# 🚗 Car Price Prediction

Predicting the resale (selling) price of used cars. The notebook trains and compares three regression models — **Linear Regression**, **Lasso Regression**, and a **Random Forest Regressor** — using feature engineering, one-hot encoding, and a log-transformed target to maximize accuracy.

---

## 📌 Problem Statement

Given details about a used car (manufacturing year, kilometers driven, fuel type, seller type, transmission, number of previous owners, and brand), predict its **selling price**.

## 🔁 Workflow

The notebook `Car_Price_Prediction.ipynb` follows these steps:

1. **Import dependencies** — Pandas, NumPy, Matplotlib, Seaborn, scikit-learn
2. **Load the data** — read `car data.csv.csv` into a Pandas DataFrame (mounted from Google Drive in Colab)
3. **Inspect the data** — check shape `(4340, 8)`, data types, confirm there are no missing values, and review the distribution of the categorical columns
4. **Feature engineering**
   - Derive `age = 2020 - year` (more meaningful to the model than a raw year)
   - Extract `brand` from the car `name` (e.g. "Maruti", "Hyundai") — a strong price signal that was previously discarded
5. **Encode & transform**
   - One-hot encode the categorical columns (`fuel`, `seller_type`, `transmission`, `owner`, `brand`) so no false numeric ordering is implied
   - Log-transform the skewed target with `np.log1p` (predictions are converted back with `np.expm1`)
6. **Split features & target** — drop `name`, `year`, and `selling_price` to form `X`; use `log1p(selling_price)` as `Y`
7. **Train/test split** — 80% train / 20% test, `random_state=2`
8. **Train & evaluate three models** — Linear Regression, Lasso Regression, and Random Forest, each scored with R²
9. **Visualize** — actual vs. predicted price scatter plots

## 📊 Dataset

The dataset (`car data.csv.csv`) contains **4,340 used-car records** with 8 columns.

| Column | Description |
|--------|-------------|
| name | Name of the car (used to derive `brand`, then dropped) |
| year | Year of manufacture (used to derive `age`, then dropped) |
| selling_price | Price the car is being sold for (🎯 target) |
| km_driven | Total kilometers driven |
| fuel | Fuel type (Petrol / Diesel / CNG / LPG / Electric) |
| seller_type | Dealer / Individual / Trustmark Dealer |
| transmission | Manual / Automatic |
| owner | Ownership history (First / Second / Third / Fourth & Above / Test Drive Car) |

### Engineered Features

| Feature | How it's built |
|---------|----------------|
| age | `2020 - year` |
| brand | First word of `name` (the manufacturer) |

Categorical columns are then expanded into binary indicator columns via `pd.get_dummies`.

> 📂 **Note:** The notebook loads the CSV from a Google Drive path (Colab) and the file is named `car data.csv.csv`. Update the path in the data-loading cell to point to your local copy before running outside Colab. The `age` feature hardcodes the year `2020` — adjust it if your data is from a different period.

## 🤖 Models & Results

Three models are trained and compared. Linear and Lasso provide a baseline; Random Forest captures the non-linear relationships in the data.

| Model | Notes |
|-------|-------|
| Linear Regression | Linear baseline |
| Lasso Regression | Linear baseline with L1 regularization |
| Random Forest Regressor | Non-linear model, `n_estimators=200` (expected best performer) |

> ⚠️ **Results pending re-run.** The notebook was recently updated (feature engineering, one-hot encoding, log-transformed target, and the new Random Forest model). The R² values currently saved in the notebook cells are from the *previous* approach and are no longer valid. **Run the notebook top to bottom (Run All)** to regenerate the metrics, then fill in the table below.

| Model | Training R² | Test R² |
|-------|:-----------:|:-------:|
| Linear Regression | _TBD_ | _TBD_ |
| Lasso Regression | _TBD_ | _TBD_ |
| Random Forest | _TBD_ | _TBD_ |

*Note: R² for the linear models is now measured on the log-transformed target, so it is not directly comparable to the earlier (raw-price) scores. Random Forest also reports a Mean Absolute Error in rupees for interpretability.*

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`

## 🚀 Getting Started

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Launch the notebook
jupyter notebook Car_Price_Prediction.ipynb
```

Make sure `car data.csv.csv` is available and update the path in the data-loading cell to point to it, then run the cells from top to bottom.

> The data is split 80% train / 20% test (`test_size=0.2`, `random_state=2`).

## 📁 Project Structure

```
Car Price Prediction/
├── Car_Price_Prediction.ipynb   # Main notebook
├── car data.csv.csv             # Dataset
└── README.md                    # This file
```

## 🔮 Possible Further Improvements

- Hyperparameter tuning (e.g. `GridSearchCV` on the Random Forest)
- Try gradient-boosted models (XGBoost, LightGBM)
- Inspect feature importances to understand the main price drivers
- Group rare brands into an "Other" category to reduce one-hot sparsity
