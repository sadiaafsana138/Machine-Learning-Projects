# 🥇 Gold Price Prediction

Predicting the price of gold (the **GLD** gold ETF) from other daily financial indicators using a **Random Forest Regressor**. This is an end-to-end supervised machine learning regression project built in a single Jupyter notebook — from data exploration to model training and evaluation.

---

## 📌 Problem Statement

Given a set of same-day financial indicators (the S&P 500 index, oil price, silver price, and the EUR/USD exchange rate), predict the **gold ETF price (GLD)**.

## 🔁 Workflow

The notebook `Gold_Price_Prediction.ipynb` follows these steps:

1. **Import dependencies** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
2. **Load the data** — read `gold_price_data.csv` into a Pandas DataFrame
3. **Explore the data** — inspect the first/last rows, shape `(2290, 6)`, data types, summary statistics, and confirm there are no missing values
4. **Correlation analysis** — plot a heatmap and check which features correlate with `GLD`
5. **Distribution check** — plot the distribution of the `GLD` price
6. **Split features & target** — drop `Date` and `GLD` to form `X`; use `GLD` as `Y`
7. **Train/test split** — 80% train / 20% test, `random_state=2`
8. **Train the model** — fit a `RandomForestRegressor` (100 trees)
9. **Evaluate** — measure R², MAE, and RMSE on the test set, plus the training R² to check for overfitting
10. **Visualize** — line plot comparing actual vs. predicted gold prices

## 📊 Dataset

The dataset (`gold_price_data.csv`) contains **2,290 daily records** spanning 2008–2018, with 6 columns.

| Column | Description |
|--------|-------------|
| Date | Trading date (dropped before training) |
| SPX | S&P 500 stock market index |
| GLD | Gold ETF price (🎯 target) |
| USO | Oil price ETF |
| SLV | Silver ETF price |
| EUR/USD | Euro-to-US-Dollar exchange rate |

**Key insight:** `GLD` is most strongly correlated with **`SLV` (silver)**, with a correlation of about **0.87** — silver's price is the single biggest predictor of gold.

## 🤖 Model & Results

**Model:** Random Forest Regressor (`n_estimators=100`, `random_state=2`)

The Random Forest captures the non-linear relationships between the indicators and the gold price and achieves a very high **R² of ≈ 0.99** on the test set. The notebook also reports MAE and RMSE (in price units) and the training R² so you can confirm the model is not overfitting.

| Metric | Description |
|--------|-------------|
| Test R² | ≈ 0.99 — proportion of price variance explained |
| Test MAE | Average absolute prediction error (price units) |
| Test RMSE | Error metric that penalizes larger misses |
| Train R² | Training score, for an overfitting check |

> ℹ️ **Re-run for exact numbers:** the evaluation cells were recently expanded (added MAE, RMSE, and the training-score check), so run the notebook top to bottom (**Run All**) to regenerate all the metrics.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`

## 🚀 Getting Started

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Launch the notebook
jupyter notebook Gold_Price_Prediction.ipynb
```

Make sure `gold_price_data.csv` is in the same folder as the notebook, then run the cells from top to bottom.

## 📁 Project Structure

```
Gold Price Prediction/
├── Gold_Price_Prediction.ipynb   # Main notebook
├── gold_price_data.csv           # Dataset
└── README.md                     # This file
```

## 🔮 Possible Further Improvements

- Tune the Random Forest hyperparameters (e.g. `GridSearchCV`)
- Try gradient-boosted models (XGBoost, LightGBM)
- Inspect feature importances to see which indicators drive the prediction most
