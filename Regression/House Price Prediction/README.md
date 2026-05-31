# 🏠 House Price Prediction

Predicting median house values in California using an **XGBoost Regressor**. This is an end-to-end supervised machine learning regression project built in a single Jupyter notebook — from loading the data to training the model and evaluating its accuracy.

---

## 📌 Problem Statement

Given a set of features describing a California census block group (median income, house age, average rooms, location, etc.), predict the **median house value** for that block group (in units of $100,000).

## 🔁 Workflow

The notebook `House_Price_Prediction.ipynb` follows these steps:

1. **Import dependencies** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, XGBoost
2. **Load the dataset** — California Housing dataset via `sklearn.datasets.fetch_california_housing()`
3. **Build the DataFrame** — convert to a Pandas DataFrame and append the target `price` column
4. **Explore the data** — check shape `(20640, 9)`, confirm there are no missing values, and review summary statistics
5. **Correlation analysis** — plot a heatmap to study positive/negative correlations between features
6. **Split features & target** — `X` (8 features) and `Y` (price)
7. **Train/test split** — 80% train (16,512 rows) / 20% test (4,128 rows), `random_state=2`
8. **Train the model** — fit an `XGBRegressor`
9. **Evaluate** — compute R² and Mean Absolute Error on both training and test sets
10. **Visualize** — scatter plot of actual vs. predicted prices

## 📊 Dataset

**California Housing dataset** — derived from the 1990 U.S. Census, one row per census block group. 20,640 samples, 8 numeric features, 1 target.

| # | Feature | Description |
|---|---------|-------------|
| 1 | MedInc | Median income in block group |
| 2 | HouseAge | Median house age in block group |
| 3 | AveRooms | Average number of rooms per household |
| 4 | AveBedrms | Average number of bedrooms per household |
| 5 | Population | Block group population |
| 6 | AveOccup | Average number of household members |
| 7 | Latitude | Block group latitude |
| 8 | Longitude | Block group longitude |
| 🎯 | **price** | Median house value in units of $100,000 (target) |

The dataset is built into scikit-learn and downloads automatically on first use — no manual download required.

## 🤖 Model & Results

**Model:** XGBoost Regressor (default hyperparameters)

| Dataset | R² Score | Mean Absolute Error |
|---------|:--------:|:-------------------:|
| 🏋️ Training | **0.944** | 0.193 |
| 🧪 Test | **0.834** | 0.311 |

The model explains about **83% of the variance** on unseen test data, with predictions off by roughly **$31,000** on average (MAE of 0.311 × $100,000). The gap between training and test scores points to some overfitting — a candidate for hyperparameter tuning in future work.

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost`

## 🚀 Getting Started

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# Launch the notebook
jupyter notebook House_Price_Prediction.ipynb
```

Then run the cells from top to bottom.

## 📁 Project Structure

```
House Price Prediction/
├── House_Price_Prediction.ipynb   # Main notebook
└── README.md                      # This file
```
