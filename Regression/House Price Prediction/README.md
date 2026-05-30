# House Price Prediction

A regression project that predicts the median value of owner-occupied homes in Boston using the **XGBoost Regressor**. The model is trained on the classic Boston Housing dataset and evaluated with R² score and Mean Absolute Error.

## Overview

The notebook (`House_Price_Prediction.ipynb`) walks through a complete supervised-regression workflow:

1. **Load the data** – the Boston House Price dataset from `sklearn.datasets`.
2. **Explore & preprocess** – build a Pandas DataFrame, inspect shape and statistics, check for missing values.
3. **Analyze correlations** – visualize feature relationships with a heatmap.
4. **Split the data** – separate features (`X`) from the target (`price`), then split into train (80%) and test (20%) sets.
5. **Train the model** – fit an `XGBRegressor`.
6. **Evaluate** – measure R² and MAE on both training and test data, and visualize actual vs. predicted prices.

## Dataset

The **Boston Housing** dataset contains 506 samples with 13 numeric features and a target value (`MEDV`, the median home value in $1000s).

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river, else 0) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)² where Bk is the proportion of black residents by town |
| LSTAT | % lower status of the population |
| **price** (target) | Median value of owner-occupied homes in $1000s |

> **Note:** The Boston Housing dataset was removed from scikit-learn in version 1.2 due to ethical concerns (the `B` feature). `sklearn.datasets.load_boston()` works only on **scikit-learn < 1.2**. To run this notebook on a newer environment, install an older version (e.g. `pip install scikit-learn==1.1.3`) or load the data from an alternative source.

## Tech Stack

- Python
- NumPy & Pandas — data handling
- Matplotlib & Seaborn — visualization
- scikit-learn — dataset, train/test split, evaluation metrics
- XGBoost — regression model

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn==1.1.3 xgboost
```

## Usage

```bash
jupyter notebook House_Price_Prediction.ipynb
```

Run the cells top to bottom to reproduce the data loading, training, and evaluation.

## Results

The XGBoost Regressor achieved the following performance:

| Dataset | R² Score | Mean Absolute Error |
|---------|----------|---------------------|
| Training | 0.973 | 1.15 |
| Test | 0.912 | 1.99 |

The high R² on the test set indicates the model generalizes well, explaining roughly 91% of the variance in home prices on unseen data.

## Project Structure

```
House Price Prediction/
├── House_Price_Prediction.ipynb   # Main notebook
└── README.md
```
