# 💊 Insurance Cost Prediction

Predicting a person's **medical insurance charges** from their demographic and health details. This is an end-to-end supervised machine learning **regression** project built in a single Jupyter notebook — from data exploration to training and comparing multiple models, ending with a working prediction system.

---

## 📌 Problem Statement

Given a person's age, sex, BMI, number of children, smoking status, and region, predict their annual **insurance charges** (in USD).

## 🔁 Workflow

The notebook `Insurance_Cost_Prediction.ipynb` follows these steps:

1. **Import dependencies** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
2. **Load the data** — read `insurance.csv` into a Pandas DataFrame
3. **Explore the data** — inspect the first rows, shape `(1338, 7)`, data types, summary statistics, and confirm there are no missing values
4. **Visual analysis** — distribution plots for age, BMI, and charges; count plots for sex, children, smoker, and region
5. **One-hot encode** the categorical columns (`sex`, `smoker`, `region`) with `pd.get_dummies`
6. **Split features & target** — drop `charges` to form `X`; use `charges` as `Y`
7. **Train/test split** — 80% train / 20% test, `random_state=2`
8. **Train & compare models** — Linear Regression, Random Forest, and Gradient Boosting, scored with R², MAE, and RMSE; the best-scoring model is selected automatically
9. **Predictive system** — feed in one person's details (in plain words) and predict their insurance cost

## 📊 Dataset

The dataset (`insurance.csv`) contains **1,338 records** with 7 columns.

| Column | Description |
|--------|-------------|
| age | Age of the person |
| sex | Gender (male / female) |
| bmi | Body Mass Index (normal range ≈ 18.5–24.9) |
| children | Number of children / dependents |
| smoker | Whether the person smokes (yes / no) |
| region | Residential area (northeast / northwest / southeast / southwest) |
| charges | Annual medical insurance cost in USD (🎯 target) |

No missing values, so no cleaning is required.

### Encoding

Categorical columns are converted with **one-hot encoding** (`pd.get_dummies(..., drop_first=True)`) rather than mapping categories to numbers like 0/1/2/3. This avoids implying a false order (e.g. that one region is "greater" than another). After encoding, the features are:

`age`, `bmi`, `children`, `sex_male`, `smoker_yes`, `region_northwest`, `region_southeast`, `region_southwest`

> **Key insight:** **Smoking** is by far the strongest driver of insurance cost — a smoker's charges are dramatically higher, especially when combined with a high BMI.

## 🤖 Models & Results

Three models are trained on the same split and compared. Linear Regression is a baseline; the tree-based ensembles capture the non-linear interactions (like smoking × BMI) that linear models miss.

| Model | Notes | Expected Test R² |
|-------|-------|:----------------:|
| Linear Regression | Linear baseline | ~0.74 |
| Random Forest | 200 trees, non-linear | ~0.84–0.87 |
| Gradient Boosting | Boosted trees (usually best here) | ~0.86–0.88 |

The notebook automatically picks the highest-scoring model and uses it in the prediction system. Each model also reports **MAE** (average dollar error) and **RMSE** (penalizes large misses).

> ℹ️ **Re-run for exact numbers:** the notebook was recently updated (one-hot encoding, extra models, added metrics). Run it top to bottom (**Run All**) to regenerate all the scores.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`

## 🚀 Getting Started

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Launch the notebook
jupyter notebook Insurance_Cost_Prediction.ipynb
```

Make sure `insurance.csv` is in the same folder as the notebook, then run the cells from top to bottom.

### Making a prediction

The prediction system takes input in plain, human-readable form:

```python
input_dict = {
    'age': 31,
    'bmi': 25.74,
    'children': 0,
    'sex': 'male',
    'smoker': 'yes',
    'region': 'southwest',
}
```

It is one-hot encoded and aligned to the training columns automatically, then passed to the best model.

## 📁 Project Structure

```
Insurance Cost Prediction/
├── Insurance_Cost_Prediction.ipynb   # Main notebook
├── insurance.csv                     # Dataset
└── README.md                         # This file
```

## 🔮 Possible Further Improvements

- Tune hyperparameters (e.g. `GridSearchCV` on Gradient Boosting)
- Log-transform the skewed `charges` target for the linear model
- Inspect feature importances to confirm smoking/BMI dominate
- Add cross-validation for a more robust score estimate
