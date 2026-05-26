# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster based on demographic and ticket information, using Logistic Regression.

---

## Overview

This project builds a binary classification model trained on the famous Titanic dataset. Given passenger details, the model predicts:

- **1** → Survived
- **0** → Did Not Survive

---

## Dataset

**File:** `train.csv` (Titanic dataset)  
**Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

### Features Used

| Column | Description | Type |
|---|---|---|
| `Pclass` | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) | int |
| `Sex` | Sex (0 = male, 1 = female) | int |
| `Age` | Age of the passenger (years) | float |
| `SibSp` | Number of siblings/spouses aboard | int |
| `Parch` | Number of parents/children aboard | int |
| `Fare` | Ticket fare | float |
| `Embarked` | Port of embarkation (0 = S, 1 = C, 2 = Q) | int |

> Columns `PassengerId`, `Name`, `Ticket`, and `Cabin` were dropped as they don't contribute to prediction.

---

## Project Structure

```
Titanic-Survival-Prediction/
│
├── train.csv                                                    # Dataset
├── Titanic_Survival_Prediction_using_Machine_Learning.ipynb     # Jupyter notebook
└── README.md
```

---

## Methodology

### 1. Data Collection & Exploration
- Loaded the CSV into a Pandas DataFrame
- Inspected shape, data types, and missing values

### 2. Handling Missing Values
- Dropped the `Cabin` column (too many missing values)
- Filled missing `Age` values with the **mean**
- Filled missing `Embarked` values with the **mode**

### 3. Data Visualization
- Survival count plot
- Gender distribution and survival by gender
- Passenger class distribution and survival by class

### 4. Encoding Categorical Columns
- `Sex`: male → 0, female → 1
- `Embarked`: S → 0, C → 1, Q → 2

### 5. Model Training
- Algorithm: **Logistic Regression** (`max_iter=1000`)
- Split: **80% training / 20% test** (`random_state=2`)

### 6. Evaluation
- Accuracy score on training and test data
- Overfitting check: flags the model if accuracy gap exceeds 10%

### 7. Predictive System
- Accepts new passenger data as a tuple
- Predicts survival outcome instantly

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/titanic)
2. Open the notebook in Jupyter or Google Colab
3. Update the CSV file path to match your environment
4. Run all cells sequentially

> **Google Colab users:** The notebook uses `google.colab.drive` to mount Google Drive. Update the path accordingly.

### Making a Prediction

Update the `input_data` tuple in the last cell with passenger details in this order:

```python
input_data = (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
```

**Example:**
```python
input_data = (3, 0, 22.0, 1, 0, 7.25, 0)
# Output: The Person did not Survive
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Array manipulation and reshaping |
| `pandas` | Data loading and preprocessing |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical data visualization |
| `scikit-learn` | Model training, splitting, and evaluation |

---

## Disclaimer

This project is intended for **educational purposes only** as part of a machine learning learning journey.
