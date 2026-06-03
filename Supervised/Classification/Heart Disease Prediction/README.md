# Heart Disease Prediction

A machine learning project that predicts whether a patient has heart disease based on clinical features, using Logistic Regression trained on the Cleveland Heart Disease dataset.

---

## Overview

This project builds a binary classification model to detect the presence of heart disease from 13 medical attributes. Given patient data, the model outputs either:

- **1** → Defective Heart (Heart Disease Detected)
- **0** → Healthy Heart (No Heart Disease)

---

## Dataset

**File:** `heart_disease_data.csv`  
**Source:** Cleveland Heart Disease Dataset  
**Shape:** 303 rows × 14 columns

### Features

| Column | Description | Type |
|---|---|---|
| `age` | Age of the patient (years) | int |
| `sex` | Sex (1 = male, 0 = female) | int |
| `cp` | Chest pain type (0–3) | int |
| `trestbps` | Resting blood pressure (mm Hg) | int |
| `chol` | Serum cholesterol (mg/dl) | int |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | int |
| `restecg` | Resting ECG results (0–2) | int |
| `thalach` | Maximum heart rate achieved | int |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) | int |
| `oldpeak` | ST depression induced by exercise relative to rest | float |
| `slope` | Slope of the peak exercise ST segment (0–2) | int |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) | int |
| `thal` | Thalassemia type (0 = normal, 1 = fixed defect, 2 = reversable defect) | int |
| `target` | **Label** — Heart disease presence (1 = disease, 0 = healthy) | int |

**Class distribution:** ~54.5% positive (heart disease), ~45.5% negative (healthy) — well balanced.

---

## Project Structure

```
Heart-Disease-Prediction/
│
├── heart_disease_data.csv           # Dataset
├── Heart_Disease_Prediction.ipynb   # Jupyter notebook (full pipeline)
└── README.md
```

---

## Methodology

### 1. Data Collection & Exploration
- Loaded the CSV into a Pandas DataFrame
- Inspected shape, data types, and descriptive statistics
- Verified there are **no missing values**
- Checked target variable distribution

### 2. Preprocessing
- Separated features (`X`) and target label (`Y`)
- Split data into **80% training / 20% test** with stratification to preserve class balance

### 3. Model Training
- Algorithm: **Logistic Regression** (`max_iter=1000`)
- Trained on the training split

### 4. Evaluation
- Measured **accuracy score** on both training and test sets
- Checked for overfitting: if the accuracy gap exceeds 10%, the model is flagged as potentially overfitting

### 5. Predictive System
- Accepts raw patient data as a tuple of 13 values
- Converts to a NumPy array, reshapes for single-instance prediction
- Outputs whether the person has heart disease or not

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn
```

### Running the Notebook

1. Clone this repository
2. Upload `heart_disease_data.csv` to your environment (or update the file path in the notebook)
3. Open `Heart_Disease_Prediction.ipynb` in Jupyter or Google Colab
4. Run all cells sequentially

> **Google Colab users:** The notebook uses `google.colab.drive` to mount Google Drive. Update the CSV path to match your Drive directory.

### Making a Prediction

To predict for a new patient, update the `input_data` tuple in the last cell with 13 feature values in this order:

```python
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
```

**Example:**
```python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
# Output: The Person has Heart Disease
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Array manipulation and reshaping |
| `pandas` | Data loading and exploration |
| `scikit-learn` | Model training, splitting, and evaluation |

---

## Disclaimer

This project is intended for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.
