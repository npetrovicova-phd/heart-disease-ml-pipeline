# Heart Disease ML Pipeline
### Machine Learning Fundamentals | Scikit-learn, Model Evaluation & Core Algorithms

---

## Project Overview
This project builds a complete Machine Learning pipeline to predict
heart disease using the UCI Heart Disease dataset (303 patients,
13 features). Three classification models are trained, evaluated,
and compared to determine the best model for deployment.

---

## Dataset
- **Source:** UCI Heart Disease Dataset (via OpenML)
- **Samples:** 303 patients
- **Features:** 13 (age, sex, chest pain type, blood pressure, etc.)
- **Target:** 0 = No Heart Disease, 1 = Heart Disease Present
- **Class balance:** ~54% no disease, ~46% disease

---

## Project Steps

### 1. Train/Test Split (80/20, Stratified)
The dataset is split into 80% training and 20% test data.
Stratification ensures both sets maintain the same class balance,
which is important for fair evaluation in classification tasks.

### 2. sklearn Pipeline (imputer → scaler → classifier)
A Pipeline is used for each model to prevent data leakage:
- **SimpleImputer** — fills missing values with the column mean
- **StandardScaler** — standardizes features to zero mean, unit variance
- **Classifier** — the model (LR, RF, or XGBoost)

The pipeline ensures preprocessing is learned only from training
data and applied consistently to the test set.

### 3. Three Models Trained
| Model | Description |
|---|---|
| Logistic Regression | Simple linear baseline model |
| Random Forest | Ensemble of decision trees (bagging) |
| XGBoost | Gradient boosted trees (often best on tabular data) |

### 4. Evaluation Metrics
Each model is evaluated with:
- **Classification Report** — Precision, Recall, F1-score per class
- **ROC-AUC Score** — How well the model separates the two classes
- **Stratified 5-Fold Cross-Validation** — Stable comparison across
  multiple data splits (on training data only)

### 5. Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | 0.823 ± 0.062 | 0.874 |
| Random Forest | 0.812 ± 0.071 | 0.862 |
| XGBoost | 0.841 ± 0.054 | 0.892 |

---

## Deployment Decision
We would deploy **XGBoost** for the following reasons:

1. XGBoost achieved the highest cross-validation ROC-AUC of 0.841,
   meaning it performed most consistently across all 5 folds on
   unseen training data.

2. On the final test set it scored 0.892 ROC-AUC with balanced
   precision and recall across both classes, indicating it
   generalizes well to new patients.

3. XGBoost is robust to missing values, handles class imbalance
   well, and is a proven top performer on tabular medical data,
   making it the most reliable choice for real-world deployment.

---

## How to Run

### Install dependencies
```bash
pip install scikit-learn xgboost pandas
```

### Run the notebook
Open `heart_disease_project.ipynb` in Jupyter or VS Code
and run all cells top to bottom (Shift + Enter each cell).

