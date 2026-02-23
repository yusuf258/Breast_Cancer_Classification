# Breast Cancer Patient Status Classification

Binary classification model to predict breast cancer patient survival status using clinical and molecular biomarker data.

## Problem Statement
Predict whether a breast cancer patient will be **Alive** or **Dead** based on clinical features such as tumour stage, protein expression levels, histology type, and surgery type.

## Dataset
| Attribute | Detail |
|---|---|
| File | `BRCA.csv` |
| Records | 341 patients (334 valid after cleaning) |
| Features | 16 (5 numeric, 11 categorical) |
| Target | `Patient_Status` (Alive / Dead) |
| Class Balance | ~80% Alive / ~20% Dead (imbalanced) |

## Methodology
1. **EDA & Visualization** — Distribution plots, correlation heatmap, missing value matrix
2. **Feature Engineering** — Date columns removed, column names standardized
3. **Outlier Clipping** — Values bounded to 10th–90th percentile range
4. **Preprocessing Pipeline** — `SimpleImputer` + `StandardScaler` + `OneHotEncoder` / `OrdinalEncoder` via `ColumnTransformer`
5. **ML Models** — Logistic Regression, Random Forest, XGBoost (comparison)
6. **DL Model** — Dense Neural Network (120 → 64 → 30 → 8 → 1), `EarlyStopping` (patience=10)
7. **Evaluation** — Confusion Matrix, ROC-AUC Curve, ML vs DL comparison table

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 0.8000 |
| Random Forest | 0.7846 |
| **XGBoost (Best)** | **0.8154** |
| Neural Network (DL) | Dense sigmoid, binary cross-entropy |

> **Note:** Due to class imbalance, `Dead` class recall is low (0.31). SMOTE or `class_weight` tuning can improve minority class detection.

## Technologies
`Python` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `missingno` · `joblib`

## File Structure
```
01_Breast_Cancer_Classification/
├── project_notebook.ipynb   # Main notebook
├── BRCA.csv                 # Dataset
└── models/
    ├── best_model.pkl        # Best ML pipeline
    ├── dl_model.keras        # Deep Learning model
    └── label_encoder.pkl     # Target label encoder
```

## How to Run
```bash
cd 01_Breast_Cancer_Classification
jupyter notebook project_notebook.ipynb
```
