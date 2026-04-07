# Credit Risk Classification

Predicts whether a loan applicant is a good or bad credit risk using statistical feature selection and ensemble ML models.

**Live demo:** *(add Streamlit Cloud URL here once deployed)*

## Overview

Banks need to assess credit risk before approving loans. This project builds an end-to-end ML pipeline that:
- Selects statistically significant features using Chi-square, VIF, and ANOVA tests
- Compares Random Forest, XGBoost, and Decision Tree models
- Tunes the best model (XGBoost) with GridSearchCV
- Serves predictions through an interactive Streamlit app

## Dataset

[German Credit Data](https://www.openml.org/d/31) — 1,000 loan applicants, 20 features, binary target (good/bad credit risk).

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| XGBoost (tuned) | ~77% | ~0.79 |
| Random Forest | ~75% | ~0.77 |
| Decision Tree | ~70% | ~0.65 |

## Feature Selection

Three-step statistical approach:
1. **Chi-square test** — filters categorical features (p ≤ 0.05)
2. **VIF** — removes multicollinear numerical features (VIF ≤ 6)
3. **ANOVA F-test** — confirms numerical features differ across classes

## Run Locally

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb   # train and save model
streamlit run app.py               # launch the app
```

## Tech Stack

Python · scikit-learn · XGBoost · pandas · Streamlit · Jupyter

## Author

**Suchit Mathur** — [LinkedIn](https://linkedin.com/in/suchitmathur) | [Email](mailto:suchitmathur96@gmail.com)
