# Credit Risk Scoring — Loan Approval Prediction

Classifies loan applicants into credit risk tiers (P1–P4) using statistical feature selection and ensemble ML models.

**Live demo:** https://mathursuchit-credit-risk-model.streamlit.app/

## Overview

Banks use credit bureau data to assess loan applicants. This project builds an end-to-end ML pipeline that:
- Merges and cleans two credit bureau datasets (51,336 applicants, 80+ features)
- Selects statistically significant features using Chi-square, VIF, and ANOVA tests
- Compares Random Forest, XGBoost, and Decision Tree models
- Tunes the best model (XGBoost) with GridSearchCV
- Serves predictions through an interactive Streamlit app

## Dataset

Credit bureau data — 51,336 loan applicants, trade line history + delinquency + demographics.
Target: P1 (lowest risk) → P4 (highest risk)

## Results

| Model | Accuracy |
|-------|----------|
| XGBoost (tuned) | ~78% |
| Random Forest | ~76% |
| Decision Tree | ~70% |

## Feature Selection

Three-step statistical approach:
1. **Chi-square test** — filters categorical features (p ≤ 0.05)
2. **VIF** — removes multicollinear numerical features (VIF ≤ 6)
3. **ANOVA F-test** — confirms numerical features differ across risk tiers

## Run Locally

```bash
pip install -r requirements.txt
# Place case_study1.xlsx and case_study2.xlsx in data/
jupyter notebook notebook.ipynb   # train and save model
streamlit run app.py               # launch the app
```

## Tech Stack

Python · scikit-learn · XGBoost · pandas · Streamlit · Jupyter

## Author

**Suchit Mathur** — [LinkedIn](https://www.linkedin.com/in/mathursuchit/) | [Email](mailto:suchitmathur96@gmail.com)
