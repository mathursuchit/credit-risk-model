# Credit Risk Scoring — Loan Approval Prediction

Classifies loan applicants into credit risk tiers (P1–P4) using XGBoost trained on real credit bureau data.

**Live demo:** https://mathursuchit-credit-risk-model.streamlit.app/

## What it does

Takes applicant details (income, age, employment history, delinquency records) and predicts which risk tier they fall into — P1 (low risk) through P4 (high risk). Built to mimic how banks actually assess loan applications.

## Dataset

Two credit bureau datasets merged on applicant ID — 51,336 applicants, 80+ features covering trade lines, delinquency history, and demographics.

## Approach

Started with 80+ raw features. Used Chi-square to filter categoricals, VIF to remove multicollinear numerics, and ANOVA to confirm the remaining features actually differ across tiers. Ended up with ~48 meaningful features.

Compared Random Forest, XGBoost, and Decision Tree. XGBoost won, tuned further with GridSearchCV.

## Results

| Model | Accuracy |
|-------|----------|
| XGBoost (tuned) | 77.6% |
| Random Forest | 76.7% |
| Decision Tree | 70.8% |

## Run locally

```bash
pip install -r requirements.txt
# place case_study1.xlsx and case_study2.xlsx in data/
jupyter notebook notebook.ipynb
streamlit run app.py
```

## Stack

Python · XGBoost · scikit-learn · pandas · Streamlit

## Author

Suchit Mathur — [LinkedIn](https://www.linkedin.com/in/mathursuchit/)
