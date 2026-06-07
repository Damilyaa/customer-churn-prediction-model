# Telco Customer Churn Prediction

Predicts which customers are likely to churn so the marketing team can call them before they cancel.

## Why not accuracy?

The dataset is ~73% non-churn, so a model that always predicts "no churn" already hits 73% accuracy while catching zero churners. We use **recall** and **ROC-AUC** instead.

## Engineered features

- `tenure_bucket` — bins tenure into lifecycle stages (0-12, 13-24, 25-48, 49+ months)
- `n_services` — count of active services (customers with more services are less likely to leave)
- `charges_per_month` — `TotalCharges / max(tenure, 1)` to normalize spend for new customers

## Results

5-fold stratified CV:

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.797 ± 0.035 | 0.518 ± 0.016 | 0.628 ± 0.021 | 0.846 ± 0.011 |
| Random Forest | 0.461 ± 0.015 | 0.641 ± 0.025 | 0.536 ± 0.016 | 0.825 ± 0.010 |
| Gradient Boosting | 0.522 ± 0.020 | 0.665 ± 0.034 | 0.584 ± 0.021 | 0.848 ± 0.011 |

Best model (Logistic Regression) on held-out test set at threshold 0.498:

| Metric | Value |
|---|---|
| Recall | 0.794 |
| Precision | 0.499 |
| ROC-AUC | 0.842 |

At this threshold the model flags ~422 out of every 1000 customers, of whom ~210 are real churners.

## Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 scripts/train.py
python3 scripts/predict.py
```
