import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import joblib

PIPELINE_PATH    = os.path.join(os.path.dirname(__file__), '..', 'results', 'churn_pipeline.pkl')
NEW_CUSTOMERS    = os.path.join(os.path.dirname(__file__), '..', 'data', 'new_customers.csv')
PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'predictions.csv')


artifact = joblib.load(PIPELINE_PATH)
pipeline  = artifact['pipeline']
threshold = artifact['threshold']

df = pd.read_csv(NEW_CUSTOMERS)

customer_ids = df['customerID'].copy()
X = df.drop(columns=['customerID'])

proba = pipeline.predict_proba(X)[:, 1]
pred  = (proba >= threshold).astype(int)

output = pd.DataFrame({
    'customerID':  customer_ids,
    'churn_pred':  pred,
    'churn_proba': proba.round(4),
})

output.to_csv(PREDICTIONS_PATH, index=False)
print(f"predictions saved to {PREDICTIONS_PATH}")
print(output.to_string(index=False))
