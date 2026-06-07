import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# service columns used to compute n_services
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

# numeric columns going into the pipeline (after feature engineering)
NUM_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'SeniorCitizen', 'n_services', 'charges_per_month',
]

# categorical columns going into the pipeline (after feature engineering)
CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_bucket',
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """adds engineered features before the column transformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # fix TotalCharges: loaded as object due to empty strings
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

        # feature 1: tenure_bucket — which lifecycle stage the customer is in
        X['tenure_bucket'] = pd.cut(
            X['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12', '13-24', '25-48', '49+'],
            include_lowest=True,
        ).astype(str)

        # feature 2: n_services — counts active services; avoids the "No"/"No internet service" trap
        def _count_active(row):
            return sum(
                1 for col in SERVICE_COLS
                if row[col] not in ('No', 'No internet service')
            )
        X['n_services'] = X.apply(_count_active, axis=1)

        # feature 3: charges_per_month — avg monthly spend proxy using full history
        X['charges_per_month'] = X['TotalCharges'] / X['tenure'].clip(lower=1)

        return X
