# FastAPI app that serves the champion fraud model (tuned XGBoost)
# Run from the project root:  uvicorn src.app:app --reload
# Then open http://127.0.0.1:8000/docs to try it out

import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# load the saved model package (created by the "Save the Champion Model" cell in 01_EDA.ipynb)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_model.joblib')
package = joblib.load(MODEL_PATH)

model = package['model']
preprocessor = package['preprocessor']
feature_columns = package['feature_columns']
medians = package['medians']
threshold = package['threshold']

feature_names = [name.replace('cat__', '').replace('num__', '') for name in preprocessor.get_feature_names_out()]
explainer = shap.TreeExplainer(model)

app = FastAPI(
    title="Nova Pay Fraud Detection API",
    description="Scores money-transfer transactions for fraud using the tuned XGBoost model.",
    version="1.0.0"
)


# ----- request and response formats -----

class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    timestamp: str = Field(..., example="2025-11-26T07:09:56+00:00")
    home_country: str = Field(..., example="CA")
    source_currency: str = Field(..., example="CAD")
    dest_currency: str = Field(..., example="NGN")
    channel: str = Field(..., example="web")
    amount_src: float = Field(..., example=537.17)
    amount_usd: Optional[float] = Field(None, example=397.51)
    fee: Optional[float] = Field(None, example=8.86)
    exchange_rate_src_to_dest: float = Field(..., example=822.22)
    new_device: bool = Field(..., example=False)
    ip_country: Optional[str] = Field(None, example="UK")
    location_mismatch: bool = Field(..., example=True)
    ip_risk_score: float = Field(..., example=0.891)
    kyc_tier: Optional[str] = Field(None, example="low")
    account_age_days: int = Field(..., example=9)
    device_trust_score: Optional[float] = Field(None, example=0.173)
    chargeback_history_count: int = Field(..., example=2)
    risk_score_internal: float = Field(..., example=0.73)
    txn_velocity_1h: int = Field(..., example=3)
    txn_velocity_24h: int = Field(..., example=6)
    corridor_risk: float = Field(..., example=0.0)


class Reason(BaseModel):
    feature: str
    impact: float


class Prediction(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    top_reasons: List[Reason]


# ----- prepare a transaction the same way as in the notebook -----

typo_fixes = {
    'mobille': 'mobile',
    'weeb': 'web',
    'standrd': 'standard',
    'enhancd': 'enhanced',
    'nan': 'unknown'
}


def prepare_features(transactions):
    df = pd.DataFrame([t.dict() for t in transactions])

    # timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    if df['timestamp'].isnull().any():
        raise HTTPException(status_code=422, detail="timestamp could not be read - use a format like 2025-11-26T07:09:56+00:00")

    # missing text values become 'unknown', then clean spaces, case and typos
    for col in ['home_country', 'channel', 'ip_country', 'kyc_tier']:
        df[col] = df[col].fillna('unknown').str.strip().str.lower().replace(typo_fixes)
    df['source_currency'] = df['source_currency'].str.strip().str.upper()
    df['dest_currency'] = df['dest_currency'].str.strip().str.upper()

    # invalid values become missing, then all missing numbers are filled with the training medians
    df.loc[(df['fee'] < 0) | (df['fee'] == 9999.99), 'fee'] = np.nan
    df.loc[df['device_trust_score'] < 0, 'device_trust_score'] = np.nan
    df.loc[df['ip_risk_score'] > 1, 'ip_risk_score'] = np.nan
    df.loc[df['txn_velocity_1h'] < 0, 'txn_velocity_1h'] = np.nan
    for col, median in medians.items():
        df[col] = df[col].fillna(median)

    # feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_night'] = df['hour'] < 6
    df['fee_percent'] = df['fee'] / df['amount_src'] * 100
    df['is_new_account'] = df['account_age_days'] < 30

    return df[feature_columns]


def predict(transactions):
    features = preprocessor.transform(prepare_features(transactions))
    if hasattr(features, 'toarray'):
        features = features.toarray()

    probabilities = model.predict_proba(features)[:, 1]
    shap_values = explainer.shap_values(features)

    results = []
    for i, t in enumerate(transactions):
        # top 3 features pushing this transaction towards fraud
        row_shap = pd.Series(shap_values[i], index=feature_names).sort_values(ascending=False).head(3)
        results.append(Prediction(
            transaction_id=t.transaction_id,
            fraud_probability=round(float(probabilities[i]), 4),
            is_fraud=bool(probabilities[i] >= threshold),
            top_reasons=[Reason(feature=name, impact=round(float(value), 4)) for name, value in row_shap.items()]
        ))
    return results


# ----- endpoints -----

@app.get("/health")
def health():
    return {"status": "ok", "model": package['model_name'], "threshold": threshold}


@app.post("/predict", response_model=Prediction)
def predict_one(transaction: Transaction):
    return predict([transaction])[0]


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(transactions: List[Transaction]):
    if len(transactions) == 0:
        raise HTTPException(status_code=422, detail="send at least one transaction")
    if len(transactions) > 1000:
        raise HTTPException(status_code=422, detail="send at most 1000 transactions per request")
    return predict(transactions)
