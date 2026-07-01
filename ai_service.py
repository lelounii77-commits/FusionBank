#!/usr/bin/env python3
"""
FusionBankAI - Python AI Service
Serves the XGBoost credit risk model via a Flask REST API on port 5001.
Called internally by the Node.js backend.
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "bank_risk_model.pkl"
CUSTOMERS_PATH = Path(__file__).parent / "customers.csv"
ACCOUNTS_PATH = Path(__file__).parent / "accounts.csv"
OBLIGATIONS_PATH = Path(__file__).parent / "obligations.csv"
AISUMMARY_PATH = Path(__file__).parent / "aisummary.csv"

try:
    model = joblib.load(MODEL_PATH)
    FEATURE_NAMES = [str(f) for f in model.feature_names_in_]
    logger.info(f"Model loaded: {type(model).__name__}, features={len(FEATURE_NAMES)}, classes={model.classes_}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    FEATURE_NAMES = []

# ── Load CSV data ─────────────────────────────────────────────────────────────
try:
    df_customers = pd.read_csv(CUSTOMERS_PATH)
    df_accounts = pd.read_csv(ACCOUNTS_PATH)
    df_obligations = pd.read_csv(OBLIGATIONS_PATH)
    df_aisummary = pd.read_csv(AISUMMARY_PATH)
    logger.info(f"Data loaded: {len(df_customers)} customers, {len(df_accounts)} accounts, {len(df_obligations)} obligations")
except Exception as e:
    logger.error(f"Failed to load CSV data: {e}")
    df_customers = df_accounts = df_obligations = df_aisummary = None

# ── Class mapping ─────────────────────────────────────────────────────────────
# Based on testing: 0=Reject, 1=Approve, 2=Manual Review
CLASS_MAP = {0: "Reject", 1: "Approve", 2: "Manual Review"}
RISK_MAP = {0: "High", 1: "Low", 2: "Medium"}

def build_feature_vector(customer_id: str) -> pd.DataFrame:
    """Build the 90-feature vector for a given customer_id."""
    if df_customers is None:
        raise ValueError("Customer data not loaded")
    
    cust_rows = df_customers[df_customers['customer_id'] == customer_id]
    if len(cust_rows) == 0:
        raise ValueError(f"Customer {customer_id} not found")
    
    cust = cust_rows.iloc[0]
    cust_accounts = df_accounts[df_accounts['customer_id'] == customer_id]
    cust_obligations = df_obligations[df_obligations['customer_id'] == customer_id]
    
    features = {f: 0 for f in FEATURE_NAMES}
    
    # Numeric features
    features['age'] = float(cust.get('age', 0) or 0)
    features['dependents'] = float(cust.get('dependents', 0) or 0)
    features['monthly_salary'] = float(cust.get('monthly_salary', 0) or 0)
    features['years_of_employment'] = float(cust.get('years_of_employment', 0) or 0)
    features['num_accounts'] = float(len(cust_accounts))
    features['total_account_balance'] = float(cust_accounts['account_balance'].sum()) if len(cust_accounts) > 0 else 0.0
    features['avg_monthly_balance'] = float(cust_accounts['average_monthly_balance'].mean()) if len(cust_accounts) > 0 else 0.0
    features['num_obligations'] = float(len(cust_obligations))
    features['total_obligation_amount'] = float(cust_obligations['total_amount'].sum()) if len(cust_obligations) > 0 else 0.0
    features['total_remaining_amount'] = float(cust_obligations['remaining_amount'].sum()) if len(cust_obligations) > 0 else 0.0
    features['total_income'] = float(cust.get('monthly_salary', 0) or 0)
    
    # One-hot features
    if str(cust.get('nationality', '')) == 'Saudi':
        features['nationality_Saudi'] = 1
    
    city = str(cust.get('city', ''))
    if f'city_{city}' in features:
        features[f'city_{city}'] = 1
    
    if str(cust.get('gender', '')) == 'Male':
        features['gender_Male'] = 1
    
    marital = str(cust.get('marital_status', ''))
    if f'marital_status_{marital}' in features:
        features[f'marital_status_{marital}'] = 1
    
    emp_status = str(cust.get('employment_status', ''))
    if f'employment_status_{emp_status}' in features:
        features[f'employment_status_{emp_status}'] = 1
    
    emp_sector = str(cust.get('employment_sector', ''))
    if f'employment_sector_{emp_sector}' in features:
        features[f'employment_sector_{emp_sector}'] = 1
    
    sal_freq = str(cust.get('salary_frequency', ''))
    if f'salary_frequency_{sal_freq}' in features:
        features[f'salary_frequency_{sal_freq}'] = 1
    
    job = str(cust.get('job_title', ''))
    if f'job_title_{job}' in features:
        features[f'job_title_{job}'] = 1
    
    edu = str(cust.get('education_level', ''))
    if f'education_level_{edu}' in features:
        features[f'education_level_{edu}'] = 1
    
    lang = str(cust.get('preferred_language', ''))
    if f'preferred_language_{lang}' in features:
        features[f'preferred_language_{lang}'] = 1
    
    status = str(cust.get('customer_status', ''))
    if f'customer_status_{status}' in features:
        features[f'customer_status_{status}'] = 1
    
    return pd.DataFrame([features])


def get_feature_importances() -> list:
    """Return top feature importances from the model."""
    if model is None:
        return []
    importances = model.feature_importances_
    feature_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in feature_imp[:15]]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route('/predict/<customer_id>', methods=['GET'])
def predict(customer_id: str):
    """Run AI prediction for a given customer_id."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        sample = build_feature_vector(customer_id)
        pred_class = int(model.predict(sample)[0])
        proba = model.predict_proba(sample)[0].tolist()
        
        recommendation = CLASS_MAP.get(pred_class, "Manual Review")
        risk_level = RISK_MAP.get(pred_class, "Medium")
        
        # Approval probability = class 1 probability
        approval_probability = round(proba[1] * 100, 1)
        default_probability = round(proba[0] * 100, 1)
        review_probability = round(proba[2] * 100, 1)
        
        # Get customer data for context
        cust = df_customers[df_customers['customer_id'] == customer_id].iloc[0]
        cust_accounts = df_accounts[df_accounts['customer_id'] == customer_id]
        cust_obligations = df_obligations[df_obligations['customer_id'] == customer_id]
        
        total_balance = float(cust_accounts['account_balance'].sum()) if len(cust_accounts) > 0 else 0
        total_obligations = float(cust_obligations['remaining_amount'].sum()) if len(cust_obligations) > 0 else 0
        monthly_salary = float(cust.get('monthly_salary', 0) or 0)
        monthly_commitments = float(cust_obligations['monthly_payment'].sum()) if len(cust_obligations) > 0 else 0
        
        dti = round(monthly_commitments / monthly_salary, 4) if monthly_salary > 0 else 0
        
        # Financial stability score (0-100)
        stability_score = min(100, max(0, round(approval_probability)))
        
        # Get AI summary from CSV if available
        ai_row = df_aisummary[df_aisummary['customer_id'] == customer_id]
        explainable_reason = ""
        if len(ai_row) > 0:
            explainable_reason = str(ai_row['explainable_reason'].values[0])
        else:
            # Generate reason based on prediction
            if recommendation == "Approve":
                explainable_reason = "Excellent Repayment Capacity"
            elif recommendation == "Reject":
                reasons = []
                if dti > 0.5:
                    reasons.append("High Debt-to-Income Ratio")
                if total_obligations > total_balance:
                    reasons.append("Obligations Exceed Assets")
                explainable_reason = " | ".join(reasons) if reasons else "High Credit Risk"
            else:
                explainable_reason = "Requires Manual Review"
        
        # Top contributing features
        top_features = get_feature_importances()[:10]
        
        result = {
            "customer_id": customer_id,
            "prediction_class": pred_class,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "approval_probability": approval_probability,
            "default_probability": default_probability,
            "review_probability": review_probability,
            "probabilities": {
                "reject": round(proba[0], 4),
                "approve": round(proba[1], 4),
                "review": round(proba[2], 4)
            },
            "financial_metrics": {
                "total_balance": round(total_balance, 2),
                "total_obligations": round(total_obligations, 2),
                "monthly_salary": round(monthly_salary, 2),
                "monthly_commitments": round(monthly_commitments, 2),
                "debt_to_income_ratio": round(dti, 4),
                "financial_stability_score": stability_score,
                "num_accounts": int(len(cust_accounts)),
                "num_obligations": int(len(cust_obligations))
            },
            "explainable_reason": explainable_reason,
            "top_features": top_features
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Prediction error for {customer_id}: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/customers', methods=['GET'])
def get_customers():
    """Return list of all customers with basic info."""
    if df_customers is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    search = request.args.get('search', '').strip().lower()
    limit = int(request.args.get('limit', 50))
    
    df = df_customers.copy()
    
    if search:
        mask = (
            df['full_name'].str.lower().str.contains(search, na=False) |
            df['national_id'].astype(str).str.contains(search, na=False) |
            df['customer_id'].str.lower().str.contains(search, na=False)
        )
        df = df[mask]
    
    df = df.head(limit)
    
    customers = []
    for _, row in df.iterrows():
        customers.append({
            "customer_id": str(row['customer_id']),
            "national_id": str(row['national_id']),
            "full_name": str(row['full_name']),
            "gender": str(row['gender']),
            "age": int(row['age']) if not math.isnan(float(row['age'])) else 0,
            "city": str(row['city']),
            "monthly_salary": float(row['monthly_salary']) if not math.isnan(float(row['monthly_salary'])) else 0,
            "employment_status": str(row['employment_status']),
            "job_title": str(row['job_title']),
            "customer_status": str(row['customer_status'])
        })
    
    return jsonify({"customers": customers, "total": len(customers)})


@app.route('/customers/<customer_id>', methods=['GET'])
def get_customer(customer_id: str):
    """Return full customer profile with accounts and obligations."""
    if df_customers is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    cust_rows = df_customers[df_customers['customer_id'] == customer_id]
    if len(cust_rows) == 0:
        return jsonify({"error": "Customer not found"}), 404
    
    cust = cust_rows.iloc[0].to_dict()
    # Convert NaN to None
    for k, v in cust.items():
        if isinstance(v, float) and math.isnan(v):
            cust[k] = None
    
    # Get accounts
    cust_accounts = df_accounts[df_accounts['customer_id'] == customer_id]
    accounts = []
    for _, row in cust_accounts.iterrows():
        acc = row.to_dict()
        for k, v in acc.items():
            if isinstance(v, float) and math.isnan(v):
                acc[k] = None
        accounts.append(acc)
    
    # Get obligations
    cust_obligations = df_obligations[df_obligations['customer_id'] == customer_id]
    obligations = []
    for _, row in cust_obligations.iterrows():
        obl = row.to_dict()
        for k, v in obl.items():
            if isinstance(v, float) and math.isnan(v):
                obl[k] = None
        obligations.append(obl)
    
    # Get AI summary
    ai_row = df_aisummary[df_aisummary['customer_id'] == customer_id]
    ai_summary = None
    if len(ai_row) > 0:
        ai_summary = ai_row.iloc[0].to_dict()
        for k, v in ai_summary.items():
            if isinstance(v, float) and math.isnan(v):
                ai_summary[k] = None
    
    return jsonify({
        "customer": cust,
        "accounts": accounts,
        "obligations": obligations,
        "ai_summary": ai_summary
    })


@app.route('/feature-importances', methods=['GET'])
def feature_importances():
    """Return top feature importances for explainability."""
    return jsonify({"features": get_feature_importances()})


if __name__ == '__main__':
    # Railway uses PORT env variable; fallback to AI_SERVICE_PORT or 5001
    port = int(os.environ.get('PORT', os.environ.get('AI_SERVICE_PORT', 5001)))
    logger.info(f"Starting FusionBankAI service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
