from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np
from datetime import datetime
import random

# ── Load model files ──
model = joblib.load("model/fraud_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")
threshold = joblib.load("model/threshold.pkl")

# ── Setup explainer ──
explainer = shap.TreeExplainer(model)

# ── Create FastAPI app ──
app = FastAPI(title="FraudShield API")

# ── Allow frontend to call this API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory transaction history ──
transaction_history = []

# ── Input format ──
class Transaction(BaseModel):
    features: dict

# ── Routes ──
@app.get("/")
def root():
    return {"message": "FraudShield API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Build input row
    row = pd.DataFrame([transaction.features])[feature_names]
    
    # Get fraud score
    score = float(model.predict_proba(row)[0][1])
    
    # Apply threshold
    if score >= threshold:
        decision = "BLOCK"
        color = "red"
    elif score >= 0.4:
        decision = "FLAG"
        color = "yellow"
    else:
        decision = "APPROVE"
        color = "green"

    # Save to history
    record = {
        "id": len(transaction_history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": round(score, 4),
        "decision": decision,
        "color": color,
        "amount": transaction.features.get("amount_log", 0),
    }
    transaction_history.append(record)

    return {
        "fraud_score": round(score, 4),
        "decision": decision,
        "color": color,
        "confidence": f"{round(score * 100, 1)}%"
    }

@app.post("/explain")
def explain(transaction: Transaction):
    row = pd.DataFrame([transaction.features])[feature_names]
    shap_values = explainer.shap_values(row)
    
    # Get top 5 most important features
    importance = dict(zip(feature_names, np.abs(shap_values[0])))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "top_features": [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in top_features
        ]
    }

@app.get("/history")
def history():
    return {"transactions": transaction_history[-20:]}

@app.get("/stats")
def stats():
    return {
        "precision": 0.96,
        "recall": 0.76,
        "f1_score": 0.85,
        "auc_roc": 0.9833,
        "threshold": round(threshold, 4),
        "total_transactions": len(transaction_history),
        "fraud_blocked": sum(1 for t in transaction_history if t["decision"] == "BLOCK"),
        "flagged": sum(1 for t in transaction_history if t["decision"] == "FLAG"),
        "approved": sum(1 for t in transaction_history if t["decision"] == "APPROVE"),
    }