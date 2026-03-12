from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np
from datetime import datetime

# ── Load both models ──
xgb_model = joblib.load("model/fraud_model.pkl")
lgb_model = joblib.load("model/lgb_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")
xgb_threshold = joblib.load("model/threshold.pkl")
lgb_threshold = joblib.load("model/lgb_threshold.pkl")

# ── SHAP explainer (XGBoost) ──
explainer = shap.TreeExplainer(xgb_model)

# ── FastAPI app ──
app = FastAPI(title="FraudShield API")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory storage ──
transaction_history = []

# ── Input models ──
class Transaction(BaseModel):
    features: dict

class AgentQuery(BaseModel):
    question: str

# ── Routes ──
@app.get("/")
def root():
    return {"message": "FraudShield API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    row = pd.DataFrame([transaction.features])[feature_names]

    # Both models score the transaction
    xgb_score = float(xgb_model.predict_proba(row)[0][1])
    lgb_score = float(lgb_model.predict_proba(row)[0][1])

    # Ensemble — weighted average (XGBoost slightly favoured)
    final_score = (xgb_score * 0.6) + (lgb_score * 0.4)

    # Decision
    if xgb_score >= xgb_threshold or lgb_score >= lgb_threshold:
        decision = "BLOCK"
        color = "red"
    elif final_score >= 0.4:
        decision = "FLAG"
        color = "yellow"
    else:
        decision = "APPROVE"
        color = "green"

    # Save to history
    record = {
        "id": len(transaction_history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": round(final_score, 4),
        "decision": decision,
        "color": color,
        "amount": transaction.features.get("amount_log", 0),
        "xgb_score": round(xgb_score, 4),
        "lgb_score": round(lgb_score, 4),
    }
    transaction_history.append(record)

    return {
        "fraud_score": round(final_score, 4),
        "decision": decision,
        "color": color,
        "confidence": f"{round(final_score * 100, 1)}%",
        "xgb_score": round(xgb_score, 4),
        "lgb_score": round(lgb_score, 4),
        "models_used": "XGBoost + LightGBM Ensemble"
    }

@app.post("/explain")
def explain(transaction: Transaction):
    row = pd.DataFrame([transaction.features])[feature_names]
    shap_values = explainer.shap_values(row)

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
        "models": {
            "xgboost": {
                "precision": 0.96,
                "recall": 0.76,
                "f1_score": 0.85,
                "auc_roc": 0.9833,
                "threshold": round(float(xgb_threshold), 4)
            },
            "lightgbm": {
                "precision": 0.88,
                "recall": 0.81,
                "f1_score": 0.84,
                "auc_roc": 0.9865,
                "threshold": round(float(lgb_threshold), 4)
            },
            "ensemble": {
                "auc_roc": 0.9854,
                "strategy": "Weighted average — XGBoost 60%, LightGBM 40%"
            }
        },
        "total_transactions": len(transaction_history),
        "fraud_blocked": sum(1 for t in transaction_history if t["decision"] == "BLOCK"),
        "flagged": sum(1 for t in transaction_history if t["decision"] == "FLAG"),
        "approved": sum(1 for t in transaction_history if t["decision"] == "APPROVE"),
    }

@app.post("/simulate")
def simulate():
    """Generate a random fake transaction for demo purposes"""
    import random
    fake = {f: round(random.uniform(-3, 3), 4) for f in feature_names if f not in ["hour", "amount_log"]}
    fake["hour"] = random.randint(0, 23)
    fake["amount_log"] = round(random.uniform(0, 8), 4)
    return {"features": fake}

@app.get("/patterns")
def patterns():
    """Detect fraud patterns from transaction history"""
    if len(transaction_history) < 3:
        return {"pattern": None, "message": "Not enough transactions to detect patterns"}

    recent = transaction_history[-10:]
    blocked = [t for t in recent if t["decision"] == "BLOCK"]
    flagged = [t for t in recent if t["decision"] == "FLAG"]
    scores = [t["score"] for t in recent]
    avg_score = sum(scores) / len(scores)

    # Card testing pattern — many small transactions before big fraud
    amounts = [t["amount"] for t in recent]
    if len(amounts) >= 3:
        small_then_large = all(amounts[i] < amounts[i+1] for i in range(len(amounts)-1))
    else:
        small_then_large = False

    if len(blocked) >= 3:
        pattern = "MASS_FRAUD"
        description = "Multiple blocked transactions detected — possible account takeover"
        risk = "CRITICAL"
    elif small_then_large and len(blocked) >= 1:
        pattern = "CARD_TESTING"
        description = "Escalating transaction amounts — possible card testing attack"
        risk = "HIGH"
    elif avg_score > 0.6:
        pattern = "SUSPICIOUS_ACTIVITY"
        description = "Consistently high fraud scores — account needs review"
        risk = "MEDIUM"
    else:
        pattern = "NORMAL"
        description = "No suspicious patterns detected"
        risk = "LOW"

    return {
        "pattern": pattern,
        "description": description,
        "risk": risk,
        "recent_blocked": len(blocked),
        "recent_flagged": len(flagged),
        "avg_fraud_score": round(avg_score, 4)
    }

@app.post("/agent/chat")
def agent_chat(query: AgentQuery):
    """AI agent endpoint — ask questions in plain English"""
    from agent import run_agent
    response = run_agent(query.question)
    return {"response": response}