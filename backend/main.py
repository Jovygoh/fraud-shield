from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np
from datetime import datetime
import math

# ── Load all 3 models ──
xgb_model = joblib.load("model/fraud_model.pkl")
lgb_model = joblib.load("model/lgb_model.pkl")
paysim_model = joblib.load("model/paysim_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")
paysim_feature_names = joblib.load("model/paysim_feature_names.pkl")
xgb_threshold = joblib.load("model/threshold.pkl")
lgb_threshold = joblib.load("model/lgb_threshold.pkl")
paysim_threshold = joblib.load("model/paysim_threshold.pkl")

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

# ── Human-readable feature name map ──
FEATURE_LABELS = {
    "amount_log":        "transaction amount",
    "hour":              "time of day",
    "is_transfer":       "transaction type (transfer)",
    "balance_mismatch":  "balance mismatch",
    "orig_balance_diff": "sender balance change",
    "dest_balance_diff": "receiver balance change",
}

def get_feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    import re
    m = re.match(r'^V(\d+)$', name, re.IGNORECASE)
    if m:
        return f"behavioural signal V{m.group(1)}"
    return name

# ── Routes ──
@app.get("/")
def root():
    return {"message": "FraudShield API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    row = pd.DataFrame([transaction.features])[feature_names]

    xgb_score = float(xgb_model.predict_proba(row)[0][1])
    lgb_score = float(lgb_model.predict_proba(row)[0][1])

    features_with_defaults = {**transaction.features}
    for col in paysim_feature_names:
        if col not in features_with_defaults:
            features_with_defaults[col] = 0
    paysim_row = pd.DataFrame([features_with_defaults])[paysim_feature_names]
    paysim_score = float(paysim_model.predict_proba(paysim_row)[0][1])

    final_score = (xgb_score * 0.4) + (lgb_score * 0.3) + (paysim_score * 0.3)

    if xgb_score >= xgb_threshold or lgb_score >= lgb_threshold or paysim_score >= paysim_threshold:
        decision = "BLOCK"
        color = "red"
    elif final_score >= 0.4:
        decision = "FLAG"
        color = "yellow"
    else:
        decision = "APPROVE"
        color = "green"

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
        "paysim_score": round(paysim_score, 4),
        "models_used": "XGBoost + LightGBM + PaySim Ensemble"
    }

@app.post("/explain")
def explain(transaction: Transaction):
    row = pd.DataFrame([transaction.features])[feature_names]
    shap_values = explainer.shap_values(row)

    signed = dict(zip(feature_names, shap_values[0]))
    top_features = sorted(signed.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    top_features_out = [
        {"feature": f, "importance": round(float(v), 4)}
        for f, v in top_features
    ]

    # ── Build context for Groq summary ───────────────────────────────────────
    amount_rm = round(math.exp(transaction.features.get("amount_log", 0)), 2)
    hour      = transaction.features.get("hour", None)
    is_transfer = transaction.features.get("is_transfer", 0)

    # Decide overall decision from features (re-score quickly for context)
    try:
        xgb_score   = float(xgb_model.predict_proba(row)[0][1])
        lgb_score   = float(lgb_model.predict_proba(row)[0][1])
        final_score = xgb_score * 0.4 + lgb_score * 0.3
        if xgb_score >= float(xgb_threshold) or lgb_score >= float(lgb_threshold):
            decision = "BLOCK"
        elif final_score >= 0.4:
            decision = "FLAG"
        else:
            decision = "APPROVE"
    except Exception:
        decision = "FLAG"

    # Build a readable list of the top contributing factors
    factor_lines = []
    for f, v in top_features:
        label     = get_feature_label(f)
        direction = "increased fraud risk" if v > 0 else "reduced fraud risk"
        factor_lines.append(f"- {label}: {direction} (SHAP {round(float(v), 4)})")
    factors_text = "\n".join(factor_lines)

    time_str = f"{int(hour):02d}:00" if hour is not None else "unknown time"
    tx_type  = "transfer" if is_transfer == 1 else "purchase"

    prompt = f"""You are a fraud analyst writing a brief, plain-English explanation for a bank transaction decision.

Transaction details:
- Amount: RM {amount_rm}
- Time: {time_str}
- Type: {tx_type}
- Decision: {decision}

Top factors from the AI model (SHAP analysis):
{factors_text}

Write ONE concise paragraph (2-3 sentences max) explaining why this transaction was {decision.lower()}ed.
- Use plain English that a bank customer could understand
- Mention the RM amount and time naturally if relevant
- Do NOT use technical terms like SHAP, XGBoost, or feature importance
- Do NOT start with "This transaction" — vary the opening
- Be direct and specific"""

    summary = ""
    try:
        from groq import Groq
        import os
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.4,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[explain] Groq summary failed: {e}")
        # Graceful fallback — build a simple rule-based summary
        top_label = get_feature_label(top_features[0][0]) if top_features else "unusual activity"
        summary = (
            f"The RM {amount_rm} {tx_type} at {time_str} was {decision.lower()}ed "
            f"primarily due to {top_label}, which significantly raised the fraud risk score."
        )

    return {
        "top_features": top_features_out,
        "summary": summary
    }

@app.get("/history")
def history():
    return {"transactions": transaction_history}

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
            "paysim": {
                "precision": 0.96,
                "recall": 0.78,
                "f1_score": 0.86,
                "auc_roc": 0.9929,
                "threshold": round(float(paysim_threshold), 4),
                "trained_on": "ASEAN mobile money transactions"
            },
            "ensemble": {
                "strategy": "XGBoost 40% + LightGBM 30% + PaySim 30%",
                "note": "PaySim adds ASEAN digital wallet context"
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
    fake["is_transfer"] = random.randint(0, 1)
    fake["balance_mismatch"] = random.randint(0, 1)
    fake["orig_balance_diff"] = round(random.uniform(-5000, 5000), 2)
    fake["dest_balance_diff"] = round(random.uniform(-5000, 5000), 2)
    return {"features": fake}

@app.get("/patterns")
def patterns():
    """Detect fraud patterns from transaction history"""
    if len(transaction_history) < 3:
        return {"pattern": "NORMAL", "description": "Not enough transactions to detect patterns yet", "risk": "LOW"}

    recent = transaction_history[-10:]
    blocked = [t for t in recent if t["decision"] == "BLOCK"]
    flagged = [t for t in recent if t["decision"] == "FLAG"]
    scores = [t["score"] for t in recent]
    avg_score = sum(scores) / len(scores)

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
    import traceback
    try:
        from agent import run_agent
        response = run_agent(query.question)
        return {"response": response}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[agent/chat ERROR]\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )
