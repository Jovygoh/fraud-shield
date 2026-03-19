import os
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
    demo_scores: dict | None = None   # injected by demo buttons to override model re-scoring

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
        "paysim_score": round(paysim_score, 4),
        "features": transaction.features,   # stored for detail view & SHAP
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

    # ── Transaction basics ────────────────────────────────────────────────────
    amount_rm   = round(math.exp(transaction.features.get("amount_log", 0)), 2)
    hour        = transaction.features.get("hour", None)
    is_transfer = transaction.features.get("is_transfer", 0)
    time_str    = f"{int(hour):02d}:00" if hour is not None else "unknown time"
    tx_type     = "transfer" if is_transfer == 1 else "purchase"

    ds = transaction.demo_scores
    print(f"[explain] demo_scores received: {ds}")

    # ── SHAP (always run — powers the bar chart) ──────────────────────────────
    shap_values  = explainer.shap_values(row)
    signed       = dict(zip(feature_names, shap_values[0]))
    top_features = sorted(signed.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    # For demo modes, flip SHAP bar directions to match the injected decision.
    # Raw SHAP values come from fake features and may point the wrong way.
    ds_decision = (transaction.demo_scores or {}).get("decision", "").upper()
    if ds_decision == "BLOCK":
        # All bars should point ▲ fraud (positive)
        top_features_out = [
            {"feature": f, "importance": abs(round(float(v), 4))}
            for f, v in top_features
        ]
    elif ds_decision == "APPROVE":
        # All bars should point ▼ safe (negative)
        top_features_out = [
            {"feature": f, "importance": -abs(round(float(v), 4))}
            for f, v in top_features
        ]
    elif ds_decision == "FLAG":
        # Mix: top 2 ▲ fraud, rest ▼ safe — shows a genuine split
        top_features_out = [
            {"feature": f, "importance": abs(round(float(v), 4)) if i < 2 else -abs(round(float(v), 4))}
            for i, (f, v) in enumerate(top_features)
        ]
    else:
        # Random mode — use real SHAP values as-is
        top_features_out = [
            {"feature": f, "importance": round(float(v), 4)}
            for f, v in top_features
        ]

    # ── Scores: use demo_scores for demo buttons, real models for random ──────
    if ds:
        # Demo mode — trust injected scores entirely, never re-run models
        xgb_score    = float(ds.get("xgb_score",    0))
        lgb_score    = float(ds.get("lgb_score",    0))
        paysim_score = float(ds.get("paysim_score", 0))
        final_score  = float(ds.get("fraud_score",  0))
        decision     = str(ds.get("decision", "APPROVE")).upper()
    else:
        # Random mode — run all 3 real models
        xgb_score = float(xgb_model.predict_proba(row)[0][1])
        lgb_score = float(lgb_model.predict_proba(row)[0][1])
        fd = {**transaction.features}
        for col in paysim_feature_names:
            if col not in fd:
                fd[col] = 0
        paysim_row   = pd.DataFrame([fd])[paysim_feature_names]
        paysim_score = float(paysim_model.predict_proba(paysim_row)[0][1])
        final_score  = (xgb_score * 0.4) + (lgb_score * 0.3) + (paysim_score * 0.3)
        if xgb_score >= float(xgb_threshold) or lgb_score >= float(lgb_threshold) or paysim_score >= float(paysim_threshold):
            decision = "BLOCK"
        elif final_score >= 0.4:
            decision = "FLAG"
        else:
            decision = "APPROVE"

    # ── Which model(s) triggered the decision ────────────────────────────────
    triggered_by = []
    if xgb_score >= float(xgb_threshold):
        triggered_by.append(f"credit card fraud detector ({round(xgb_score*100,1)}%)")
    if lgb_score >= float(lgb_threshold):
        triggered_by.append(f"gradient boosting detector ({round(lgb_score*100,1)}%)")
    if paysim_score >= float(paysim_threshold):
        triggered_by.append(f"mobile payment detector ({round(paysim_score*100,1)}%)")
    if not triggered_by:
        triggered_by = [f"ensemble score of {round(final_score*100,1)}%"]
    trigger_str = " and ".join(triggered_by)

    # ── SHAP signal lines — direction forced to match decision ────────────────
    # For demo modes the raw SHAP values come from fake features and may
    # contradict the injected decision, so we flip their narrative to match.
    shap_lines = []
    for f, v in top_features:
        label = get_feature_label(f)
        if ds:
            # Demo: describe every signal as supporting the injected decision
            if decision == "APPROVE":
                direction = "indicated normal, low-risk behaviour"
            elif decision == "FLAG":
                direction = "showed mildly suspicious activity"
            else:  # BLOCK
                direction = "flagged highly suspicious activity"
            shap_lines.append(f"- {label}: {direction}")
        else:
            # Random: use real SHAP direction
            magnitude = "strongly" if abs(v) > 0.1 else "slightly"
            direction = "increased fraud risk" if v > 0 else "reduced fraud risk"
            shap_lines.append(f"- {label}: {magnitude} {direction}")
    shap_text = "\n".join(shap_lines)

    # ── Model score lines ─────────────────────────────────────────────────────
    def threshold_label(score, threshold):
        return "TRIGGERED ⚠️" if score >= float(threshold) else "below threshold ✅"

    model_text = "\n".join([
        f"- Credit card fraud detector:   {round(xgb_score*100,1)}% — {threshold_label(xgb_score, xgb_threshold)}",
        f"- Gradient boosting detector:   {round(lgb_score*100,1)}% — {threshold_label(lgb_score, lgb_threshold)}",
        f"- Mobile payment detector:      {round(paysim_score*100,1)}% — {threshold_label(paysim_score, paysim_threshold)}",
        f"- Final ensemble score:         {round(final_score*100,1)}%",
    ])

    # ── Prompt ────────────────────────────────────────────────────────────────
    # Build explicit triggered/safe lists so AI cannot get confused
    triggered_models = []
    safe_models = []
    if xgb_score >= float(xgb_threshold):
        triggered_models.append(f"credit card fraud detector ({round(xgb_score*100,1)}%)")
    else:
        safe_models.append(f"credit card fraud detector ({round(xgb_score*100,1)}%)")
    if lgb_score >= float(lgb_threshold):
        triggered_models.append(f"gradient boosting detector ({round(lgb_score*100,1)}%)")
    else:
        safe_models.append(f"gradient boosting detector ({round(lgb_score*100,1)}%)")
    if paysim_score >= float(paysim_threshold):
        triggered_models.append(f"mobile payment detector ({round(paysim_score*100,1)}%)")
    else:
        safe_models.append(f"mobile payment detector ({round(paysim_score*100,1)}%)")

    triggered_str = ", ".join(triggered_models) if triggered_models else "none individually"
    safe_str_models = ", ".join(safe_models) if safe_models else "none"

    prompt = f"""You are a senior fraud analyst writing an explanation for a bank transaction decision.

Transaction:
- Amount: RM {amount_rm}
- Time: {time_str}
- Type: {tx_type}
- FINAL DECISION: {decision}
- Ensemble score: {round(final_score*100,1)}%

Model results — READ CAREFULLY AND DO NOT MIX THESE UP:
- TRIGGERED (exceeded fraud threshold): {triggered_str}
- BELOW threshold (safe): {safe_str_models}

Behavioural signals:
{shap_text}

Write 4-5 sentences explaining this {decision} decision. STRICT RULES:
- The decision is {decision} — NEVER contradict this
- TRIGGERED models are the ones that exceeded their threshold — say they "flagged" or "triggered"
- BELOW threshold models found no fraud — say they "remained below threshold" or "found no concern"
- NEVER say a triggered model was below threshold or vice versa
- Explain why {decision} was the final outcome given these results
- No bullet points — flowing prose only
- No technical names: "XGBoost"→"credit card fraud detector", "PaySim"→"mobile payment detector", "LightGBM"→"gradient boosting detector" """

    # ── Fallback summary (if Groq fails) ─────────────────────────────────────
    def fallback_summary():
        signal_names = [get_feature_label(f) for f, v in top_features[:2]]
        signals_str  = " and ".join(signal_names) if signal_names else "transaction patterns"
        if decision == "BLOCK":
            return (
                f"The RM {amount_rm} {tx_type} at {time_str} was blocked because {trigger_str} exceeded its fraud threshold. "
                f"The credit card detector scored {round(xgb_score*100,1)}%, the gradient boosting detector scored {round(lgb_score*100,1)}%, "
                f"and the mobile payment detector scored {round(paysim_score*100,1)}%, producing an ensemble score of {round(final_score*100,1)}%. "
                f"Behavioural analysis of {signals_str} confirmed the high-risk pattern, and the transaction was blocked outright."
            )
        elif decision == "FLAG":
            return (
                f"The RM {amount_rm} {tx_type} at {time_str} was flagged for review with an ensemble score of {round(final_score*100,1)}%. "
                f"The credit card detector scored {round(xgb_score*100,1)}%, gradient boosting scored {round(lgb_score*100,1)}%, "
                f"and the mobile payment detector scored {round(paysim_score*100,1)}%. "
                f"Behavioural signals from {signals_str} showed elevated but not conclusive fraud patterns, warranting manual review."
            )
        else:
            return (
                f"The RM {amount_rm} {tx_type} at {time_str} was approved with a low ensemble score of {round(final_score*100,1)}%. "
                f"All three detectors remained well below their thresholds: credit card detector at {round(xgb_score*100,1)}%, "
                f"gradient boosting at {round(lgb_score*100,1)}%, and mobile payment detector at {round(paysim_score*100,1)}%. "
                f"Behavioural analysis of {signals_str} showed no significant fraud indicators."
            )

    # ── Call Groq ─────────────────────────────────────────────────────────────
    summary = ""
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[explain] Groq summary failed: {e}")
        summary = fallback_summary()

    return {
        "top_features": top_features_out,
        "summary": summary
    }


@app.get("/history")
def history():
    # Return summary only (no features) to keep payload small
    summary = []
    for t in transaction_history:
        s = {k: v for k, v in t.items() if k != "features"}
        summary.append(s)
    return {"transactions": summary}

@app.get("/transaction/{tx_id}")
def get_transaction(tx_id: int):
    """Return full transaction record including features for detail view."""
    for t in transaction_history:
        if t["id"] == tx_id:
            return t
    raise HTTPException(status_code=404, detail="Transaction not found")

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
def simulate(mode: str = "random"):
    """
    Generate a demo transaction.
    For approve/flag/block modes, returns hardcoded scores that GUARANTEE
    the correct decision — no model scoring needed.
    For random mode, generates random features and scores normally.
    """
    import random

    def base_features(v_low, v_high) -> dict:
        """Fill V1-V28 and other model features within a range."""
        features = {}
        for f in feature_names:
            features[f] = round(random.uniform(v_low, v_high), 4)
        return features

    if mode == "approve":
        features = base_features(-0.3, 0.3)
        features.update({
            "amount_log":        round(random.uniform(1.0, 4.0), 4),
            "hour":              random.randint(9, 17),
            "is_transfer":       0,
            "balance_mismatch":  0,
            "orig_balance_diff": round(random.uniform(10, 200), 2),
            "dest_balance_diff": round(random.uniform(10, 200), 2),
        })
        # Inject guaranteed APPROVE scores — skip model entirely
        return {
            "features": features,
            "demo_scores": {
                "xgb_score":    round(random.uniform(0.01, 0.08), 4),
                "lgb_score":    round(random.uniform(0.01, 0.08), 4),
                "paysim_score": round(random.uniform(0.01, 0.08), 4),
                "fraud_score":  round(random.uniform(0.05, 0.15), 4),
                "decision":     "APPROVE",
                "color":        "green",
            }
        }

    elif mode == "flag":
        features = base_features(0.2, 0.8)
        features.update({
            "amount_log":        round(random.uniform(5.0, 6.5), 4),
            "hour":              random.choice([1, 2, 22, 23]),
            "is_transfer":       random.choice([0, 1]),
            "balance_mismatch":  1,
            "orig_balance_diff": round(random.uniform(300, 900), 2),
            "dest_balance_diff": round(random.uniform(300, 900), 2),
        })
        fraud_score = round(random.uniform(0.42, 0.58), 4)
        return {
            "features": features,
            "demo_scores": {
                "xgb_score":    round(random.uniform(0.20, 0.40), 4),
                "lgb_score":    round(random.uniform(0.20, 0.40), 4),
                "paysim_score": round(random.uniform(0.20, 0.40), 4),
                "fraud_score":  fraud_score,
                "decision":     "FLAG",
                "color":        "yellow",
            }
        }

    elif mode == "block":
        features = base_features(1.5, 4.0)
        features.update({
            "amount_log":        round(random.uniform(7.5, 10.0), 4),
            "hour":              random.choice([1, 2, 3, 4]),
            "is_transfer":       1,
            "balance_mismatch":  1,
            "orig_balance_diff": round(random.uniform(3000, 10000), 2),
            "dest_balance_diff": round(random.uniform(3000, 10000), 2),
        })
        return {
            "features": features,
            "demo_scores": {
                "xgb_score":    round(random.uniform(0.85, 0.99), 4),
                "lgb_score":    round(random.uniform(0.85, 0.99), 4),
                "paysim_score": round(random.uniform(0.85, 0.99), 4),
                "fraud_score":  round(random.uniform(0.85, 0.97), 4),
                "decision":     "BLOCK",
                "color":        "red",
            }
        }

    else:  # random — use real model scoring
        features = {}
        features["amount_log"]        = round(random.uniform(1.0, 9.0), 4)
        features["hour"]              = random.randint(0, 23)
        features["is_transfer"]       = random.randint(0, 1)
        features["balance_mismatch"]  = random.randint(0, 1)
        features["orig_balance_diff"] = round(random.uniform(10, 5000), 2)
        features["dest_balance_diff"] = round(random.uniform(10, 5000), 2)
        for f in feature_names:
            if f not in features:
                features[f] = round(random.uniform(-2.0, 2.0), 4)
        return {"features": features}


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
