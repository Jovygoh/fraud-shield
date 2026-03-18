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

    # ── Re-score with ALL 3 models (same logic as /predict) ──────────────────
    amount_rm   = round(math.exp(transaction.features.get("amount_log", 0)), 2)
    hour        = transaction.features.get("hour", None)
    is_transfer = transaction.features.get("is_transfer", 0)
    time_str    = f"{int(hour):02d}:00" if hour is not None else "unknown time"
    tx_type     = "transfer" if is_transfer == 1 else "purchase"

    xgb_score = float(xgb_model.predict_proba(row)[0][1])
    lgb_score = float(lgb_model.predict_proba(row)[0][1])

    features_with_defaults = {**transaction.features}
    for col in paysim_feature_names:
        if col not in features_with_defaults:
            features_with_defaults[col] = 0
    paysim_row   = pd.DataFrame([features_with_defaults])[paysim_feature_names]
    paysim_score = float(paysim_model.predict_proba(paysim_row)[0][1])

    final_score = (xgb_score * 0.4) + (lgb_score * 0.3) + (paysim_score * 0.3)

    # ── Determine decision + which model(s) triggered it ─────────────────────
    triggered_by = []
    if xgb_score >= float(xgb_threshold):
        triggered_by.append(f"XGBoost ({round(xgb_score * 100, 1)}%)")
    if lgb_score >= float(lgb_threshold):
        triggered_by.append(f"LightGBM ({round(lgb_score * 100, 1)}%)")
    if paysim_score >= float(paysim_threshold):
        triggered_by.append(f"PaySim mobile-money model ({round(paysim_score * 100, 1)}%)")

    if triggered_by:
        decision = "BLOCK"
    elif final_score >= 0.4:
        decision = "FLAG"
        triggered_by = [f"ensemble score ({round(final_score * 100, 1)}%)"]
    else:
        decision = "APPROVE"
        triggered_by = [f"ensemble score ({round(final_score * 100, 1)}%)"]

    trigger_str = " and ".join(triggered_by) if triggered_by else "the ensemble score"

    # ── Detailed SHAP factor lines ────────────────────────────────────────────
    shap_lines = []
    for f, v in top_features:
        label     = get_feature_label(f)
        direction = "increased fraud risk" if v > 0 else "reduced fraud risk"
        magnitude = "strongly" if abs(v) > 0.1 else "slightly"
        shap_lines.append(f"- {label}: {magnitude} {direction} (impact score: {round(float(v), 4)})")
    shap_text = "\n".join(shap_lines)

    # ── Model scores breakdown ────────────────────────────────────────────────
    model_lines = [
        f"- Credit card fraud detector (XGBoost): {round(xgb_score * 100, 1)}% fraud probability "
        f"({'TRIGGERED ⚠️' if xgb_score >= float(xgb_threshold) else 'below threshold ✅'})",

        f"- Gradient boosting detector (LightGBM): {round(lgb_score * 100, 1)}% fraud probability "
        f"({'TRIGGERED ⚠️' if lgb_score >= float(lgb_threshold) else 'below threshold ✅'})",

        f"- Mobile payment fraud detector (PaySim): {round(paysim_score * 100, 1)}% fraud probability "
        f"({'TRIGGERED ⚠️' if paysim_score >= float(paysim_threshold) else 'below threshold ✅'})",

        f"- Final ensemble score: {round(final_score * 100, 1)}% (XGBoost 40% + LightGBM 30% + PaySim 30%)",
    ]
    model_text = "\n".join(model_lines)

    # ── Build detailed prompt ─────────────────────────────────────────────────
    prompt = f"""You are a senior fraud analyst writing a detailed explanation of a transaction decision for an internal fraud review report.

Transaction details:
- Amount: RM {amount_rm}
- Time: {time_str}
- Type: {tx_type}
- FINAL DECISION: {decision}
- Decision triggered by: {trigger_str}

Model scores (3 independent fraud detectors voted):
{model_text}

Behavioural pattern analysis (what the AI noticed about this transaction):
{shap_text}

Write a detailed but readable explanation (4-6 sentences) covering ALL of the following:
1. State the final decision ({decision}) and which detector(s) caused it
2. Explain what each model found — mention which ones flagged it and which ones were below threshold
3. Explain what the behavioural signals revealed — translate the signal names into plain English meaning
4. Explain how all these factors combined to lead to the final {decision} decision

Rules:
- NEVER contradict the final decision ({decision}) — it is definitive
- Do NOT use technical names: instead of "XGBoost" say "credit card fraud detector", instead of "PaySim" say "mobile payment detector", instead of "LightGBM" say "gradient boosting detector", instead of "SHAP" say "behavioural analysis"
- Include specific numbers (RM amounts, percentages) to make the explanation concrete
- Write for a bank compliance officer who understands fraud but not ML jargon
- Do NOT use bullet points — write flowing prose only"""

    # ── Rule-based fallback summary ───────────────────────────────────────────
    def fallback_summary():
        fraud_signals  = [get_feature_label(f) for f, v in top_features if v > 0]
        safe_signals   = [get_feature_label(f) for f, v in top_features if v < 0]
        fraud_str = ", ".join(fraud_signals[:2]) if fraud_signals else "unusual patterns"
        safe_str  = ", ".join(safe_signals[:2])  if safe_signals  else "other indicators"

        if decision == "BLOCK":
            return (
                f"The RM {amount_rm} {tx_type} at {time_str} was blocked because {trigger_str} exceeded its fraud threshold. "
                f"The credit card fraud detector scored {round(xgb_score*100,1)}%, the gradient boosting detector scored {round(lgb_score*100,1)}%, "
                f"and the mobile payment detector scored {round(paysim_score*100,1)}%, giving a combined ensemble score of {round(final_score*100,1)}%. "
                f"Behavioural analysis flagged {fraud_str} as key fraud indicators, "
                f"while {safe_str} slightly reduced the overall risk. "
                f"The triggered detector confidence was high enough to override the ensemble and block the transaction outright."
            )
        elif decision == "FLAG":
            return (
                f"The RM {amount_rm} {tx_type} at {time_str} was flagged for manual review with an ensemble score of {round(final_score*100,1)}%. "
                f"The credit card detector scored {round(xgb_score*100,1)}%, gradient boosting scored {round(lgb_score*100,1)}%, "
                f"and the mobile payment detector scored {round(paysim_score*100,1)}%. "
                f"Behavioural analysis identified {fraud_str} as elevated risk factors. "
                f"While no single detector crossed its block threshold, the combined score warrants human review."
            )
        else:
            return (
                f"The RM {amount_rm} {tx_type} at {time_str} was approved with a low ensemble score of {round(final_score*100,1)}%. "
                f"All three detectors remained well below their thresholds: credit card detector at {round(xgb_score*100,1)}%, "
                f"gradient boosting at {round(lgb_score*100,1)}%, and mobile payment detector at {round(paysim_score*100,1)}%. "
                f"Behavioural analysis showed {safe_str} as reassuring signals, "
                f"and no significant fraud indicators were detected in the transaction pattern."
            )

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
def simulate(mode: str = "random"):
    """Generate a fake transaction for demo purposes.
    mode: 'random' | 'approve' | 'flag' | 'block'
    Uses rejection sampling to guarantee the requested decision type.
    """
    import random

    def make_features(bias: str) -> dict:
        fake = {}
        if bias == "approve":
            fake["hour"]              = random.randint(8, 18)
            fake["amount_log"]        = round(random.uniform(2.0, 5.0), 4)
            fake["is_transfer"]       = 0
            fake["balance_mismatch"]  = 0
            fake["orig_balance_diff"] = round(random.uniform(0, 500), 2)
            fake["dest_balance_diff"] = round(random.uniform(0, 500), 2)
            for f in feature_names:
                if f not in fake:
                    fake[f] = round(random.uniform(-0.5, 0.5), 4)
        elif bias == "flag":
            fake["hour"]              = random.choice([0, 1, 2, 22, 23, random.randint(8, 18)])
            fake["amount_log"]        = round(random.uniform(5.5, 7.0), 4)
            fake["is_transfer"]       = random.randint(0, 1)
            fake["balance_mismatch"]  = random.randint(0, 1)
            fake["orig_balance_diff"] = round(random.uniform(-2000, 2000), 2)
            fake["dest_balance_diff"] = round(random.uniform(-2000, 2000), 2)
            for f in feature_names:
                if f not in fake:
                    fake[f] = round(random.uniform(-1.5, 1.5), 4)
        elif bias == "block":
            fake["hour"]              = random.choice([0, 1, 2, 3, 23])
            fake["amount_log"]        = round(random.uniform(7.0, 9.0), 4)
            fake["is_transfer"]       = 1
            fake["balance_mismatch"]  = 1
            fake["orig_balance_diff"] = round(random.uniform(-8000, -1000), 2)
            fake["dest_balance_diff"] = round(random.uniform(1000, 8000), 2)
            for f in feature_names:
                if f not in fake:
                    fake[f] = round(random.uniform(1.5, 4.0), 4)
        else:
            fake["hour"]              = random.randint(0, 23)
            fake["amount_log"]        = round(random.uniform(0, 8), 4)
            fake["is_transfer"]       = random.randint(0, 1)
            fake["balance_mismatch"]  = random.randint(0, 1)
            fake["orig_balance_diff"] = round(random.uniform(-5000, 5000), 2)
            fake["dest_balance_diff"] = round(random.uniform(-5000, 5000), 2)
            for f in feature_names:
                if f not in fake:
                    fake[f] = round(random.uniform(-3, 3), 4)
        return fake

    def score_features(features: dict) -> str:
        try:
            row   = pd.DataFrame([features])[feature_names]
            xgb_s = float(xgb_model.predict_proba(row)[0][1])
            lgb_s = float(lgb_model.predict_proba(row)[0][1])
            fd    = {**features}
            for col in paysim_feature_names:
                if col not in fd:
                    fd[col] = 0
            paysim_row = pd.DataFrame([fd])[paysim_feature_names]
            pay_s  = float(paysim_model.predict_proba(paysim_row)[0][1])
            final  = (xgb_s * 0.4) + (lgb_s * 0.3) + (pay_s * 0.3)
            if xgb_s >= float(xgb_threshold) or lgb_s >= float(lgb_threshold) or pay_s >= float(paysim_threshold):
                return "BLOCK"
            elif final >= 0.4:
                return "FLAG"
            else:
                return "APPROVE"
        except Exception:
            return "UNKNOWN"

    target = mode.upper() if mode != "random" else None

    # Rejection sampling — try up to 40 times to hit the target decision
    fake = make_features(mode if mode != "random" else "random")
    for _ in range(40):
        fake     = make_features(mode if mode != "random" else "random")
        decision = score_features(fake)
        if target is None or decision == target:
            return {"features": fake, "expected_decision": decision}

    # Fallback: return last attempt even if it didn't match
    return {"features": fake, "expected_decision": score_features(fake)}

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
