from groq import Groq
import json
import os
import requests
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

API_URL = "https://fraud-shield-production-d3a8.up.railway.app"

# ── Small fast model to conserve daily token quota ────────────────────────────
# llama-3.1-8b-instant uses ~3-4x fewer tokens than llama-3.3-70b-versatile
CHAT_MODEL = "llama-3.1-8b-instant"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_stats",
            "description": "Get model performance statistics including precision, recall and AUC-ROC",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": """Fetch and filter transactions. Use for listing/showing transactions only.
            Do NOT use for counting — use analyze_trends for counts and summaries.
            Params: limit (only if user says a specific number), decision_filter (APPROVE/FLAG/BLOCK/ALL),
            min_score, max_score, min_amount, max_amount (RM), minutes_ago.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Only set when user explicitly requests a specific count"},
                    "decision_filter": {"type": "string", "enum": ["APPROVE", "FLAG", "BLOCK", "ALL"]},
                    "min_score": {"type": "number"},
                    "max_score": {"type": "number"},
                    "min_amount": {"type": "number", "description": "Minimum amount in RM"},
                    "max_amount": {"type": "number", "description": "Maximum amount in RM"},
                    "minutes_ago": {"type": "integer", "description": "60=last hour, 1440=last day"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trends",
            "description": """Analyze fraud counts and patterns across ALL transactions.
            Use for: 'how many blocked/flagged/approved?', 'total transactions', 'fraud rate',
            'average score', 'trends', 'patterns', 'is system working well?'
            Always use this for counting — never get_transactions for counts.""",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_account",
            "description": "Full risk analysis for a specific user account by user_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        }
    }
]


def parse_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def execute_tool(tool_name, tool_input):

    if tool_name == "get_model_stats":
        return requests.get(f"{API_URL}/stats").json()

    elif tool_name == "get_transactions":
        all_txs = requests.get(f"{API_URL}/history").json().get("transactions", [])

        limit           = tool_input.get("limit", len(all_txs))
        decision_filter = tool_input.get("decision_filter", "ALL")
        min_score       = tool_input.get("min_score")
        max_score       = tool_input.get("max_score")
        min_amount      = tool_input.get("min_amount")
        max_amount      = tool_input.get("max_amount")
        minutes_ago     = tool_input.get("minutes_ago")

        filtered = all_txs

        if minutes_ago is not None:
            cutoff = datetime.now() - timedelta(minutes=minutes_ago)
            filtered = [t for t in filtered
                        if parse_timestamp(t.get("timestamp", "")) is not None
                        and parse_timestamp(t["timestamp"]) >= cutoff]

        if decision_filter and decision_filter != "ALL":
            filtered = [t for t in filtered if t["decision"] == decision_filter]

        if min_score is not None:
            filtered = [t for t in filtered if t.get("score", 0) >= min_score]
        if max_score is not None:
            filtered = [t for t in filtered if t.get("score", 0) <= max_score]
        if min_amount is not None:
            filtered = [t for t in filtered if math.exp(t.get("amount", 0)) >= min_amount]
        if max_amount is not None:
            filtered = [t for t in filtered if math.exp(t.get("amount", 0)) <= max_amount]

        for t in filtered:
            t["amount_rm"] = round(math.exp(t.get("amount", 0)), 2)

        result = filtered[-limit:]
        return {"total_matching": len(filtered), "returned": len(result), "transactions": result}

    elif tool_name == "analyze_trends":
        all_txs = requests.get(f"{API_URL}/history").json().get("transactions", [])

        if not all_txs:
            return {"error": "No transactions available yet"}

        scores  = [t.get("score", 0) for t in all_txs]
        amounts = [math.exp(t.get("amount", 0)) for t in all_txs]
        blocked  = [t for t in all_txs if t["decision"] == "BLOCK"]
        flagged  = [t for t in all_txs if t["decision"] == "FLAG"]
        approved = [t for t in all_txs if t["decision"] == "APPROVE"]

        mid = len(all_txs) // 2
        older_rate  = len([t for t in all_txs[:mid]  if t["decision"] == "BLOCK"]) / max(mid, 1)
        recent_rate = len([t for t in all_txs[mid:]  if t["decision"] == "BLOCK"]) / max(len(all_txs) - mid, 1)

        if recent_rate > older_rate + 0.05:   trend = "increasing ⚠️"
        elif recent_rate < older_rate - 0.05: trend = "decreasing ✅"
        else:                                  trend = "stable"

        hour_buckets = {"midnight_to_6am": 0, "6am_to_12pm": 0, "12pm_to_6pm": 0, "6pm_to_midnight": 0}
        for t in blocked:
            ts = parse_timestamp(t.get("timestamp", ""))
            if ts:
                h = ts.hour
                if h < 6:    hour_buckets["midnight_to_6am"] += 1
                elif h < 12: hour_buckets["6am_to_12pm"] += 1
                elif h < 18: hour_buckets["12pm_to_6pm"] += 1
                else:        hour_buckets["6pm_to_midnight"] += 1

        return {
            "total_transactions": len(all_txs),
            "decision_breakdown": {
                "approved": len(approved), "flagged": len(flagged), "blocked": len(blocked),
                "fraud_rate_pct": round(len(blocked) / len(all_txs) * 100, 1)
            },
            "score_distribution": {
                "high_risk_above_0.7": len([t for t in all_txs if t.get("score", 0) >= 0.7]),
                "medium_risk_0.4_to_0.7": len([t for t in all_txs if 0.4 <= t.get("score", 0) < 0.7]),
                "low_risk_below_0.4": len([t for t in all_txs if t.get("score", 0) < 0.4]),
                "average_score": round(sum(scores) / len(scores), 4),
            },
            "amount_stats_rm": {
                "average": round(sum(amounts) / len(amounts), 2),
                "highest": round(max(amounts), 2),
            },
            "fraud_trend": trend,
            "peak_fraud_time_window": max(hour_buckets, key=hour_buckets.get)
        }

    elif tool_name == "analyze_account":
        transactions = requests.get(f"{API_URL}/history").json().get("transactions", [])
        blocked  = [t for t in transactions if t["decision"] == "BLOCK"]
        flagged  = [t for t in transactions if t["decision"] == "FLAG"]
        approved = [t for t in transactions if t["decision"] == "APPROVE"]

        if len(blocked) > 2:
            risk_level     = "HIGH 🚨"
            recommendation = "Immediately suspend account and contact user"
        elif len(blocked) > 0 or len(flagged) > 3:
            risk_level     = "MEDIUM ⚠️"
            recommendation = "Flag account for manual review"
        else:
            risk_level     = "LOW ✅"
            recommendation = "Account looks normal — no action needed"

        return {
            "user_id": tool_input.get("user_id", "unknown"),
            "risk_level": risk_level, "recommendation": recommendation,
            "total_transactions": len(transactions),
            "approved": len(approved), "flagged": len(flagged), "blocked": len(blocked),
        }

    return {"error": "Unknown tool"}


def _friendly_rate_limit_msg(error_str: str) -> str:
    """Extract wait time from Groq 429 message and return a friendly string."""
    import re
    match = re.search(r'try again in ([\d]+m[\d.]+s|[\d.]+s|[\d]+m)', error_str)
    wait  = match.group(1) if match else "a few minutes"
    return (
        f"⏳ The AI is temporarily unavailable — the daily free-tier token limit has been reached. "
        f"Please try again in **{wait}**.\n\n"
        f"_(Tip: the limit resets every 24 hours. "
        f"Upgrade at console.groq.com for higher limits.)_"
    )


def run_agent(user_message):
    from groq import RateLimitError

    messages = [
        {
            "role": "system",
            "content": """You are FraudShield AI, a fraud detection assistant for Malaysian banking.
Always call a tool before answering. Never guess numbers.

Tool routing:
- COUNT questions (how many blocked/flagged/approved, fraud rate, totals) → analyze_trends
- LIST/SHOW transactions → get_transactions (only set limit if user says a number)
- Model metrics (precision, recall, AUC) → get_model_stats
- Specific user account → analyze_account

Style: use ✅ approved, ⚠️ flagged, 🚨 blocked. Show RM amounts. Be concise."""
        },
        {"role": "user", "content": user_message}
    ]

    try:
        while True:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=600
            )

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    tool_name  = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    result     = execute_tool(tool_name, tool_input)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                return message.content

    except RateLimitError as e:
        # Return a friendly message instead of crashing with a 500
        return _friendly_rate_limit_msg(str(e))
