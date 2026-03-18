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
            "description": """Fetch and filter transactions. Use for ANY question about transactions.
            Supports filtering by decision, amount, fraud score, time, and count.
            Examples:
            - 'show blocked transactions'       → decision_filter='BLOCK'
            - 'last 5 transactions'             → limit=5
            - 'transactions in last 10 minutes' → minutes_ago=10
            - 'transactions in last hour'        → minutes_ago=60
            - 'high risk transactions'           → min_score=0.7
            - 'large transactions over RM500'    → min_amount=500
            - 'show all flagged'                 → decision_filter='FLAG'
            - 'blocked transactions above RM200' → decision_filter='BLOCK', min_amount=200
            Always call this before answering any transaction question.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many transactions to return (default 20)"
                    },
                    "decision_filter": {
                        "type": "string",
                        "enum": ["APPROVE", "FLAG", "BLOCK", "ALL"],
                        "description": "Filter by decision. Use ALL to get everything."
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum fraud score 0.0 to 1.0"
                    },
                    "max_score": {
                        "type": "number",
                        "description": "Maximum fraud score 0.0 to 1.0"
                    },
                    "min_amount": {
                        "type": "number",
                        "description": "Minimum transaction amount in RM"
                    },
                    "max_amount": {
                        "type": "number",
                        "description": "Maximum transaction amount in RM"
                    },
                    "minutes_ago": {
                        "type": "integer",
                        "description": "Only return transactions from the last N minutes. Use 60 for last hour, 1440 for last day."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trends",
            "description": """Analyze fraud trends and patterns across all transactions.
            Returns fraud rate, score distribution, amount stats, peak risk periods, and trend direction.
            Use for questions like:
            - 'what are the trends?'
            - 'how is the system performing overall?'
            - 'what patterns do you see?'
            - 'is fraud increasing or decreasing?'
            - 'what is the average fraud score?'""",
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
                    "user_id": {
                        "type": "string",
                        "description": "User ID to analyze"
                    }
                },
                "required": ["user_id"]
            }
        }
    }
]


def parse_timestamp(ts_str):
    """Parse timestamp string from transaction history."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def execute_tool(tool_name, tool_input):

    if tool_name == "get_model_stats":
        response = requests.get(f"{API_URL}/stats")
        return response.json()

    elif tool_name == "get_transactions":
        response = requests.get(f"{API_URL}/history")
        all_txs = response.json().get("transactions", [])

        limit          = tool_input.get("limit", 20)
        decision_filter = tool_input.get("decision_filter", "ALL")
        min_score      = tool_input.get("min_score")
        max_score      = tool_input.get("max_score")
        min_amount     = tool_input.get("min_amount")
        max_amount     = tool_input.get("max_amount")
        minutes_ago    = tool_input.get("minutes_ago")

        filtered = all_txs

        # ── Time filter ──────────────────────────────
        if minutes_ago is not None:
            cutoff = datetime.now() - timedelta(minutes=minutes_ago)
            filtered = [
                t for t in filtered
                if parse_timestamp(t.get("timestamp", "")) is not None
                and parse_timestamp(t["timestamp"]) >= cutoff
            ]

        # ── Decision filter ───────────────────────────
        if decision_filter and decision_filter != "ALL":
            filtered = [t for t in filtered if t["decision"] == decision_filter]

        # ── Score filters ─────────────────────────────
        if min_score is not None:
            filtered = [t for t in filtered if t.get("score", 0) >= min_score]
        if max_score is not None:
            filtered = [t for t in filtered if t.get("score", 0) <= max_score]

        # ── Amount filters (amount field is log scale) ─
        if min_amount is not None:
            filtered = [t for t in filtered
                        if math.exp(t.get("amount", 0)) >= min_amount]
        if max_amount is not None:
            filtered = [t for t in filtered
                        if math.exp(t.get("amount", 0)) <= max_amount]

        # ── Attach human-readable RM amount ───────────
        for t in filtered:
            t["amount_rm"] = round(math.exp(t.get("amount", 0)), 2)

        result = filtered[-limit:]

        return {
            "total_matching": len(filtered),
            "returned": len(result),
            "filters_applied": tool_input,
            "transactions": result
        }

    elif tool_name == "analyze_trends":
        response = requests.get(f"{API_URL}/history")
        all_txs = response.json().get("transactions", [])

        if not all_txs:
            return {"error": "No transactions available yet"}

        scores  = [t.get("score", 0) for t in all_txs]
        amounts = [math.exp(t.get("amount", 0)) for t in all_txs]

        blocked  = [t for t in all_txs if t["decision"] == "BLOCK"]
        flagged  = [t for t in all_txs if t["decision"] == "FLAG"]
        approved = [t for t in all_txs if t["decision"] == "APPROVE"]

        # ── Score buckets ──────────────────────────────
        high_risk   = [t for t in all_txs if t.get("score", 0) >= 0.7]
        medium_risk = [t for t in all_txs if 0.4 <= t.get("score", 0) < 0.7]
        low_risk    = [t for t in all_txs if t.get("score", 0) < 0.4]

        # ── Trend: compare older half vs recent half ───
        mid = len(all_txs) // 2
        older_fraud_rate  = len([t for t in all_txs[:mid]  if t["decision"] == "BLOCK"]) / max(mid, 1)
        recent_fraud_rate = len([t for t in all_txs[mid:]  if t["decision"] == "BLOCK"]) / max(len(all_txs) - mid, 1)

        if recent_fraud_rate > older_fraud_rate + 0.05:
            trend = "increasing ⚠️"
        elif recent_fraud_rate < older_fraud_rate - 0.05:
            trend = "decreasing ✅"
        else:
            trend = "stable"

        # ── Time-of-day analysis ───────────────────────
        hour_buckets = {"midnight_to_6am": 0, "6am_to_12pm": 0,
                        "12pm_to_6pm": 0, "6pm_to_midnight": 0}
        for t in blocked:
            ts = parse_timestamp(t.get("timestamp", ""))
            if ts:
                h = ts.hour
                if h < 6:    hour_buckets["midnight_to_6am"] += 1
                elif h < 12: hour_buckets["6am_to_12pm"] += 1
                elif h < 18: hour_buckets["12pm_to_6pm"] += 1
                else:        hour_buckets["6pm_to_midnight"] += 1

        peak_hour = max(hour_buckets, key=hour_buckets.get)

        return {
            "total_transactions": len(all_txs),
            "decision_breakdown": {
                "approved": len(approved),
                "flagged": len(flagged),
                "blocked": len(blocked),
                "fraud_rate_pct": round(len(blocked) / len(all_txs) * 100, 1)
            },
            "score_distribution": {
                "high_risk_above_0.7": len(high_risk),
                "medium_risk_0.4_to_0.7": len(medium_risk),
                "low_risk_below_0.4": len(low_risk),
                "average_score": round(sum(scores) / len(scores), 4),
                "max_score": round(max(scores), 4),
                "min_score": round(min(scores), 4)
            },
            "amount_stats_rm": {
                "average": round(sum(amounts) / len(amounts), 2),
                "highest": round(max(amounts), 2),
                "lowest":  round(min(amounts), 2)
            },
            "fraud_trend": trend,
            "recent_fraud_rate_pct": round(recent_fraud_rate * 100, 1),
            "older_fraud_rate_pct":  round(older_fraud_rate * 100, 1),
            "peak_fraud_time_window": peak_hour
        }

    elif tool_name == "analyze_account":
        history = requests.get(f"{API_URL}/history").json()
        transactions = history.get("transactions", [])

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
            "user_id":           tool_input.get("user_id", "unknown"),
            "risk_level":        risk_level,
            "recommendation":    recommendation,
            "total_transactions": len(transactions),
            "approved":          len(approved),
            "flagged":           len(flagged),
            "blocked":           len(blocked),
        }

    return {"error": "Unknown tool"}


def run_agent(user_message):
    messages = [
        {
            "role": "system",
            "content": """You are FraudShield AI — an intelligent fraud detection assistant for Malaysian banking.

You have access to real-time transaction data. ALWAYS call a tool before answering — never guess numbers.

Tool selection guide:
- Any question mentioning transactions, amounts, decisions → get_transactions with correct filters
- 'last N transactions' → get_transactions(limit=N)
- 'last hour / last 30 minutes' → get_transactions(minutes_ago=60 or 30)
- 'blocked / flagged / approved' → get_transactions(decision_filter='BLOCK'/'FLAG'/'APPROVE')
- 'above RM X' → get_transactions(min_amount=X)
- 'high risk' → get_transactions(min_score=0.7)
- Trend / pattern / performance questions → analyze_trends
- Model metrics (precision, recall, AUC) → get_model_stats
- You can call multiple tools if needed

Response style:
- Use ✅ approved, ⚠️ flagged, 🚨 blocked
- Always show RM amounts (already provided as amount_rm field)
- Be concise and specific — give real numbers
- If no transactions match the filter, say so clearly"""
        },
        {"role": "user", "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000
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


if __name__ == "__main__":
    run_agent("Show me only the blocked transactions")
    run_agent("Show me transactions in the last 30 minutes")
    run_agent("Any large transactions above RM 500?")
    run_agent("What are the fraud trends?")
    run_agent("Show me the last 5 transactions")
