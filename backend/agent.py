from groq import Groq
import json
import os
import requests
from dotenv import load_dotenv

# ── Load API key ──
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── FastAPI base URL ──
API_URL = "https://fraud-shield-production-d3a8.up.railway.app"

# ── Tool definitions ──
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_transaction_history",
            "description": "Get the last 20 transactions processed by the system",
            "parameters": { "type": "object", "properties": {} }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_stats",
            "description": "Get model performance statistics including precision, recall and AUC-ROC",
            "parameters": { "type": "object", "properties": {} }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_account",
            "description": "Full account risk analysis based on transaction history",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": { "type": "string", "description": "User ID to analyze" }
                },
                "required": ["user_id"]
            }
        }
    }
]

# ── Tool executor ──
def execute_tool(tool_name, tool_input):
    if tool_name == "get_transaction_history":
        response = requests.get(f"{API_URL}/history")
        return response.json()

    elif tool_name == "get_model_stats":
        response = requests.get(f"{API_URL}/stats")
        return response.json()

    elif tool_name == "analyze_account":
        history = requests.get(f"{API_URL}/history").json()
        transactions = history.get("transactions", [])
        blocked = [t for t in transactions if t["decision"] == "BLOCK"]
        flagged = [t for t in transactions if t["decision"] == "FLAG"]
        approved = [t for t in transactions if t["decision"] == "APPROVE"]

        if len(blocked) > 2:
            risk_level = "HIGH"
            recommendation = "Immediately suspend account"
        elif len(blocked) > 0 or len(flagged) > 3:
            risk_level = "MEDIUM"
            recommendation = "Flag for manual review"
        else:
            risk_level = "LOW"
            recommendation = "Account looks normal"

        return {
            "user_id": tool_input.get("user_id", "unknown"),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "total_transactions": len(transactions),
            "approved": len(approved),
            "flagged": len(flagged),
            "blocked": len(blocked),
        }

    return {"error": "Unknown tool"}

# ── Agentic loop ──
def run_agent(user_message):
    print(f"\n👤 User: {user_message}")
    print("🤖 Agent thinking...\n")

    messages = [
        {
            "role": "system",
            "content": """You are FraudShield AI — an intelligent fraud detection agent.
            You help analyze transactions and user accounts for suspicious activity.
            Always use your tools to get real data before answering.
            Give clear simple explanations that non-technical users can understand.
            Use emojis: ✅ for safe, ⚠️ for suspicious, 🚨 for fraud."""
        },
        { "role": "user", "content": user_message }
    ]

    # Agentic loop — keeps running until agent stops calling tools
    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000
        )

        message = response.choices[0].message

        # If agent wants to use a tool
        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)

                print(f"🔧 Using tool: {tool_name}")
                result = execute_tool(tool_name, tool_input)
                print(f"📊 Result: {json.dumps(result, indent=2)}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        # Agent is done
        else:
            final = message.content
            print(f"🤖 FraudShield Agent: {final}")
            return final

# ── Test the agent ──
if __name__ == "__main__":
    run_agent("What are the current model performance statistics?")
    run_agent("Analyze account U123 and tell me if it looks suspicious")