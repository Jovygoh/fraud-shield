import asyncio
import json
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── FastAPI base URL ──
API_URL = "http://127.0.0.1:8000"

# ── Create MCP server ──
server = Server("fraudshield-mcp")

# ── Define tools ──
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="score_transaction",
            description="Score a single transaction and return fraud probability, decision (APPROVE/FLAG/BLOCK) and confidence",
            inputSchema={
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Transaction features as key-value pairs"
                    }
                },
                "required": ["features"]
            }
        ),
        Tool(
            name="get_transaction_history",
            description="Get the last 20 transactions processed by the system",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="explain_transaction",
            description="Explain why a transaction was flagged by returning the top features that influenced the decision",
            inputSchema={
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Transaction features as key-value pairs"
                    }
                },
                "required": ["features"]
            }
        ),
        Tool(
            name="get_model_stats",
            description="Get current model performance statistics including precision, recall, F1 and AUC-ROC",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="analyze_account",
            description="Full account risk analysis — scores all recent transactions and returns a risk summary report",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID to analyze"
                    }
                },
                "required": ["user_id"]
            }
        ),
    ]

# ── Tool handlers ──
@server.call_tool()
async def call_tool(name: str, arguments: dict):

    if name == "score_transaction":
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": arguments["features"]}
        )
        result = response.json()
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "get_transaction_history":
        response = requests.get(f"{API_URL}/history")
        result = response.json()
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "explain_transaction":
        response = requests.post(
            f"{API_URL}/explain",
            json={"features": arguments["features"]}
        )
        result = response.json()
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "get_model_stats":
        response = requests.get(f"{API_URL}/stats")
        result = response.json()
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "analyze_account":
        # Step 1 — Get transaction history
        history = requests.get(f"{API_URL}/history").json()
        transactions = history.get("transactions", [])

        # Step 2 — Get model stats
        stats = requests.get(f"{API_URL}/stats").json()

        # Step 3 — Count decisions
        blocked = [t for t in transactions if t["decision"] == "BLOCK"]
        flagged = [t for t in transactions if t["decision"] == "FLAG"]
        approved = [t for t in transactions if t["decision"] == "APPROVE"]

        # Step 4 — Build risk report
        if len(blocked) > 2:
            risk_level = "HIGH"
            recommendation = "Immediately suspend account and contact user"
        elif len(blocked) > 0 or len(flagged) > 3:
            risk_level = "MEDIUM"
            recommendation = "Flag account for manual review"
        else:
            risk_level = "LOW"
            recommendation = "Account looks normal — no action needed"

        report = {
            "user_id": arguments["user_id"],
            "risk_level": risk_level,
            "recommendation": recommendation,
            "summary": {
                "total_transactions": len(transactions),
                "approved": len(approved),
                "flagged": len(flagged),
                "blocked": len(blocked),
            },
            "model_performance": stats
        }

        return [TextContent(
            type="text",
            text=json.dumps(report, indent=2)
        )]

    return [TextContent(type="text", text="Unknown tool")]

# ── Run server ──
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())