# 🛡️ FraudShield AI

> **Real-time AI-powered fraud detection for financial transactions**  
> Built for VHack 2026 — Case Study 2: Financial Fraud Detection

[![Live Frontend](https://fraud-shield-beryl.vercel.app/)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Railway-success)](https://fraud-shield-production-d3a8.up.railway.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| **Live Frontend Dashboard** | https://fraud-shield-flame.vercel.app |
| **Live Backend API** | https://fraud-shield-production-d3a8.up.railway.app |
| **Interactive API Docs** | https://fraud-shield-production-d3a8.up.railway.app/docs |

---

## 🚀 What is FraudShield?

FraudShield is a real-time fraud detection system that uses an ensemble of three machine learning models to analyse financial transactions and classify them as **APPROVE**, **FLAG**, or **BLOCK** within milliseconds.

It detects two types of fraud:
- **Credit card fraud** — using anonymised PCA-transformed transaction features (V1–V28)
- **ASEAN mobile money fraud** — targeting TRANSFER and CASH_OUT transactions common in Southeast Asian payment systems like Touch 'n Go, GrabPay, and DuitNow

---

## 🧠 ML Models

### Three-Model Ensemble

| Model | Dataset | Precision | Recall | F1 Score | AUC-ROC | Weight |
|---|---|---|---|---|---|---|
| **XGBoost** | Credit Card (284,807 rows) | 96% | 76% | 85% | 0.9833 | 40% |
| **LightGBM** | Credit Card (284,807 rows) | 88% | 81% | 84% | 0.9865 | 30% |
| **PaySim** | Mobile Money (2.77M rows) | 96% | 78% | 86% | **0.9929** | 30% |

### Decision Logic

```
BLOCK   → if ANY single model score ≥ its individual threshold
FLAG    → if ensemble weighted score ≥ 0.4
APPROVE → otherwise
```

### Thresholds (optimised via precision-recall curve)

| Model | Threshold |
|---|---|
| XGBoost | 0.9928 |
| LightGBM | 0.9874 |
| PaySim | 0.9908 |

### Why Three Models?

- **XGBoost** — high precision, best at avoiding false positives
- **LightGBM** — higher recall, catches more actual fraud
- **PaySim** — purpose-built for ASEAN mobile money patterns (TRANSFER/CASH_OUT)

The ensemble combines their strengths. If any single model is highly confident, the transaction is blocked immediately — we don't wait for consensus when one model is certain.

---

## 🏗️ Architecture

```
Users / Judges
      │
      ▼
┌─────────────────────────────────────┐
│  Vercel (Frontend)                  │
│  fraud-shield-flame.vercel.app      │
│  Dashboard · Scorer · Stats · Chat  │
└─────────────────────────────────────┘
      │  HTTPS fetch()
      ▼
┌─────────────────────────────────────┐
│  Railway Cloud (us-west-2)          │
│  FastAPI Backend                    │
│  ┌─────────┬──────────┬──────────┐  │
│  │XGBoost  │LightGBM  │ PaySim   │  │
│  │  40%    │  30%     │  30%     │  │
│  └────────────────────────────────  │
│         Ensemble Decision           │
│  AI Agent (Groq + Llama 3.3 70B)   │
│  MCP Server (5 tools)               │
└─────────────────────────────────────┘
      │
      ▼
Groq API (external)
```

---

## 🖥️ Frontend Dashboard

Built with vanilla HTML, CSS, and JavaScript — no framework needed. 4 pages:

| Page | Description |
|---|---|
| **Dashboard** | System overview — live stats, pattern alerts, transaction history |
| **Transaction Scorer** | Demo button, fraud gauge, model breakdown, SHAP explanation |
| **Model Performance** | Precision, recall, F1, AUC-ROC for all 3 models |
| **AI Chat** | Conversational AI agent powered by Groq + Llama 3.3 70B |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Score a transaction → APPROVE / FLAG / BLOCK |
| `POST` | `/explain` | SHAP explanation — top 5 fraud features |
| `GET` | `/history` | Last 20 scored transactions |
| `GET` | `/stats` | Model performance metrics |
| `POST` | `/simulate` | Generate a random fake transaction for demo |
| `GET` | `/patterns` | Detect fraud patterns (CARD_TESTING, MASS_FRAUD, etc.) |
| `POST` | `/agent/chat` | Ask AI agent a question in plain English |

### Example: Score a Transaction

```bash
curl -X POST https://fraud-shield-production-d3a8.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"V1": -1.36, "V2": -0.07, "V3": 2.54, "amount_log": 5.02, "is_transfer": 1, "balance_mismatch": 1}}'
```

Response:
```json
{
  "fraud_score": 0.9945,
  "decision": "BLOCK",
  "color": "red",
  "confidence": "99.5%",
  "xgb_score": 0.9941,
  "lgb_score": 0.9932,
  "paysim_score": 0.9961,
  "models_used": "XGBoost + LightGBM + PaySim Ensemble"
}
```

### Example: Ask the AI Agent

```bash
curl -X POST https://fraud-shield-production-d3a8.up.railway.app/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many transactions were blocked today?"}'
```

---

## 🤖 AI Agent

FraudShield includes a conversational AI agent powered by **Groq** (Llama 3.3 70B) that can:

- Answer questions about model performance in plain English
- Analyse account transaction history
- Explain why a transaction was flagged
- Summarise fraud patterns and risk levels

The agent has access to three tools: `get_model_stats`, `get_transaction_history`, and `analyze_account`.

---

## 🔌 MCP Server

FraudShield exposes a **Model Context Protocol (MCP)** server for programmatic integration with AI tools and agents:

| Tool | Description |
|---|---|
| `score_transaction` | Score a transaction and return fraud decision |
| `get_user_history` | Get transaction history for a user |
| `explain_prediction` | Get SHAP feature importance |
| `get_risk_profile` | Get overall risk profile |
| `flag_for_review` | Manually flag a transaction |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Frontend Hosting | Vercel |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Data Balancing | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn |
| AI Agent | Groq API, Llama 3.3 70B |
| MCP | Model Context Protocol |
| Backend Deployment | Railway (Docker) |
| Language | Python 3.11 |

---

## 📊 Datasets

| Dataset | Source | Rows | Use |
|---|---|---|---|
| Credit Card Fraud | [Kaggle — ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 | XGBoost + LightGBM training |
| PaySim Mobile Money | [Kaggle — ealaxi](https://www.kaggle.com/datasets/ealaxi/paysim1) | 2,770,409 | PaySim model training |

> **Note:** Datasets are not included in this repository due to file size. Download them from the Kaggle links above and place them in `notebooks/`.

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- pip

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/Jovygoh/fraud-shield.git
cd fraud-shield/backend

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Start the server
uvicorn main:app --reload
```

Server runs at: `http://127.0.0.1:8000`  
API docs at: `http://127.0.0.1:8000/docs`

### Frontend Setup

```bash
cd fraud-shield/frontend
# Open index.html in browser directly
# OR use Live Server extension in VS Code / Cursor
```

### Get a Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Add it to your `.env` file

### Train Models (Optional)

Pre-trained model files are included in `backend/model/`. To retrain from scratch:

```bash
cd notebooks
jupyter notebook eda.ipynb
```

> Download datasets from Kaggle first (see Datasets section above).

---

## 🚢 Deployment

### Backend — Railway

```bash
cd backend
railway login
railway up
```

The `Dockerfile` handles the `libgomp` system dependency required by LightGBM:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Frontend — Vercel

Push to the `main` branch on GitHub → Vercel auto-deploys instantly. No manual steps needed after initial setup.

---

## 📁 Project Structure

```
fraud-shield/
├── backend/
│   ├── main.py                  # FastAPI — 8 endpoints, 3-model ensemble
│   ├── agent.py                 # Groq AI agent (Llama 3.3 70B)
│   ├── mcp_server.py            # MCP server (5 tools)
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Docker config for Railway
│   ├── Procfile                 # Railway process file
│   └── model/
│       ├── fraud_model.pkl          # XGBoost model
│       ├── lgb_model.pkl            # LightGBM model
│       ├── paysim_model.pkl         # PaySim model
│       ├── feature_names.pkl        # Credit card feature names
│       ├── paysim_feature_names.pkl # PaySim feature names
│       ├── threshold.pkl            # XGBoost threshold (0.9928)
│       ├── lgb_threshold.pkl        # LightGBM threshold (0.9874)
│       └── paysim_threshold.pkl     # PaySim threshold (0.9908)
├── frontend/
│   ├── index.html               # Main HTML structure
│   ├── styles.css               # All styles
│   └── app.js                   # All JavaScript + API calls
├── notebooks/
│   └── eda.ipynb                # Full model training notebook
└── .gitignore
```

---

## 🏆 VHack 2026

Built for **VHack 2026 — Case Study 2: Financial Fraud Detection**

Key highlights for judges:
- ✅ **Live frontend** — https://fraud-shield-flame.vercel.app
- ✅ **Live backend API** — https://fraud-shield-production-d3a8.up.railway.app/docs
- ✅ **3-model ensemble** — XGBoost + LightGBM + PaySim covering both credit card and ASEAN mobile money fraud
- ✅ **Explainable AI** — SHAP values show exactly which features caused each fraud decision
- ✅ **Conversational AI agent** — ask questions in plain English via Groq + Llama 3.3 70B
- ✅ **MCP integration** — production-ready tool interface for enterprise use
- ✅ **96% precision, 0.9929 AUC-ROC** on mobile money fraud detection

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
