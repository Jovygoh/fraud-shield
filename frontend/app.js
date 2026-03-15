const API = 'https://fraud-shield-production-d3a8.up.railway.app';

// ── NAV ──────────────────────────────────────────────────────────────────────
function showPage(id, tab) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  tab.classList.add('active');
  if (id === 'dashboard') loadDashboard();
  if (id === 'performance') loadStats();
}

// ── API STATUS CHECK ──────────────────────────────────────────────────────────
async function checkAPI() {
  try {
    const res = await fetch(`${API}/`);
    if (res.ok) {
      document.getElementById('api-dot').className = 'nav-dot';
      document.getElementById('api-status').textContent = 'API Live';
    } else {
      throw new Error();
    }
  } catch {
    document.getElementById('api-dot').className = 'nav-dot red';
    document.getElementById('api-status').textContent = 'API Offline';
  }
}

// ── DASHBOARD ────────────────────────────────────────────────────────────────
async function loadDashboard() {
  loadStats_mini();
  loadHistory();
  loadPatterns();
}

async function loadStats_mini() {
  try {
    const res = await fetch(`${API}/stats`);
    const data = await res.json();
    document.getElementById('stat-total').textContent = data.total_transactions ?? 0;
    document.getElementById('stat-approved').textContent = data.approved ?? 0;
    document.getElementById('stat-flagged').textContent = data.flagged ?? 0;
    document.getElementById('stat-blocked').textContent = data.fraud_blocked ?? 0;
  } catch {}
}

async function loadPatterns() {
  try {
    const res = await fetch(`${API}/patterns`);
    const data = await res.json();
    const banner = document.getElementById('pattern-banner');
    const text = document.getElementById('pattern-text');
    const risk = document.getElementById('pattern-risk');
    const pattern = data.pattern || 'NORMAL';
    const riskLevel = data.risk || 'LOW';
    risk.textContent = riskLevel;
    const classMap = {
      NORMAL: 'normal',
      SUSPICIOUS_ACTIVITY: 'warning',
      CARD_TESTING: 'danger',
      MASS_FRAUD: 'danger'
    };
    banner.className = 'banner ' + (classMap[pattern] || 'normal');
    text.innerHTML = `Pattern Status: <strong>${pattern}</strong> — ${data.message || 'System monitoring active'}`;
  } catch {}
}

async function loadHistory() {
  try {
    const res = await fetch(`${API}/history`);
    const data = await res.json();
    const tbody = document.getElementById('history-table');
    const transactions = data.transactions || [];
    if (transactions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:24px">No transactions yet — run a demo on the Transaction Scorer page</td></tr>';
      return;
    }
    tbody.innerHTML = transactions.slice().reverse().map(tx => {
      const score = (tx.score * 100).toFixed(1);
      const barColor = tx.color || (tx.decision === 'BLOCK' ? 'red' : tx.decision === 'FLAG' ? 'yellow' : 'green');
      const badgeClass = { APPROVE: 'badge-green', FLAG: 'badge-yellow', BLOCK: 'badge-red' }[tx.decision] || 'badge-green';
      const scoreColor = { red: 'var(--red)', yellow: 'var(--yellow)', green: 'var(--green)' }[barColor] || 'var(--muted)';
      return `<tr>
        <td style="color:var(--muted);font-family:var(--mono)">#${tx.id}</td>
        <td>${tx.timestamp || '—'}</td>
        <td style="font-family:var(--mono)">${tx.amount ? tx.amount.toFixed(2) : '—'}</td>
        <td><div class="mini-bar-wrap">
          <div class="mini-bar"><div class="mini-bar-fill ${barColor}" style="width:${score}%"></div></div>
          <div class="mini-score" style="color:${scoreColor}">${score}%</div>
        </div></td>
        <td><span class="badge ${badgeClass}">${tx.decision}</span></td>
      </tr>`;
    }).join('');
  } catch {
    document.getElementById('history-table').innerHTML =
      '<tr><td colspan="5" style="text-align:center;color:var(--muted)">Could not load history</td></tr>';
  }
}

// ── TRANSACTION SCORER ────────────────────────────────────────────────────────
let lastFeatures = null;

async function runDemo() {
  const btn = document.getElementById('demo-btn');
  const errEl = document.getElementById('scorer-error');
  errEl.style.display = 'none';
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> SCORING TRANSACTION...';

  try {
    // Step 1: simulate
    const simRes = await fetch(`${API}/simulate`, { method: 'POST' });
    if (!simRes.ok) throw new Error('Simulate failed');
    const simData = await simRes.json();
    lastFeatures = simData.features;

    // Step 2: predict
    const predRes = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: lastFeatures })
    });
    if (!predRes.ok) throw new Error('Predict failed');
    const d = await predRes.json();

    // Step 3: decision block
    const block = document.getElementById('decision-block');
    const color = d.color || (d.decision === 'BLOCK' ? 'red' : d.decision === 'FLAG' ? 'yellow' : 'green');
    block.className = 'decision-block ' + color;
    document.getElementById('decision-word').textContent = d.decision;
    const subs = {
      BLOCK: 'Transaction blocked — fraud score exceeds threshold',
      APPROVE: 'Transaction approved — all models below threshold',
      FLAG: 'Transaction flagged for manual review'
    };
    document.getElementById('decision-sub').textContent = subs[d.decision] || '';

    // Step 4: gauge
    const score = d.fraud_score * 100;
    const gaugeClass = score > 80 ? 'high' : score > 40 ? 'mid' : 'low';
    const gf = document.getElementById('gauge-fill');
    gf.className = 'gauge-fill ' + gaugeClass;
    gf.style.width = score.toFixed(1) + '%';
    const pctColors = { red: 'var(--red)', yellow: 'var(--yellow)', green: 'var(--green)' };
    const pctColor = pctColors[color] || 'var(--text)';
    document.getElementById('gauge-pct').style.color = pctColor;
    document.getElementById('gauge-pct').textContent = score.toFixed(1) + '%';

    // Step 5: model bars
    const xgb = (d.xgb_score || 0) * 100;
    const lgb = (d.lgb_score || 0) * 100;
    const psm = (d.paysim_score || 0) * 100;
    document.getElementById('xgb-bar').style.width = xgb.toFixed(1) + '%';
    document.getElementById('xgb-bar').style.background = pctColor;
    document.getElementById('xgb-pct').textContent = xgb.toFixed(1) + '%';
    document.getElementById('lgb-bar').style.width = lgb.toFixed(1) + '%';
    document.getElementById('lgb-bar').style.background = pctColor;
    document.getElementById('lgb-pct').textContent = lgb.toFixed(1) + '%';
    document.getElementById('psm-bar').style.width = psm.toFixed(1) + '%';
    document.getElementById('psm-bar').style.background = pctColor;
    document.getElementById('psm-pct').textContent = psm.toFixed(1) + '%';
    document.getElementById('models-used').textContent = d.models_used || 'XGBoost 40% + LightGBM 30% + PaySim 30%';

    // Step 6: transaction details
    // FIX: Use explicit == 1 checks to handle 0 as a valid falsy value
    const f = lastFeatures;
    document.getElementById('d-amount').textContent = f.amount_log ? f.amount_log.toFixed(2) : '0.00';

    // FIX: f.is_transfer == 0 is falsy in JS, so we check == 1 explicitly
    document.getElementById('d-type').textContent = f.is_transfer == 1 ? 'Transfer' : 'Purchase';

    document.getElementById('d-time').textContent = f.hour !== undefined ? f.hour + ':00' : '0:00';

    // FIX: f.balance_mismatch == 0 is falsy in JS, so we check == 1 explicitly
    const mismatch = f.balance_mismatch;
    document.getElementById('d-mismatch').textContent = mismatch == 1 ? 'YES' : 'NO';
    document.getElementById('d-mismatch').style.color = mismatch == 1 ? 'var(--red)' : 'var(--green)';

    // Step 7: SHAP explanation
    loadExplain(lastFeatures);

  } catch (err) {
    errEl.textContent = 'Error: Could not connect to API. Make sure the backend is running.';
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶  RUN DEMO TRANSACTION';
  }
}

async function loadExplain(features) {
  try {
    const res = await fetch(`${API}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    const data = await res.json();
    const container = document.getElementById('shap-container');
    const topFeatures = data.top_features || [];
    if (topFeatures.length === 0) {
      container.innerHTML = '<div style="color:var(--muted);font-size:12px">No explanation available</div>';
      return;
    }
    container.innerHTML = topFeatures.map(f => {
      const pct = (f.importance * 100).toFixed(1);
      return `<div class="shap-row">
        <div class="shap-name">${f.feature}</div>
        <div class="shap-track"><div class="shap-fill" style="width:${pct}%"></div></div>
        <div class="shap-pct">${pct}%</div>
      </div>`;
    }).join('');
  } catch {}
}

// ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res = await fetch(`${API}/stats`);
    const data = await res.json();
    const models = data.models || {};
    const grid = document.getElementById('stats-grid');

    const modelDefs = [
      { key: 'xgboost',  label: 'XGBOOST',  sub: 'Credit card · 284K rows · 40% weight' },
      { key: 'lightgbm', label: 'LIGHTGBM', sub: 'Credit card · 284K rows · 30% weight' },
      { key: 'paysim',   label: 'PAYSIM',   sub: 'Mobile money · 2.77M rows · 30% weight' }
    ];

    grid.innerHTML = modelDefs.map(m => {
      const stats = models[m.key] || {};
      const precision = stats.precision ?? 0;
      const recall    = stats.recall ?? 0;
      const f1        = stats.f1_score ?? 0;
      const auc       = stats.auc_roc ?? 0;
      const isTopAuc  = m.key === 'paysim';
      return `<div class="model-card">
        <div class="model-card-header">
          <div class="model-card-title">${m.label}</div>
          <div class="model-card-sub">${m.sub}</div>
        </div>
        <div class="model-card-body">
          <div class="metric-row">
            <div class="metric-name">Precision</div>
            <div class="metric-bar-wrap">
              <div class="metric-bar-t"><div class="metric-bar-f" style="width:${(precision * 100).toFixed(0)}%"></div></div>
            </div>
            <div class="metric-val ${precision >= 0.95 ? 'highlight' : ''}">${(precision * 100).toFixed(0)}%</div>
          </div>
          <div class="metric-row">
            <div class="metric-name">Recall</div>
            <div class="metric-bar-wrap">
              <div class="metric-bar-t"><div class="metric-bar-f" style="width:${(recall * 100).toFixed(0)}%"></div></div>
            </div>
            <div class="metric-val">${(recall * 100).toFixed(0)}%</div>
          </div>
          <div class="metric-row">
            <div class="metric-name">F1 Score</div>
            <div class="metric-bar-wrap">
              <div class="metric-bar-t"><div class="metric-bar-f" style="width:${(f1 * 100).toFixed(0)}%"></div></div>
            </div>
            <div class="metric-val">${(f1 * 100).toFixed(0)}%</div>
          </div>
          <div class="metric-row">
            <div class="metric-name">AUC-ROC</div>
            <div class="metric-bar-wrap">
              <div class="metric-bar-t"><div class="metric-bar-f" style="width:${(auc * 100).toFixed(1)}%"></div></div>
            </div>
            <div class="metric-val ${isTopAuc ? 'highlight' : ''}">${auc.toFixed(4)}${isTopAuc ? ' ⭐' : ''}</div>
          </div>
        </div>
      </div>`;
    }).join('');
  } catch {
    document.getElementById('stats-grid').innerHTML =
      '<div class="card" style="text-align:center;color:var(--muted);padding:40px;grid-column:1/-1">Could not load model statistics</div>';
  }
}

// ── AI CHAT ───────────────────────────────────────────────────────────────────
function sendQuick(q) {
  document.getElementById('chat-input').value = q;
  sendMsg();
}

async function sendMsg() {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('chat-send-btn');
  const q = input.value.trim();
  if (!q) return;

  const msgs = document.getElementById('chat-messages');
  msgs.innerHTML += `<div class="chat-msg user">${q}</div>`;
  input.value = '';
  input.disabled = true;
  btn.disabled = true;
  msgs.scrollTop = msgs.scrollHeight;

  const thinkId = 'think-' + Date.now();
  msgs.innerHTML += `<div class="chat-msg ai thinking" id="${thinkId}"><div class="ai-label">FRAUDSHIELD AI</div><span class="spinner"></span> Thinking...</div>`;
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch(`${API}/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    document.getElementById(thinkId).remove();
    msgs.innerHTML += `<div class="chat-msg ai"><div class="ai-label">FRAUDSHIELD AI</div>${data.response || 'No response received.'}</div>`;
  } catch {
    document.getElementById(thinkId).remove();
    msgs.innerHTML += `<div class="chat-msg ai"><div class="ai-label">FRAUDSHIELD AI</div>Sorry, I could not connect to the AI service. Please check that the backend is running.</div>`;
  } finally {
    input.disabled = false;
    btn.disabled = false;
    msgs.scrollTop = msgs.scrollHeight;
    input.focus();
  }
}

// ── INIT ──────────────────────────────────────────────────────────────────────
checkAPI();
loadDashboard();