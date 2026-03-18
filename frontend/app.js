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
  } catch { }
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
    text.innerHTML = `Pattern Status: <strong>${pattern}</strong> — ${data.description || data.message || 'System monitoring active'}`;
  } catch { }
}

async function loadHistory() {
  try {
    const res = await fetch(`${API}/history`);
    const data = await res.json();
    const tbody = document.getElementById('history-table');
    const countEl = document.getElementById('history-count');
    const transactions = data.transactions || [];

    if (transactions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:24px">No transactions yet — run a demo on the Transaction Scorer page</td></tr>';
      if (countEl) countEl.textContent = '';
      return;
    }

    if (countEl) countEl.textContent = `${transactions.length} total`;

    tbody.innerHTML = transactions.slice().reverse().map(tx => {
      const score = (tx.score * 100).toFixed(1);
      const barColor = tx.color || (tx.decision === 'BLOCK' ? 'red' : tx.decision === 'FLAG' ? 'yellow' : 'green');
      const badgeClass = { APPROVE: 'badge-green', FLAG: 'badge-yellow', BLOCK: 'badge-red' }[tx.decision] || 'badge-green';
      const scoreColor = { red: 'var(--red)', yellow: 'var(--yellow)', green: 'var(--green)' }[barColor] || 'var(--muted)';
      const rmAmount = tx.amount ? `RM ${Math.exp(tx.amount).toFixed(2)}` : '—';
      return `<tr>
        <td style="color:var(--muted);font-family:var(--mono)">#${tx.id}</td>
        <td>${tx.timestamp || '—'}</td>
        <td style="font-family:var(--mono)">${rmAmount}</td>
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

async function runDemo(mode = 'random') {
  // Disable all demo buttons while running
  document.querySelectorAll('.demo-btn').forEach(b => b.disabled = true);
  const activeBtn = document.getElementById('demo-btn-' + mode);
  if (activeBtn) activeBtn.innerHTML = '<span class="spinner"></span> SCORING...';

  const errEl = document.getElementById('scorer-error');
  errEl.style.display = 'none';

  try {
    const simRes = await fetch(`${API}/simulate?mode=${mode}`, { method: 'POST' });
    if (!simRes.ok) throw new Error('Simulate failed');
    const simData = await simRes.json();
    lastFeatures = simData.features;

    const predRes = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: lastFeatures })
    });
    if (!predRes.ok) throw new Error('Predict failed');
    const d = await predRes.json();

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

    const score = d.fraud_score * 100;
    const gaugeClass = score > 80 ? 'high' : score > 40 ? 'mid' : 'low';
    const gf = document.getElementById('gauge-fill');
    gf.className = 'gauge-fill ' + gaugeClass;
    gf.style.width = score.toFixed(1) + '%';
    const pctColors = { red: 'var(--red)', yellow: 'var(--yellow)', green: 'var(--green)' };
    const pctColor = pctColors[color] || 'var(--text)';
    document.getElementById('gauge-pct').style.color = pctColor;
    document.getElementById('gauge-pct').textContent = score.toFixed(1) + '%';

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

    const f = lastFeatures;
    const rmAmount = Math.exp(f.amount_log || 0).toFixed(2);
    document.getElementById('d-amount').textContent = `RM ${rmAmount}`;
    document.getElementById('d-type').textContent = f.is_transfer == 1 ? 'Transfer' : 'Purchase';
    document.getElementById('d-time').textContent = f.hour !== undefined ? f.hour + ':00' : '0:00';

    const mismatch = f.balance_mismatch;
    document.getElementById('d-mismatch').textContent = mismatch == 1 ? 'YES' : 'NO';
    document.getElementById('d-mismatch').style.color = mismatch == 1 ? 'var(--red)' : 'var(--green)';

    const container = document.getElementById('shap-container');
    container.innerHTML = `<div style="color:var(--muted);font-size:12px;text-align:center;padding:10px"><span class="spinner"></span> Generating explanation...</div>`;
    await loadExplain(lastFeatures);

  } catch (err) {
    errEl.textContent = 'Error: Could not connect to API. Make sure the backend is running.';
    errEl.style.display = 'block';
  } finally {
    document.querySelectorAll('.demo-btn').forEach(b => b.disabled = false);
    document.getElementById('demo-btn-random').innerHTML  = '🎲 RANDOM';
    document.getElementById('demo-btn-approve').innerHTML = '✅ APPROVE';
    document.getElementById('demo-btn-flag').innerHTML    = '⚠️ FLAG';
    document.getElementById('demo-btn-block').innerHTML   = '🚨 BLOCK';
  }
}

// ── SHAP FEATURE LABELS ───────────────────────────────────────────────────────
const featureLabels = {
  'amount_log':        'Amount',
  'hour':              'Time of Day',
  'is_transfer':       'Transfer Type',
  'balance_mismatch':  'Bal. Mismatch',
  'orig_balance_diff': 'Sender Bal. Δ',
  'dest_balance_diff': 'Receiver Bal. Δ',
};

function getFeatureLabel(name) {
  if (featureLabels[name]) return featureLabels[name];
  const match = name.match(/^V(\d+)$/i);
  if (match) return `Signal ${match[1]}`;
  return name;
}

// ── SHAP EXPLANATION ──────────────────────────────────────────────────────────
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
    const summary = data.summary || '';

    if (topFeatures.length === 0) {
      container.innerHTML = '<div style="color:var(--muted);font-size:12px">No explanation available</div>';
      return;
    }

    // Format multi-paragraph AI analysis
    const formattedSummary = summary
      .replace(/\n\n/g, '</p><p style="margin-top:8px">')
      .replace(/\n/g, '<br>');

    const summaryHtml = summary
      ? `<div style="background:var(--bg3);border:1px solid var(--border2);border-radius:10px;padding:14px 16px;margin-bottom:16px">
          <div style="font-family:var(--mono);font-size:10px;color:var(--accent);margin-bottom:8px;letter-spacing:1.5px;display:flex;align-items:center;gap:6px">
            <span>🤖</span> AI ANALYSIS
          </div>
          <div style="font-size:13px;color:var(--text);line-height:1.7">
            <p>${formattedSummary}</p>
          </div>
        </div>`
      : '';

    const maxVal = Math.max(...topFeatures.map(f => Math.abs(f.importance)));

    const barsHtml = topFeatures.map(f => {
      const absVal   = Math.abs(f.importance);
      const barWidth = maxVal > 0 ? (absVal / maxVal * 100).toFixed(1) : 0;
      const isPositive = f.importance >= 0;
      const barColor  = isPositive ? 'var(--red)'  : 'var(--green)';
      const direction = isPositive ? '▲ fraud'     : '▼ safe';
      const dirColor  = isPositive ? 'var(--red)'  : 'var(--green)';
      const label     = getFeatureLabel(f.feature);

      return `<div class="shap-row" title="Raw SHAP: ${f.importance.toFixed(4)} | Feature: ${f.feature}">
        <div class="shap-name" style="width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:0" title="${f.feature}">${label}</div>
        <div class="shap-track"><div class="shap-fill" style="width:${barWidth}%;background:${barColor}"></div></div>
        <div style="font-family:var(--mono);font-size:11px;color:${dirColor};width:60px;text-align:right;flex-shrink:0">${direction}</div>
      </div>`;
    }).join('');

    const legendHtml = `
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border);display:flex;gap:16px;font-size:11px;color:var(--muted)">
        <span><span style="color:var(--red)">▲ fraud</span> — pushed score higher</span>
        <span><span style="color:var(--green)">▼ safe</span> — pushed score lower</span>
      </div>`;

    container.innerHTML = summaryHtml + barsHtml + legendHtml;

  } catch {
    document.getElementById('shap-container').innerHTML =
      '<div style="color:var(--muted);font-size:12px">Could not load explanation</div>';
  }
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
      const stats     = models[m.key] || {};
      const precision = stats.precision  ?? 0;
      const recall    = stats.recall     ?? 0;
      const f1        = stats.f1_score   ?? 0;
      const auc       = stats.auc_roc    ?? 0;
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

// Fetch with explicit timeout — AI tool calls can take 15–30 seconds
async function fetchWithTimeout(url, options = {}, timeoutMs = 90000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function sendMsg() {
  const input = document.getElementById('chat-input');
  const btn   = document.getElementById('chat-send-btn');
  const q     = input.value.trim();
  if (!q) return;

  const msgs = document.getElementById('chat-messages');
  msgs.innerHTML += `<div class="chat-msg user">${q}</div>`;
  input.value = '';
  input.disabled = true;
  btn.disabled   = true;
  msgs.scrollTop = msgs.scrollHeight;

  // Thinking bubble with animated dots
  const thinkId = 'think-' + Date.now();
  msgs.innerHTML += `
    <div class="chat-msg ai thinking" id="${thinkId}">
      <div class="ai-label">FRAUDSHIELD AI</div>
      <span class="spinner"></span> Thinking<span class="dots"></span>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const dotsEl = document.querySelector(`#${thinkId} .dots`);
  let dotCount = 0;
  const dotsTimer = setInterval(() => {
    dotCount = (dotCount + 1) % 4;
    if (dotsEl) dotsEl.textContent = '.'.repeat(dotCount);
  }, 400);

  try {
    const res = await fetchWithTimeout(
      `${API}/agent/chat`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      },
      90000
    );

    clearInterval(dotsTimer);
    document.getElementById(thinkId)?.remove();

    // ── Parse response body first, then check status ──────────────────────
    // Always try to get the body — even 500s return a JSON detail from FastAPI
    let data;
    try {
      data = await res.json();
    } catch {
      data = null;
    }

    if (!res.ok) {
      // FastAPI error format: { "detail": "Agent error: ..." }
      const detail = data?.detail || `Server error ${res.status}`;
      throw new Error(detail);
    }

    msgs.innerHTML += `
      <div class="chat-msg ai">
        <div class="ai-label">FRAUDSHIELD AI</div>
        ${data.response || 'No response received.'}
      </div>`;

  } catch (err) {
    clearInterval(dotsTimer);
    document.getElementById(thinkId)?.remove();

    let userMsg;
    if (err.name === 'AbortError') {
      userMsg = '⏱️ Request timed out — the AI took longer than 90 seconds. Try a simpler question or check if the backend is under load.';
    } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.message.includes('Load failed')) {
      userMsg = '🔌 Could not reach the backend. Check that the server is running.';
    } else {
      // Show the actual server error message (e.g. "Agent error: ...")
      userMsg = `⚠️ ${err.message}`;
    }

    msgs.innerHTML += `
      <div class="chat-msg ai" style="border-color:rgba(255,61,87,0.3)">
        <div class="ai-label" style="color:var(--red)">ERROR</div>
        ${userMsg}
      </div>`;
  } finally {
    input.disabled = false;
    btn.disabled   = false;
    msgs.scrollTop = msgs.scrollHeight;
    input.focus();
  }
}

// ── INIT ──────────────────────────────────────────────────────────────────────
checkAPI();
loadDashboard();
