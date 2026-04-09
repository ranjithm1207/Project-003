/**
 * Federated Healthcare Learning Framework
 * Frontend Dashboard Logic
 * 
 * Handles:
 * - Training control and progress monitoring
 * - Real-time metrics polling and display
 * - Chart.js visualizations for privacy-accuracy trade-offs
 * - Hospital node status rendering
 * - Compliance report display
 */

// ═════════════════════════════════════════════════════════════
// State & Configuration
// ═════════════════════════════════════════════════════════════

const state = {
    pollInterval: null,
    isTraining: false,
    charts: {},
    currentEpsilon: 1.0,
    config: {
        epochs: 3,
        batch_size: 32,
        learning_rate: 0.001
    }
};

const API = {
    train: '/api/train',
    status: '/api/status',
    privacyAnalysis: '/api/privacy-analysis',
    privacyResults: '/api/privacy-results',
    compliance: '/api/compliance',
    hospitals: '/api/hospitals',
    dataInsights: '/api/data-insights'
};

const CHART_COLORS = {
    primary: '#00d4ff',
    secondary: '#a855f7',
    success: '#22c55e',
    warning: '#f59e0b',
    danger: '#ef4444',
    muted: '#64748b',
    grid: 'rgba(255,255,255,0.06)',
    text: '#94a3b8',
};

// ═════════════════════════════════════════════════════════════
// Initialization
// ═════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadHospitals();
    loadComplianceReport();
    loadDataInsights();
    initNetworkCanvas();

    // Epsilon slider
    const slider = document.getElementById('epsilonSlider');
    const display = document.getElementById('epsilonDisplay');
    if (slider) {
        slider.addEventListener('input', (e) => {
            state.currentEpsilon = parseFloat(e.target.value);
            display.textContent = `ε = ${state.currentEpsilon}`;
        });
    }
});


// ═════════════════════════════════════════════════════════════
// Training Control
// ═════════════════════════════════════════════════════════════

async function startTraining() {
    const btn = document.getElementById('btnTrain');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Training...';

    try {
        const res = await fetch(API.train, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_rounds: 30,
                epsilon: state.currentEpsilon,
                epochs: state.config.epochs,
                batch_size: state.config.batch_size,
                learning_rate: state.config.learning_rate
            }),
        });
        const data = await res.json();

        if (res.ok) {
            state.isTraining = true;
            showToast('Training started successfully', 'success');
            startPolling();
        } else {
            showToast(data.error || 'Failed to start training', 'error');
            btn.disabled = false;
            btn.innerHTML = '🚀 Start Training';
        }
    } catch (err) {
        showToast('Connection error: ' + err.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '🚀 Start Training';
    }
}

async function startPrivacyAnalysis() {
    const btn = document.getElementById('btnAnalysis');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';

    try {
        const res = await fetch(API.privacyAnalysis, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_rounds: 15 }),
        });
        const data = await res.json();

        if (res.ok) {
            state.isTraining = true;
            showToast('Privacy analysis started — this may take a few minutes', 'success');
            startPolling(true);
        } else {
            showToast(data.error || 'Failed to start analysis', 'error');
            btn.disabled = false;
            btn.innerHTML = '📊 Privacy Analysis';
        }
    } catch (err) {
        showToast('Connection error: ' + err.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '📊 Privacy Analysis';
    }
}


// ═════════════════════════════════════════════════════════════
// Polling & Status Updates
// ═════════════════════════════════════════════════════════════

function startPolling(isAnalysis = false) {
    if (state.pollInterval) clearInterval(state.pollInterval);

    state.pollInterval = setInterval(async () => {
        try {
            const res = await fetch(API.status);
            const data = await res.json();
            updateDashboard(data);

            if (data.status === 'complete') {
                clearInterval(state.pollInterval);
                state.isTraining = false;

                const btnTrain = document.getElementById('btnTrain');
                btnTrain.disabled = false;
                btnTrain.innerHTML = '🚀 Start Training';

                const btnAnalysis = document.getElementById('btnAnalysis');
                btnAnalysis.disabled = false;
                btnAnalysis.innerHTML = '📊 Privacy Analysis';

                showToast('Process completed successfully!', 'success');

                if (isAnalysis) loadPrivacyResults();
                loadHospitals();
                loadComplianceReport();
            } else if (data.status === 'error') {
                clearInterval(state.pollInterval);
                state.isTraining = false;
                showToast('Error: ' + (data.error || 'Unknown'), 'error');

                document.getElementById('btnTrain').disabled = false;
                document.getElementById('btnTrain').innerHTML = '🚀 Start Training';
                document.getElementById('btnAnalysis').disabled = false;
                document.getElementById('btnAnalysis').innerHTML = '📊 Privacy Analysis';
            }
        } catch (err) {
            console.error('Polling error:', err);
        }
    }, 1500);
}

function updateDashboard(data) {
    // Progress bar
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusText = document.getElementById('statusText');

    if (progressFill) {
        progressFill.style.width = `${data.progress}%`;
    }
    if (progressText) {
        progressText.textContent = `${data.progress}%`;
    }
    if (statusText) {
        const statusNames = {
            'idle': '⏸ Idle',
            'generating_data': '🔄 Generating Data...',
            'training': '🧠 Training...',
            'complete': '✅ Complete',
            'error': '❌ Error',
        };
        statusText.textContent = statusNames[data.status] || data.status;
    }

    // Metrics
    if (data.latest_metrics) {
        updateMetric('metricAccuracy', data.latest_metrics.accuracy, '%', 100);
        updateMetric('metricF1', data.latest_metrics.f1, '', 1);
        updateMetric('metricAUC', data.latest_metrics.auc_roc, '', 1);
        updateMetric('metricLoss', data.latest_metrics.loss, '', 1, true);
        updateMetric('metricPrecision', data.latest_metrics.precision, '', 1);
        updateMetric('metricRecall', data.latest_metrics.recall, '', 1);
    }

    // Round info
    if (data.current_round) {
        const roundInfo = document.getElementById('roundInfo');
        if (roundInfo) {
            roundInfo.textContent = `Round ${data.current_round} / ${data.total_rounds}`;
        }
    }

    // Convergence chart
    if (data.summary && data.summary.rounds_history) {
        updateConvergenceChart(data.summary.rounds_history);
    }

    // Client metrics
    if (data.client_metrics && data.client_metrics.length > 0) {
        updateHospitalMetrics(data.client_metrics);
    }

    // Update Logs
    if (data.audit_log) {
        updateLogs(data.audit_log);
    }

    // Refresh Hospitals Data quietly for live telemetry if API is open
    // We already fetch hospitals independently, but we can hit the endpoint to refresh telemetry.
    if(Math.random() > 0.5) { loadHospitalsQuietly(); }

    // Update Beams
    if (data.status === 'training') {
        state.networkActive = true;
    } else {
        state.networkActive = false;
    }
}

function updateLogs(auditLog) {
    const logWindow = document.getElementById('logWindow');
    if (!logWindow) return;
    
    // Only update if there are new logs to prevent constant complete re-renders
    if (state.lastLogLength === auditLog.length) return;
    state.lastLogLength = auditLog.length;

    const currentLogs = logWindow.querySelectorAll('.log-entry:not(.system)').length;
    
    if (auditLog.length > currentLogs) {
        logWindow.innerHTML = '<div class="log-entry system">[System] Connection established.</div>';
        auditLog.forEach(log => {
            const el = document.createElement('div');
            el.className = 'log-entry audit';
            el.innerHTML = `[${log.timestamp}] Node [${log.client_id}] | Signature: <span style="color:var(--text-muted)">${log.signature_prefix}</span> | Integrity: <span class="verify">${log.verified ? 'VERIFIED' : 'FAILED'}</span>`;
            logWindow.appendChild(el);
        });
        logWindow.scrollTop = logWindow.scrollHeight;
    }
}

function updateMetric(elementId, value, suffix = '', multiplier = 1, invert = false) {
    const el = document.getElementById(elementId);
    if (el) {
        const displayed = (value * multiplier).toFixed(multiplier >= 100 ? 1 : 4);
        el.textContent = `${displayed}${suffix}`;
    }
}


// ═════════════════════════════════════════════════════════════
// Charts
// ═════════════════════════════════════════════════════════════

function initCharts() {
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: { color: CHART_COLORS.text, font: { family: "'Inter', sans-serif" } }
            },
        },
        scales: {
            x: {
                grid: { color: CHART_COLORS.grid },
                ticks: { color: CHART_COLORS.text },
            },
            y: {
                grid: { color: CHART_COLORS.grid },
                ticks: { color: CHART_COLORS.text },
            },
        },
    };

    // Convergence Chart
    const convCtx = document.getElementById('convergenceChart');
    if (convCtx) {
        state.charts.convergence = new Chart(convCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Accuracy',
                        data: [],
                        borderColor: CHART_COLORS.primary,
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    },
                    {
                        label: 'F1 Score',
                        data: [],
                        borderColor: CHART_COLORS.secondary,
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    },
                ],
            },
            options: {
                ...defaultOptions,
                plugins: {
                    ...defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Training Convergence',
                        color: CHART_COLORS.text,
                        font: { size: 14, weight: 'bold' },
                    },
                },
                scales: {
                    ...defaultOptions.scales,
                    y: { ...defaultOptions.scales.y, min: 0, max: 1 },
                },
            },
        });
    }

    // Trade-off Chart
    const tradeCtx = document.getElementById('tradeoffChart');
    if (tradeCtx) {
        state.charts.tradeoff = new Chart(tradeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Accuracy',
                        data: [],
                        backgroundColor: 'rgba(0, 212, 255, 0.7)',
                        borderColor: CHART_COLORS.primary,
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                    {
                        label: 'Leakage Risk',
                        data: [],
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: CHART_COLORS.danger,
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                ],
            },
            options: {
                ...defaultOptions,
                plugins: {
                    ...defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Privacy-Accuracy Trade-off',
                        color: CHART_COLORS.text,
                        font: { size: 14, weight: 'bold' },
                    },
                },
                scales: {
                    ...defaultOptions.scales,
                    y: { ...defaultOptions.scales.y, min: 0, max: 1 },
                },
            },
        });
    }

    // Data Insights Charts
    const pvCtx = document.getElementById('patientVolumeChart');
    if (pvCtx) {
        state.charts.patientVolume = new Chart(pvCtx, {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: [CHART_COLORS.primary, '#3b82f6', CHART_COLORS.secondary, '#d946ef', CHART_COLORS.success], borderWidth: 0 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: CHART_COLORS.text } } }
            }
        });
    }

    const orCtx = document.getElementById('outbreakRateChart');
    if (orCtx) {
        state.charts.outbreakRate = new Chart(orCtx, {
            type: 'polarArea',
            data: { labels: [], datasets: [{ data: [], backgroundColor: ['rgba(239, 68, 68, 0.5)', 'rgba(245, 158, 11, 0.5)', 'rgba(16, 185, 129, 0.5)', 'rgba(56, 187, 248, 0.5)', 'rgba(139, 92, 246, 0.5)'], borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: CHART_COLORS.text } } },
                scales: { r: { grid: { color: CHART_COLORS.grid }, ticks: { display: false } } }
            }
        });
    }
}

function updateConvergenceChart(roundsHistory) {
    const chart = state.charts.convergence;
    if (!chart) return;

    chart.data.labels = roundsHistory.map(r => `R${r.round}`);
    chart.data.datasets[0].data = roundsHistory.map(r => r.accuracy);
    chart.data.datasets[1].data = roundsHistory.map(r => r.f1);
    chart.update('none');
}

function updateTradeoffChart(privacyResults) {
    const chart = state.charts.tradeoff;
    if (!chart) return;

    chart.data.labels = privacyResults.epsilon_labels;
    chart.data.datasets[0].data = privacyResults.accuracies;
    chart.data.datasets[1].data = privacyResults.leakage_risks;
    chart.update();
}


// ═════════════════════════════════════════════════════════════
// Data Loaders
// ═════════════════════════════════════════════════════════════

async function loadHospitalsQuietly() {
    try {
        const res = await fetch(API.hospitals);
        const hospitals = await res.json();
        // Update DOM elements for telemetry only
        hospitals.forEach(h => {
            if(h.telemetry) {
                const cpu = document.getElementById(`tel-cpu-${h.id}`);
                const mem = document.getElementById(`tel-mem-${h.id}`);
                const ping = document.getElementById(`tel-ping-${h.id}`);
                if(cpu) cpu.textContent = `${h.telemetry.cpu}%`;
                if(mem) mem.textContent = `${h.telemetry.mem}%`;
                if(ping) ping.textContent = `${h.telemetry.ping}ms`;
            }
        });
    } catch(e) {}
}

async function loadHospitals() {
    try {
        const res = await fetch(API.hospitals);
        const hospitals = await res.json();
        renderHospitals(hospitals);
    } catch (err) {
        console.error('Failed to load hospitals:', err);
    }
}

async function loadPrivacyResults() {
    try {
        const res = await fetch(API.privacyResults);
        const data = await res.json();

        if (data.status === 'available') {
            updateTradeoffChart(data);
            // Show trade-off section
            const section = document.getElementById('tradeoffSection');
            if (section) section.classList.remove('hidden');
        }
    } catch (err) {
        console.error('Failed to load privacy results:', err);
    }
}

async function loadComplianceReport() {
    try {
        const res = await fetch(API.compliance);
        const report = await res.json();
        renderCompliance(report);
    } catch (err) {
        console.error('Failed to load compliance:', err);
    }
}

async function loadDataInsights() {
    try {
        const res = await fetch(API.dataInsights);
        const data = await res.json();
        
        const pvChart = state.charts.patientVolume;
        const orChart = state.charts.outbreakRate;
        if (pvChart && orChart && Array.isArray(data)) {
            pvChart.data.labels = data.map(d => d.name);
            pvChart.data.datasets[0].data = data.map(d => d.patients);
            pvChart.update();

            orChart.data.labels = data.map(d => d.name);
            orChart.data.datasets[0].data = data.map(d => d.outbreak_rate * 100);
            orChart.update();
        }
    } catch (err) {
        console.error('Failed to load data insights:', err);
    }
}


// ═════════════════════════════════════════════════════════════
// Renderers
// ═════════════════════════════════════════════════════════════

function renderHospitals(hospitals) {
    const grid = document.getElementById('hospitalGrid');
    if (!grid) return;

    grid.innerHTML = hospitals.map(h => `
        <div class="h-card" id="hospital-${h.id}">
            <div class="h-head">
                <span class="h-flag">${h.flag}</span>
                <div>
                    <div class="h-name">${h.name}</div>
                    <div class="h-country">${h.country}</div>
                </div>
            </div>
            <div class="h-status ${h.latest_metrics ? 'training' : 'idle'}">
                <span class="dot"></span>
                ${h.latest_metrics ? 'Training' : 'Ready'}
            </div>
            <div class="h-stats">
                <div class="h-stat">
                    <span class="h-stat-label">Patients</span>
                    <span class="h-stat-val">${h.num_samples.toLocaleString()}</span>
                </div>
                <div class="h-stat">
                    <span class="h-stat-label">Outbreak Rate</span>
                    <span class="h-stat-val">${(h.outbreak_rate * 100).toFixed(0)}%</span>
                </div>
                ${h.latest_metrics ? `
                    <div class="h-stat">
                        <span class="h-stat-label">Accuracy</span>
                        <span class="h-stat-val">${(h.latest_metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div class="h-stat">
                        <span class="h-stat-label">F1 Score</span>
                        <span class="h-stat-val">${h.latest_metrics.f1_score.toFixed(3)}</span>
                    </div>
                ` : `
                    <div class="h-stat">
                        <span class="h-stat-label">Accuracy</span>
                        <span class="h-stat-val">—</span>
                    </div>
                    <div class="h-stat">
                        <span class="h-stat-label">F1 Score</span>
                        <span class="h-stat-val">—</span>
                    </div>
                `}
            </div>
            ${h.telemetry ? `
            <div class="telemetry-grid">
                <div class="tel-item"><span class="t-lbl">CPU</span><span class="t-val" id="tel-cpu-${h.id}">${h.telemetry.cpu}%</span></div>
                <div class="tel-item"><span class="t-lbl">MEM</span><span class="t-val" id="tel-mem-${h.id}">${h.telemetry.mem}%</span></div>
                <div class="tel-item"><span class="t-lbl">PING</span><span class="t-val" id="tel-ping-${h.id}">${h.telemetry.ping}ms</span></div>
                <div class="tel-item"><span class="t-lbl">ENC</span><span class="t-val" id="tel-enc-${h.id}">${h.telemetry.enc}</span></div>
            </div>` : ''}
        </div>
    `).join('');
}

function updateHospitalMetrics(clientMetrics) {
    clientMetrics.forEach(cm => {
        const card = document.getElementById(`hospital-${cm.hospital_id}`);
        if (!card) return;

        const status = card.querySelector('.h-status');
        if (status) {
            status.className = 'h-status training';
            status.innerHTML = '<span class="dot"></span> Training';
        }
    });
}

function renderCompliance(report) {
    // HIPAA
    const hipaaContainer = document.getElementById('hipaaChecks');
    if (hipaaContainer && report.hipaa) {
        hipaaContainer.innerHTML = report.hipaa.controls.map(c => `
            <div class="c-item">
                <div class="c-icon ${c.status === 'COMPLIANT' ? 'pass' : 'warn'}">
                    <i class="ti ${c.status === 'COMPLIANT' ? 'ti-check' : 'ti-alert-circle'}"></i>
                </div>
                <div class="c-content">
                    <h4>${c.requirement}</h4>
                    <p>${c.implementation.substring(0, 120)}...</p>
                </div>
            </div>
        `).join('');

        const hipaaScore = document.getElementById('hipaaScore');
        if (hipaaScore) {
            hipaaScore.innerHTML = `${report.hipaa.summary.compliance_score}%`;
        }
    }

    // GDPR
    const gdprContainer = document.getElementById('gdprChecks');
    if (gdprContainer && report.gdpr) {
        gdprContainer.innerHTML = report.gdpr.controls.map(c => `
            <div class="c-item">
                <div class="c-icon ${c.status === 'COMPLIANT' ? 'pass' : 'warn'}">
                    <i class="ti ${c.status === 'COMPLIANT' ? 'ti-check' : 'ti-alert-circle'}"></i>
                </div>
                <div class="c-content">
                    <h4>${c.requirement}</h4>
                    <p>${c.implementation.substring(0, 120)}...</p>
                </div>
            </div>
        `).join('');

        const gdprScore = document.getElementById('gdprScore');
        if (gdprScore) {
            gdprScore.innerHTML = `${report.gdpr.summary.compliance_score}%`;
        }
    }
}


// ═════════════════════════════════════════════════════════════
// Toast Notifications
// ═════════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ═════════════════════════════════════════════════════════════
// Settings & Interactivity
// ═════════════════════════════════════════════════════════════

function toggleSettings() {
    const modal = document.getElementById('settingsModal');
    if (modal) {
        modal.classList.toggle('active');
        // Populate current state into inputs
        document.getElementById('inputEpochs').value = state.config.epochs;
        document.getElementById('inputBatchSize').value = state.config.batch_size;
        document.getElementById('inputLR').value = state.config.learning_rate;
    }
}

function saveSettings() {
    state.config.epochs = parseInt(document.getElementById('inputEpochs').value) || 3;
    state.config.batch_size = parseInt(document.getElementById('inputBatchSize').value) || 32;
    state.config.learning_rate = parseFloat(document.getElementById('inputLR').value) || 0.001;
    
    toggleSettings();
    showToast('Advanced configuration saved!', 'success');
}

function clearLog() {
    const logWindow = document.getElementById('logWindow');
    if (logWindow) {
        logWindow.innerHTML = '<div class="log-entry system">[System] Log cleared by user.</div>';
        state.lastLogLength = 0;
    }
}

async function downloadComplianceReport() {
    try {
        const res = await fetch(API.compliance);
        const report = await res.json();
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `FedHealth_Compliance_Report_${new Date().toISOString().slice(0,10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Report downloaded successfully!', 'success');
    } catch (err) {
        showToast('Error downloading report.', 'error');
    }
}

// ═════════════════════════════════════════════════════════════
// Network Canvas Animation
// ═════════════════════════════════════════════════════════════

function initNetworkCanvas() {
    const canvas = document.getElementById('networkCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Resize
    const resize = () => {
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    };
    window.addEventListener('resize', resize);
    resize();

    const particles = [];
    const createParticle = (x1, y1, x2, y2) => {
        particles.push({
            x: x1, y: y1,
            targetX: x2, targetY: y2,
            progress: 0,
            speed: 0.01 + Math.random() * 0.02,
            size: 2 + Math.random() * 2
        });
    };

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const serverEl = document.querySelector('.arch-server-box');
        const nodeEls = document.querySelectorAll('.arch-nodes .node');
        
        if (!serverEl || nodeEls.length === 0) {
            requestAnimationFrame(draw);
            return;
        }

        const sRect = serverEl.getBoundingClientRect();
        const cRect = canvas.getBoundingClientRect();
        
        const serverX = sRect.left - cRect.left + sRect.width / 2;
        const serverY = sRect.bottom - cRect.top - 10;

        nodeEls.forEach(node => {
            const nRect = node.getBoundingClientRect();
            const nodeX = nRect.left - cRect.left + nRect.width / 2;
            const nodeY = nRect.top - cRect.top;

            // Draw line
            ctx.beginPath();
            ctx.moveTo(serverX, serverY);
            ctx.lineTo(nodeX, nodeY);
            ctx.strokeStyle = state.networkActive ? 'rgba(16, 185, 129, 0.4)' : 'rgba(14, 165, 233, 0.2)';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Spawn particles if active
            if (state.networkActive && Math.random() < 0.05) {
                // Bi-directional flow
                if (Math.random() > 0.5) createParticle(serverX, serverY, nodeX, nodeY);
                else createParticle(nodeX, nodeY, serverX, serverY);
            }
        });

        // Draw and update particles
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.progress += p.speed;
            
            if (p.progress >= 1) {
                particles.splice(i, 1);
                continue;
            }

            const x = p.x + (p.targetX - p.x) * p.progress;
            const y = p.y + (p.targetY - p.y) * p.progress;

            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 212, 255, 0.8)';
            ctx.fill();
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#00d4ff';
        }
        ctx.shadowBlur = 0; // reset

        requestAnimationFrame(draw);
    }
    draw();
}
