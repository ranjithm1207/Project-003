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
};

const API = {
    train: '/api/train',
    status: '/api/status',
    privacyAnalysis: '/api/privacy-analysis',
    privacyResults: '/api/privacy-results',
    compliance: '/api/compliance',
    hospitals: '/api/hospitals',
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


// ═════════════════════════════════════════════════════════════
// Renderers
// ═════════════════════════════════════════════════════════════

function renderHospitals(hospitals) {
    const grid = document.getElementById('hospitalGrid');
    if (!grid) return;

    grid.innerHTML = hospitals.map(h => `
        <div class="hospital-card" id="hospital-${h.id}">
            <div class="hospital-header">
                <span class="hospital-flag">${h.flag}</span>
                <div>
                    <div class="hospital-name">${h.name}</div>
                    <div class="hospital-country">${h.country}</div>
                </div>
            </div>
            <div class="hospital-status ${h.latest_metrics ? 'connected' : 'idle'}">
                <span class="dot"></span>
                ${h.latest_metrics ? 'Trained' : 'Ready'}
            </div>
            <div class="hospital-stats">
                <div class="hospital-stat">
                    <div class="hospital-stat-label">Patients</div>
                    <div class="hospital-stat-value">${h.num_samples.toLocaleString()}</div>
                </div>
                <div class="hospital-stat">
                    <div class="hospital-stat-label">Outbreak Rate</div>
                    <div class="hospital-stat-value">${(h.outbreak_rate * 100).toFixed(0)}%</div>
                </div>
                ${h.latest_metrics ? `
                    <div class="hospital-stat">
                        <div class="hospital-stat-label">Accuracy</div>
                        <div class="hospital-stat-value">${(h.latest_metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div class="hospital-stat">
                        <div class="hospital-stat-label">F1 Score</div>
                        <div class="hospital-stat-value">${h.latest_metrics.f1_score.toFixed(3)}</div>
                    </div>
                ` : `
                    <div class="hospital-stat">
                        <div class="hospital-stat-label">Accuracy</div>
                        <div class="hospital-stat-value">—</div>
                    </div>
                    <div class="hospital-stat">
                        <div class="hospital-stat-label">F1 Score</div>
                        <div class="hospital-stat-value">—</div>
                    </div>
                `}
            </div>
        </div>
    `).join('');
}

function updateHospitalMetrics(clientMetrics) {
    clientMetrics.forEach(cm => {
        const card = document.getElementById(`hospital-${cm.hospital_id}`);
        if (!card) return;

        const status = card.querySelector('.hospital-status');
        if (status) {
            status.className = 'hospital-status connected';
            status.innerHTML = '<span class="dot"></span> Training';
        }
    });
}

function renderCompliance(report) {
    // HIPAA
    const hipaaContainer = document.getElementById('hipaaChecks');
    if (hipaaContainer && report.hipaa) {
        hipaaContainer.innerHTML = report.hipaa.controls.map(c => `
            <div class="compliance-item">
                <div class="compliance-status ${c.status === 'COMPLIANT' ? 'pass' : 'partial'}">
                    ${c.status === 'COMPLIANT' ? '✓' : '~'}
                </div>
                <div>
                    <div class="compliance-req">${c.requirement}</div>
                    <div class="compliance-desc">${c.implementation.substring(0, 120)}...</div>
                </div>
            </div>
        `).join('');

        const hipaaScore = document.getElementById('hipaaScore');
        if (hipaaScore) {
            hipaaScore.textContent = `${report.hipaa.summary.compliance_score}%`;
        }
    }

    // GDPR
    const gdprContainer = document.getElementById('gdprChecks');
    if (gdprContainer && report.gdpr) {
        gdprContainer.innerHTML = report.gdpr.controls.map(c => `
            <div class="compliance-item">
                <div class="compliance-status ${c.status === 'COMPLIANT' ? 'pass' : 'partial'}">
                    ${c.status === 'COMPLIANT' ? '✓' : '~'}
                </div>
                <div>
                    <div class="compliance-req">${c.requirement}</div>
                    <div class="compliance-desc">${c.implementation.substring(0, 120)}...</div>
                </div>
            </div>
        `).join('');

        const gdprScore = document.getElementById('gdprScore');
        if (gdprScore) {
            gdprScore.textContent = `${report.gdpr.summary.compliance_score}%`;
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
