"""
Flask Web Dashboard
====================
Serves the federated learning monitoring dashboard with APIs for:
- Triggering federated training
- Viewing real-time training metrics
- Running privacy-accuracy trade-off analysis
- Viewing compliance reports
- Per-hospital metrics
"""

import json
import threading
import os
from flask import Flask, render_template, jsonify, request

import config
from data_generator import generate_all_hospital_data
from federated_engine import setup_federation
from privacy_analysis import run_privacy_sweep, generate_plots
from compliance import ComplianceEngine

app = Flask(__name__)

# ─── Global State ─────────────────────────────────────────
training_state = {
    'status': 'idle',          # idle, generating_data, training, complete, error
    'progress': 0,
    'total_rounds': 0,
    'current_round': 0,
    'metrics': None,
    'server': None,
    'hospital_data': None,
    'privacy_results': None,
    'compliance_report': None,
    'plot_paths': None,
    'error': None,
}
training_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def start_training():
    """
    Start federated training pipeline in background.
    
    Request body (optional):
        - num_rounds: int (default: 30)
        - epsilon: float (default: 1.0)
    """
    with training_lock:
        if training_state['status'] == 'training':
            return jsonify({'error': 'Training already in progress'}), 400

    data = request.get_json(silent=True) or {}
    num_rounds = data.get('num_rounds', config.NUM_ROUNDS)
    epsilon = data.get('epsilon', config.DEFAULT_EPSILON)

    def train_pipeline():
        try:
            with training_lock:
                training_state['status'] = 'generating_data'
                training_state['progress'] = 0
                training_state['error'] = None

            # Step 1: Generate synthetic data
            hospital_data = generate_all_hospital_data()
            training_state['hospital_data'] = hospital_data

            with training_lock:
                training_state['status'] = 'training'
                training_state['total_rounds'] = num_rounds

            # Step 2: Set up federation and train
            server = setup_federation(hospital_data, epsilon=epsilon)
            training_state['server'] = server

            for r in range(num_rounds):
                metrics = server.train_round()
                with training_lock:
                    training_state['current_round'] = r + 1
                    training_state['progress'] = int((r + 1) / num_rounds * 100)
                    training_state['metrics'] = metrics

            # Step 3: Generate compliance report
            compliance_engine = ComplianceEngine()
            compliance_engine.log_event('TRAINING_COMPLETE',
                                         f'Completed {num_rounds} rounds with ε={epsilon}')
            compliance_report = compliance_engine.generate_full_report(server)

            with training_lock:
                training_state['status'] = 'complete'
                training_state['compliance_report'] = compliance_report

        except Exception as e:
            with training_lock:
                training_state['status'] = 'error'
                training_state['error'] = str(e)
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=train_pipeline, daemon=True)
    thread.start()

    return jsonify({
        'message': 'Training started',
        'num_rounds': num_rounds,
        'epsilon': epsilon,
    })


@app.route('/api/status')
def get_status():
    """Get current training status and metrics."""
    with training_lock:
        response = {
            'status': training_state['status'],
            'progress': training_state['progress'],
            'current_round': training_state['current_round'],
            'total_rounds': training_state['total_rounds'],
            'error': training_state['error'],
        }

        if training_state['server']:
            response['summary'] = training_state['server'].get_training_summary()

        if training_state['metrics']:
            m = training_state['metrics']
            response['latest_metrics'] = {
                'round': m['round'],
                'accuracy': round(m['avg_accuracy'], 4),
                'f1': round(m['avg_f1'], 4),
                'auc_roc': round(m['avg_auc_roc'], 4),
                'loss': round(m['avg_loss'], 4),
                'precision': round(m['avg_precision'], 4),
                'recall': round(m['avg_recall'], 4),
            }
            response['client_metrics'] = m.get('client_metrics', [])

    return jsonify(response)


@app.route('/api/privacy-analysis', methods=['POST'])
def run_privacy_analysis():
    """
    Run privacy-accuracy trade-off sweep.
    
    Request body (optional):
        - num_rounds: int (default: 15)
    """
    with training_lock:
        if training_state['status'] == 'training':
            return jsonify({'error': 'Training in progress, please wait'}), 400

    data = request.get_json(silent=True) or {}
    num_rounds = data.get('num_rounds', 15)

    def analysis_pipeline():
        try:
            with training_lock:
                training_state['status'] = 'training'
                training_state['progress'] = 0

            # Generate data if not already done
            if training_state['hospital_data'] is None:
                hospital_data = generate_all_hospital_data()
                training_state['hospital_data'] = hospital_data
            else:
                hospital_data = training_state['hospital_data']

            # Run sweep
            results = run_privacy_sweep(hospital_data, num_rounds=num_rounds)
            plot_paths = generate_plots(results)

            with training_lock:
                training_state['privacy_results'] = results
                training_state['plot_paths'] = plot_paths
                training_state['status'] = 'complete'
                training_state['progress'] = 100

        except Exception as e:
            with training_lock:
                training_state['status'] = 'error'
                training_state['error'] = str(e)
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=analysis_pipeline, daemon=True)
    thread.start()

    return jsonify({'message': 'Privacy analysis started', 'num_rounds': num_rounds})


@app.route('/api/privacy-results')
def get_privacy_results():
    """Get privacy-accuracy sweep results."""
    with training_lock:
        if training_state['privacy_results'] is None:
            return jsonify({'status': 'not_available', 'message': 'Run privacy analysis first'})

        results = training_state['privacy_results']
        return jsonify({
            'status': 'available',
            'epsilon_labels': results['epsilon_labels'],
            'accuracies': results['accuracies'],
            'f1_scores': results['f1_scores'],
            'auc_rocs': results['auc_rocs'],
            'leakage_risks': results['leakage_risks'],
            'attack_success_rates': results['attack_success_rates'],
            'convergence_curves': results['convergence_curves'],
            'per_hospital_metrics': results['per_hospital_metrics'],
            'plot_paths': training_state['plot_paths'],
        })


@app.route('/api/compliance')
def get_compliance():
    """Get compliance report."""
    with training_lock:
        if training_state['compliance_report']:
            return jsonify(training_state['compliance_report'])

    # Generate a fresh report without server context
    engine = ComplianceEngine()
    report = engine.generate_full_report(training_state.get('server'))
    return jsonify(report)


@app.route('/api/hospitals')
def get_hospitals():
    """Get list of hospital nodes and their configurations."""
    hospitals = []
    for h in config.HOSPITALS:
        hosp_info = {
            'id': h['id'],
            'name': h['name'],
            'country': h['country'],
            'flag': h['flag'],
            'num_samples': h['num_samples'],
            'outbreak_rate': h['outbreak_rate'],
        }

        # Add training metrics if available
        if training_state['server']:
            for client in training_state['server'].clients:
                if client.hospital_id == h['id'] and client.metrics_history:
                    latest = client.metrics_history[-1]
                    hosp_info['latest_metrics'] = {
                        'accuracy': round(latest['accuracy'], 4),
                        'f1_score': round(latest['f1_score'], 4),
                        'auc_roc': round(latest['auc_roc'], 4),
                    }

        hospitals.append(hosp_info)

    return jsonify(hospitals)


@app.route('/api/hospital/<int:hospital_id>')
def get_hospital(hospital_id):
    """Get detailed metrics for a specific hospital."""
    hosp_config = None
    for h in config.HOSPITALS:
        if h['id'] == hospital_id:
            hosp_config = h
            break

    if not hosp_config:
        return jsonify({'error': f'Hospital {hospital_id} not found'}), 404

    response = {
        'config': hosp_config,
        'metrics_history': [],
    }

    if training_state['server']:
        for client in training_state['server'].clients:
            if client.hospital_id == hospital_id:
                response['metrics_history'] = client.metrics_history
                response['num_training_samples'] = client.num_samples
                break

    return jsonify(response)


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  FEDERATED HEALTHCARE LEARNING FRAMEWORK")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
