"""
Centralized Configuration for Federated Healthcare Learning Framework
=====================================================================
All tunable parameters for federated learning, privacy, model architecture,
hospital nodes, and server settings.
"""

import os

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# ─────────────────────────────────────────────
# Federated Learning Hyperparameters
# ─────────────────────────────────────────────
NUM_ROUNDS = 30                # Number of federated aggregation rounds
NUM_CLIENTS = 5                # Number of hospital nodes
LOCAL_EPOCHS = 3               # Local training epochs per round per client
LOCAL_BATCH_SIZE = 32          # Batch size for local training
LEARNING_RATE = 0.001          # Learning rate for local SGD
MOMENTUM = 0.9                 # SGD momentum
WEIGHT_DECAY = 1e-4            # L2 regularization
CLIENT_FRACTION = 1.0          # Fraction of clients participating per round

# ─────────────────────────────────────────────
# Differential Privacy Settings
# ─────────────────────────────────────────────
DEFAULT_EPSILON = 1.0          # Default privacy budget (lower = more private)
DELTA = 1e-5                   # Failure probability for (ε,δ)-DP
CLIP_NORM = 1.0                # Per-sample gradient clipping norm
NOISE_MULTIPLIER = None        # Auto-calculated from epsilon if None

# Epsilon values for privacy-accuracy sweep
EPSILON_SWEEP = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
EPSILON_LABELS = ['0.1', '0.5', '1.0', '2.0', '5.0', '10.0', '∞ (No DP)']

# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
INPUT_FEATURES = 12            # Number of input features
HIDDEN_LAYERS = [128, 64, 32]  # Hidden layer sizes
DROPOUT_RATE = 0.3             # Dropout probability
USE_BATCH_NORM = True          # Whether to use BatchNorm

# Feature names for the model
FEATURE_NAMES = [
    'age', 'gender', 'region_code', 'symptom_severity',
    'body_temperature', 'contact_count', 'vaccination_status',
    'comorbidity_index', 'population_density', 'travel_history',
    'days_since_symptoms', 'hospitalization_history'
]

# ─────────────────────────────────────────────
# Hospital Node Configurations
# ─────────────────────────────────────────────
HOSPITALS = [
    {
        'id': 0,
        'name': 'Apollo Medical Center',
        'country': 'India',
        'flag': '🇮🇳',
        'num_samples': 4000,
        'outbreak_rate': 0.15,
        'age_mean': 35,
        'age_std': 15,
    },
    {
        'id': 1,
        'name': 'Johns Hopkins Hospital',
        'country': 'USA',
        'flag': '🇺🇸',
        'num_samples': 5000,
        'outbreak_rate': 0.10,
        'age_mean': 45,
        'age_std': 18,
    },
    {
        'id': 2,
        'name': 'Charité Hospital',
        'country': 'Germany',
        'flag': '🇩🇪',
        'num_samples': 3500,
        'outbreak_rate': 0.12,
        'age_mean': 50,
        'age_std': 16,
    },
    {
        'id': 3,
        'name': 'Tokyo Medical University',
        'country': 'Japan',
        'flag': '🇯🇵',
        'num_samples': 3000,
        'outbreak_rate': 0.08,
        'age_mean': 48,
        'age_std': 14,
    },
    {
        'id': 4,
        'name': 'São Paulo General Hospital',
        'country': 'Brazil',
        'flag': '🇧🇷',
        'num_samples': 3500,
        'outbreak_rate': 0.18,
        'age_mean': 32,
        'age_std': 13,
    },
]

# ─────────────────────────────────────────────
# Membership Inference Attack Settings
# ─────────────────────────────────────────────
MIA_SHADOW_MODELS = 3          # Number of shadow models for attack
MIA_TEST_SIZE = 0.3            # Fraction of data for attack testing

# ─────────────────────────────────────────────
# Flask / Dashboard Settings
# ─────────────────────────────────────────────
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# ─────────────────────────────────────────────
# Ensure directories exist
# ─────────────────────────────────────────────
for d in [DATA_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
