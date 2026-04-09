# Secure Federated Learning for Healthcare

## Overview
Prototype FL framework for national healthcare institutions to train disease prediction models without sharing raw patient data. Uses Flower + PyTorch + Opacus (DP).

### Compliance
- **Data Sovereignty**: Local training.
- **Privacy**: DP-SGD (epsilon ~1.0).
- **HIPAA/GDPR**: No central data; audit logs.

### Quick Start
```bash
cd secure_fl_healthcare
pip install -r requirements.txt
python data_generator.py  # Generate synthetic data
python run_centralized.py  # Baseline
python run_federated.py    # FL demo
python plot_results.py     # Trade-off plot
```

### Expected Trade-off
- Centralized: ~92% accuracy, no privacy.
- Federated+DP (eps=1.0): ~85% accuracy, low leakage risk.

See `analysis.md` for details.

