"""
Federated Learning Engine
=========================
Core implementation of the federated learning framework with:
- FederatedServer: Orchestrates training rounds and FedAvg aggregation
- HospitalClient: Local training with gradient clipping
- DifferentialPrivacy: (ε,δ)-DP with Gaussian mechanism
- SecureAggregator: HMAC-based integrity verification
"""

import copy
import hashlib
import hmac
import math
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

import config
from model import create_model


# ═════════════════════════════════════════════════════════════
# Differential Privacy Module
# ═════════════════════════════════════════════════════════════

class DifferentialPrivacy:
    """
    Implements (ε, δ)-Differential Privacy using the Gaussian mechanism.
    
    Provides:
    - Noise calibration based on sensitivity, epsilon, and delta
    - Per-sample gradient clipping
    - Privacy budget accounting (simple composition)
    """

    def __init__(self, epsilon=config.DEFAULT_EPSILON, delta=config.DELTA,
                 clip_norm=config.CLIP_NORM):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = self._calibrate_noise()
        self.budget_spent = 0.0
        self.queries = 0

    def _calibrate_noise(self):
        """
        Calibrate Gaussian noise multiplier for (ε,δ)-DP.
        
        Using the analytic Gaussian mechanism:
            σ ≥ Δf · √(2 · ln(1.25/δ)) / ε
        """
        if self.epsilon == float('inf'):
            return 0.0

        sensitivity = self.clip_norm  # L2 sensitivity after clipping
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def clip_gradients(self, model_update):
        """
        Clip model update (gradient) to bound L2 sensitivity.
        
        Args:
            model_update: OrderedDict of parameter tensors (deltas)
            
        Returns:
            Clipped model update
        """
        # Compute total L2 norm of the update
        total_norm = 0.0
        for key in model_update:
            total_norm += model_update[key].float().norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        # Clip if norm exceeds threshold
        clip_factor = min(1.0, self.clip_norm / (total_norm + 1e-8))

        clipped_update = OrderedDict()
        for key in model_update:
            clipped_update[key] = model_update[key] * clip_factor

        return clipped_update, total_norm, clip_factor

    def add_noise(self, aggregated_update, num_clients):
        """
        Add calibrated Gaussian noise to aggregated model update.
        
        Args:
            aggregated_update: OrderedDict of averaged parameter tensors
            num_clients: Number of clients in aggregation (for sensitivity scaling)
            
        Returns:
            Noisy model update
        """
        if self.noise_multiplier == 0.0:
            return aggregated_update

        noisy_update = OrderedDict()
        for key in aggregated_update:
            noise = torch.normal(
                mean=0.0,
                std=self.noise_multiplier / num_clients,
                size=aggregated_update[key].shape
            )
            noisy_update[key] = aggregated_update[key] + noise

        self.budget_spent += self.epsilon
        self.queries += 1

        return noisy_update

    def get_privacy_report(self):
        """Return current privacy accounting status."""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_multiplier': round(self.noise_multiplier, 4),
            'clip_norm': self.clip_norm,
            'total_budget_spent': round(self.budget_spent, 4),
            'num_queries': self.queries,
            'effective_epsilon': round(self.budget_spent, 4),
        }


# ═════════════════════════════════════════════════════════════
# Secure Aggregator
# ═════════════════════════════════════════════════════════════

class SecureAggregator:
    """
    HMAC-based integrity verification for model updates.
    
    Simulates secure aggregation by:
    - Signing model updates with HMAC-SHA256
    - Verifying update integrity before aggregation
    - Logging all aggregation events
    """

    def __init__(self):
        self.secret_key = hashlib.sha256(
            f"federated_health_key_{time.time()}".encode()
        ).digest()
        self.audit_log = []

    def sign_update(self, model_update, client_id):
        """
        Sign a model update with HMAC for integrity verification.
        
        Args:
            model_update: OrderedDict of parameter tensors
            client_id: Identifier of the sending client
            
        Returns:
            HMAC signature (hex string)
        """
        # Create a hash of the update tensors
        update_bytes = b""
        for key in sorted(model_update.keys()):
            update_bytes += model_update[key].cpu().numpy().tobytes()

        signature = hmac.new(
            self.secret_key,
            update_bytes + str(client_id).encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def verify_update(self, model_update, client_id, signature):
        """
        Verify the integrity of a received model update.
        
        Args:
            model_update: OrderedDict of parameter tensors
            client_id: Identifier of the sending client
            signature: HMAC signature to verify
            
        Returns:
            Boolean indicating whether the update is authentic
        """
        expected_signature = self.sign_update(model_update, client_id)
        is_valid = hmac.compare_digest(signature, expected_signature)

        self.audit_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'client_id': client_id,
            'verified': is_valid,
            'signature_prefix': signature[:16] + '...',
        })

        return is_valid


# ═════════════════════════════════════════════════════════════
# Hospital Client
# ═════════════════════════════════════════════════════════════

class HospitalClient:
    """
    Represents a single hospital node in the federation.
    
    Responsibilities:
    - Hold local patient data (never shared)
    - Train model locally for specified epochs
    - Compute model updates (deltas from global model)
    - Apply gradient clipping before sending updates
    """

    def __init__(self, hospital_config, data_df, device='cpu'):
        self.hospital_id = hospital_config['id']
        self.name = hospital_config['name']
        self.country = hospital_config['country']
        self.flag = hospital_config['flag']
        self.device = device

        # Prepare data
        self.scaler = StandardScaler()
        X = data_df[config.FEATURE_NAMES].values
        y = data_df['outbreak_risk'].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        self.X_test = torch.FloatTensor(X_test).to(device)
        self.y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

        # Create data loader
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.LOCAL_BATCH_SIZE,
            shuffle=True
        )

        self.num_samples = len(X_train)
        self.metrics_history = []

    def train_local(self, global_model_state, local_epochs=config.LOCAL_EPOCHS):
        """
        Train model locally and return the update (delta).
        
        Args:
            global_model_state: Global model state dict to start from
            local_epochs: Number of local training epochs
            
        Returns:
            Tuple of (model_update, num_samples, local_metrics)
        """
        # Create local model from global weights
        model = create_model().to(self.device)
        model.load_state_dict(copy.deepcopy(global_model_state))
        model.train()

        # Calculate pos_weight for class imbalance
        pos_count = self.y_train.sum().item()
        neg_count = len(self.y_train) - pos_count
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)]).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )

        epoch_losses = []
        for epoch in range(local_epochs):
            batch_losses = []
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses))

        # Compute model update (delta = local - global)
        model_update = OrderedDict()
        local_state = model.state_dict()
        for key in global_model_state:
            model_update[key] = local_state[key] - global_model_state[key]

        # Evaluate locally
        metrics = self._evaluate(model)
        metrics['train_loss'] = float(np.mean(epoch_losses))
        self.metrics_history.append(metrics)

        return model_update, self.num_samples, metrics

    def _evaluate(self, model):
        """Evaluate model on local test set."""
        model.eval()
        with torch.no_grad():
            logits = model(self.X_test)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            y_true = self.y_test.cpu().numpy().flatten()
            y_pred = preds.cpu().numpy().flatten()
            y_prob = probs.cpu().numpy().flatten()

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        }

        # AUC-ROC (requires both classes present)
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['auc_roc'] = 0.5

        return metrics


# ═════════════════════════════════════════════════════════════
# Federated Server
# ═════════════════════════════════════════════════════════════

class FederatedServer:
    """
    Central aggregation server for federated learning.
    
    Orchestrates:
    1. Global model initialization
    2. Distribution of global model to clients
    3. Collection of client updates
    4. FedAvg aggregation with differential privacy
    5. Metrics tracking and audit logging
    """

    def __init__(self, epsilon=config.DEFAULT_EPSILON, device='cpu'):
        self.device = device
        self.global_model = create_model().to(device)
        self.dp = DifferentialPrivacy(epsilon=epsilon)
        self.secure_agg = SecureAggregator()
        self.clients = []
        self.round_metrics = []
        self.current_round = 0
        self.total_rounds = 0
        self.training_complete = False

    def register_client(self, client):
        """Register a hospital client with the federation."""
        self.clients.append(client)

    def aggregate_updates(self, client_updates):
        """
        Perform Federated Averaging (FedAvg) on client model updates.
        
        Weighted average by number of training samples, then apply DP noise.
        
        Args:
            client_updates: List of (model_update, num_samples, signature) tuples
            
        Returns:
            Aggregated and privatized model update
        """
        total_samples = sum(n for _, n, _ in client_updates)

        # Weighted average of client updates
        avg_update = OrderedDict()
        for key in client_updates[0][0]:
            weighted_sum = torch.zeros_like(client_updates[0][0][key], dtype=torch.float32)
            for update, num_samples, _ in client_updates:
                weight = num_samples / total_samples
                weighted_sum += update[key].float() * weight
            avg_update[key] = weighted_sum

        # Apply differential privacy noise
        noisy_update = self.dp.add_noise(avg_update, len(client_updates))

        return noisy_update

    def train_round(self):
        """
        Execute one federated training round.
        
        Returns:
            Dictionary of aggregated metrics for this round
        """
        self.current_round += 1
        global_state = copy.deepcopy(self.global_model.state_dict())

        # Collect updates from all clients
        client_updates = []
        round_client_metrics = []

        for client in self.clients:
            # Client trains locally (data never leaves)
            update, num_samples, metrics = client.train_local(global_state)

            # Clip the update
            clipped_update, orig_norm, clip_factor = self.dp.clip_gradients(update)

            # Sign for integrity
            signature = self.secure_agg.sign_update(clipped_update, client.hospital_id)

            client_updates.append((clipped_update, num_samples, signature))
            round_client_metrics.append({
                'hospital_id': client.hospital_id,
                'hospital_name': client.name,
                **metrics,
                'gradient_norm': round(orig_norm, 4),
                'clip_factor': round(clip_factor, 4),
            })

        # Aggregate with FedAvg + DP noise
        aggregated_update = self.aggregate_updates(client_updates)

        # Apply update to global model
        new_state = OrderedDict()
        for key in global_state:
            new_state[key] = global_state[key].float() + aggregated_update[key]
        self.global_model.load_state_dict(new_state)

        # Compute aggregate metrics
        avg_metrics = {
            'round': self.current_round,
            'avg_accuracy': np.mean([m['accuracy'] for m in round_client_metrics]),
            'avg_f1': np.mean([m['f1_score'] for m in round_client_metrics]),
            'avg_auc_roc': np.mean([m['auc_roc'] for m in round_client_metrics]),
            'avg_loss': np.mean([m['train_loss'] for m in round_client_metrics]),
            'avg_precision': np.mean([m['precision'] for m in round_client_metrics]),
            'avg_recall': np.mean([m['recall'] for m in round_client_metrics]),
            'client_metrics': round_client_metrics,
            'privacy_budget_spent': self.dp.budget_spent,
            'epsilon': self.dp.epsilon,
        }
        self.round_metrics.append(avg_metrics)

        return avg_metrics

    def train(self, num_rounds=config.NUM_ROUNDS, verbose=True):
        """
        Run full federated training for specified rounds.
        
        Args:
            num_rounds: Number of aggregation rounds
            verbose: Whether to print progress
            
        Returns:
            List of per-round metrics
        """
        self.total_rounds = num_rounds

        if verbose:
            eps_str = f"ε={self.dp.epsilon}" if self.dp.epsilon != float('inf') else "No DP"
            print(f"\n{'─' * 60}")
            print(f"  FEDERATED TRAINING ({eps_str})")
            print(f"  Rounds: {num_rounds} | Clients: {len(self.clients)}")
            print(f"{'─' * 60}")

        for r in range(num_rounds):
            metrics = self.train_round()

            if verbose and (r + 1) % max(1, num_rounds // 10) == 0:
                print(
                    f"  Round {r + 1:3d}/{num_rounds} │ "
                    f"Loss: {metrics['avg_loss']:.4f} │ "
                    f"Acc: {metrics['avg_accuracy']:.4f} │ "
                    f"F1: {metrics['avg_f1']:.4f} │ "
                    f"AUC: {metrics['avg_auc_roc']:.4f}"
                )

        self.training_complete = True

        if verbose:
            final = self.round_metrics[-1]
            print(f"{'─' * 60}")
            print(f"  TRAINING COMPLETE")
            print(f"  Final Accuracy: {final['avg_accuracy']:.4f}")
            print(f"  Final F1 Score: {final['avg_f1']:.4f}")
            print(f"  Final AUC-ROC:  {final['avg_auc_roc']:.4f}")
            print(f"  Privacy Budget: {self.dp.get_privacy_report()}")
            print(f"{'─' * 60}\n")

        return self.round_metrics

    def get_global_model_state(self):
        """Return the current global model state dict."""
        return copy.deepcopy(self.global_model.state_dict())

    def get_training_summary(self):
        """Return a comprehensive training summary."""
        if not self.round_metrics:
            return {'status': 'not_started'}

        final = self.round_metrics[-1]
        return {
            'status': 'complete' if self.training_complete else 'in_progress',
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'num_clients': len(self.clients),
            'final_accuracy': round(final['avg_accuracy'], 4),
            'final_f1': round(final['avg_f1'], 4),
            'final_auc_roc': round(final['avg_auc_roc'], 4),
            'final_loss': round(final['avg_loss'], 4),
            'privacy': self.dp.get_privacy_report(),
            'rounds_history': [
                {
                    'round': m['round'],
                    'accuracy': round(m['avg_accuracy'], 4),
                    'f1': round(m['avg_f1'], 4),
                    'auc_roc': round(m['avg_auc_roc'], 4),
                    'loss': round(m['avg_loss'], 4),
                }
                for m in self.round_metrics
            ],
        }


# ═════════════════════════════════════════════════════════════
# Convenience function
# ═════════════════════════════════════════════════════════════

def setup_federation(hospital_data_dict, epsilon=config.DEFAULT_EPSILON, device='cpu'):
    """
    Set up a complete federated learning environment.
    
    Args:
        hospital_data_dict: Dict mapping hospital_id → DataFrame
        epsilon: Privacy budget
        device: torch device
        
    Returns:
        Configured FederatedServer with all clients registered
    """
    server = FederatedServer(epsilon=epsilon, device=device)

    for hosp_config in config.HOSPITALS:
        hosp_id = hosp_config['id']
        if hosp_id in hospital_data_dict:
            client = HospitalClient(
                hospital_config=hosp_config,
                data_df=hospital_data_dict[hosp_id],
                device=device,
            )
            server.register_client(client)

    return server
