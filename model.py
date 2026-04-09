"""
Outbreak Prediction Model
=========================
PyTorch neural network for binary classification of disease outbreak risk.
Multi-layer perceptron with BatchNorm, Dropout, and configurable architecture.
"""

import torch
import torch.nn as nn
import config


class OutbreakPredictionModel(nn.Module):
    """
    Multi-layer perceptron for predicting disease outbreak risk.
    
    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Sigmoid
    
    Designed for federated learning: lightweight, fast convergence,
    and compatible with per-sample gradient clipping for differential privacy.
    """

    def __init__(
        self,
        input_dim=config.INPUT_FEATURES,
        hidden_layers=None,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM,
    ):
        super(OutbreakPredictionModel, self).__init__()

        if hidden_layers is None:
            hidden_layers = config.HIDDEN_LAYERS

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        # Output layer (single neuron for binary classification)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Kaiming initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Raw logits of shape (batch_size, 1)
        """
        return self.network(x)

    def predict_proba(self, x):
        """
        Predict probabilities (applies sigmoid to logits).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x, threshold=0.5):
        """
        Predict binary labels.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            threshold: Classification threshold
            
        Returns:
            Binary prediction tensor of shape (batch_size, 1)
        """
        proba = self.predict_proba(x)
        return (proba >= threshold).float()

    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self):
        """Return a string summary of the model architecture."""
        summary_lines = [
            f"OutbreakPredictionModel",
            f"  Parameters: {self.get_num_parameters():,}",
            f"  Architecture: {self.network}",
        ]
        return "\n".join(summary_lines)


def create_model(**kwargs):
    """Factory function to create a new model instance."""
    return OutbreakPredictionModel(**kwargs)


def get_model_size_mb(model):
    """Calculate model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)
