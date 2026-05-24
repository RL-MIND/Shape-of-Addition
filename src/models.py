"""
Model definitions for vertical flow classification.

Contains all neural network models and the factory function to create them.
"""

import math
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class ProbeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, num_classes=2):
        super(ProbeMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ProbeMLPRegressor(nn.Module):
    """MLP regressor with the same hidden layout as ProbeMLP."""

    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, output_dim=1, squeeze_output=True):
        super().__init__()
        self.squeeze_output = squeeze_output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, x):
        out = self.net(x)
        if self.squeeze_output and out.shape[-1] == 1:
            return out.squeeze(-1)
        return out


class CircularProbe(nn.Module):
    """Project hidden states to an angle on a circle and classify digits."""

    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.w1 = nn.Linear(input_dim, 1, bias=False)
        self.w2 = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj1 = self.w1(x).squeeze(-1)
        proj2 = self.w2(x).squeeze(-1)
        theta = torch.atan2(proj1, proj2)
        theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
        angles_per_class = 2 * math.pi / self.num_classes
        class_angles = torch.arange(self.num_classes, device=x.device, dtype=x.dtype) * angles_per_class
        theta_expanded = theta.unsqueeze(1)
        class_angles_expanded = class_angles.unsqueeze(0)
        logits = torch.cos(theta_expanded - class_angles_expanded) * 10.0
        return logits

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        proj1 = self.w1(x).squeeze(-1)
        proj2 = self.w2(x).squeeze(-1)
        theta = torch.atan2(proj1, proj2)
        theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
        preds = theta * (self.num_classes / (2 * math.pi))
        return preds.round().long() % self.num_classes


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, feature_dim, d_model=256, nhead=8, num_layers=2, 
                 dim_feedforward=1024, dropout=0.1, num_classes=2):
        super(TransformerClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        x = self.input_projection(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, feature_dim, num_filters=[64, 128, 256], 
                 kernel_sizes=[3, 3, 3], dropout=0.2, num_classes=2):
        super(CNNClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        conv_layers = []
        in_channels = feature_dim
        
        for out_channels, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        fc_input_dim = num_filters[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.feature_dim, self.seq_len)
        x = self.conv_layers(x)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)
        max_pooled = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        output = self.classifier(x)
        return output


class CNN2DClassifier(nn.Module):
    """2D CNN treating vertical flow (seq_len, feature_dim) as an image."""
    def __init__(self, input_dim, seq_len, feature_dim, channels=[32, 64, 128, 256], 
                 dropout=0.2, num_classes=2):
        """
        Args:
            input_dim: flattened input dim (seq_len * feature_dim)
            seq_len: sequence length (layers), image height
            feature_dim: feature dim, image width
            channels: output channels per layer
            dropout: dropout rate
            num_classes: number of classes
        """
        super(CNN2DClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Asymmetric kernels for seq_len × feature_dim layout (small seq_len, large feature_dim)
        self.conv_layers = nn.Sequential(
            # Layer 1: local inter-layer patterns
            nn.Conv2d(1, channels[0], kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # seq_len × (feature_dim/4)
            
            # Layer 2
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # seq_len × (feature_dim/16)
            
            # Layer 3
            nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4)),  # (seq_len/2) × (feature_dim/64)
            
            # Layer 4
            nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pool to 1×1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[3], channels[3] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[3] // 4, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        # Reshape to image: (batch, 1, seq_len, feature_dim)
        x = x.view(batch_size, 1, self.seq_len, self.feature_dim)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class LogisticRegressionClassifier(nn.Module):
    """Linear logistic regression (softmax); linearly separable baseline."""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LinearSVMClassifier(nn.Module):
    """Linear SVM (scores for MultiMarginLoss)."""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class DecisionTreeWrapper:
    """sklearn decision tree wrapper with PyTorch-compatible API."""
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 num_classes=2, random_state=42):
        """
        Args:
            max_depth: max depth (None = unlimited)
            min_samples_split: min samples to split internal node
            min_samples_leaf: min samples per leaf
            num_classes: number of classes
            random_state: random seed
        """
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.num_classes = num_classes
        self.is_sklearn_model = True  # sklearn-backed model
        self._is_fitted = False
    
    def fit(self, X, y):
        """Train the model."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        self.clf.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """Predict class labels."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.clf.predict_proba(X)
    
    def eval(self):
        """PyTorch API compatibility."""
        pass
    
    def train(self, mode=True):
        """PyTorch API compatibility."""
        pass
    
    def to(self, device):
        """PyTorch API compatibility; sklearn models need no GPU move."""
        return self
    
    def parameters(self):
        """PyTorch API compatibility; empty iterator."""
        return iter([])
    
    def state_dict(self):
        """PyTorch API compatibility."""
        return {'clf': self.clf, 'num_classes': self.num_classes}
    
    def load_state_dict(self, state_dict):
        """PyTorch API compatibility."""
        self.clf = state_dict['clf']
        self.num_classes = state_dict['num_classes']

    def __call__(self, X):
        """PyTorch call API; returns log-probability logits."""
        # Probabilities -> log-probs
        probs = self.predict_proba(X)
        if isinstance(probs, list):
            probs = np.array(probs)
        
        epsilon = 1e-10  # avoid log(0)
        log_probs = np.log(probs + epsilon)
        
        return torch.tensor(log_probs, dtype=torch.float32)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for sequence modeling."""
    def __init__(self, input_dim, seq_len, feature_dim, hidden_dim=256, num_layers=2,
                 dropout=0.3, num_classes=2):
        """
        Args:
            input_dim: flattened input dim (seq_len * feature_dim)
            seq_len: sequence length (layers)
            feature_dim: feature dimension
            hidden_dim: LSTM hidden size
            num_layers: number of LSTM layers
            dropout: dropout rate
            num_classes: number of classes
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier on BiLSTM output (hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        # (batch, seq_len, feature_dim)
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Last timestep (bidirectional)
        out = lstm_out[:, -1, :]
        
        output = self.classifier(out)
        return output


def create_model(model_type, input_dim, seq_len, feature_dim, num_classes=2, device=None, **kwargs):
    """
    Create a model instance for the given model_type.
    
    Args:
        model_type: 'mlp', 'transformer', 'cnn', 'cnn2d', 'bilstm', 'logreg', 'svm', 'dtree'
        input_dim: input dimension
        seq_len: sequence length
        feature_dim: feature dimension
        num_classes: number of classes
        device: device (None = do not move model)
        **kwargs: model-specific hyperparameters
        
    Returns:
        Model instance
    """
    if model_type == 'mlp':
        model = ProbeMLP(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 512),
            dropout=kwargs.get('dropout', 0.2),
            num_classes=num_classes
        )
    elif model_type == 'transformer':
        model = TransformerClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            d_model=kwargs.get('d_model', 256),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 2),
            dim_feedforward=kwargs.get('dim_feedforward', 1024),
            dropout=kwargs.get('dropout', 0.1),
            num_classes=num_classes
        )
    elif model_type == 'cnn':
        model = CNNClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            num_filters=kwargs.get('num_filters', [64, 128, 256]),
            kernel_sizes=kwargs.get('kernel_sizes', [3, 3, 3]),
            dropout=kwargs.get('dropout', 0.2),
            num_classes=num_classes
        )
    elif model_type == 'cnn2d':
        model = CNN2DClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            channels=kwargs.get('channels', [32, 64, 128, 256]),
            dropout=kwargs.get('dropout', 0.2),
            num_classes=num_classes
        )
    elif model_type == 'logreg':
        model = LogisticRegressionClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
    elif model_type == 'svm':
        model = LinearSVMClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
    elif model_type == 'bilstm':
        model = BiLSTMClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3),
            num_classes=num_classes
        )
    elif model_type == 'dtree':
        # sklearn decision tree (no GPU)
        model = DecisionTreeWrapper(
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            num_classes=num_classes,
            random_state=kwargs.get('random_state', 42)
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if device is not None:
        model = model.to(device)
    
    return model
