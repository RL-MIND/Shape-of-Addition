"""Model definitions for residual-stream probing."""

import math
import torch
import torch.nn as nn


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
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ProbeMLPRegressor(nn.Module):
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


def create_model(model_type, input_dim, seq_len, feature_dim, num_classes=2, device=None, **kwargs):
    if model_type == "mlp":
        model = ProbeMLP(
            input_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.2),
            num_classes=num_classes,
        )
    elif model_type == "circular":
        model = CircularProbe(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if device is not None:
        model = model.to(device)

    return model
