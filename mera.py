import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class ErrorProbe(nn.Module, ABC):
    """
    Abstract base class for Error Probes.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, h):
        """
        Predicts the error score/probability for hidden state h.
        """
        pass

    @abstractmethod
    def get_steering_vector(self, h, target_alpha, **kwargs):
        """
        Computes the steering vector v to reduce error below target_alpha.
        """
        pass

class LinearProbe(ErrorProbe):
    """
    Linear Probe: p(h) = w^T h
    """
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, h):
        return self.linear(h)

    def get_steering_vector(self, h, target_alpha, **kwargs):
        """
        Closed-form solution for Linear Case.
        """
        w = self.linear.weight.squeeze(0) # (input_dim,)
        
        # current_error = w^T h + b
        current_error = self.forward(h).squeeze(-1) # (batch_size,)
        
        # Calculate delta
        delta = target_alpha - current_error
        
        # lambda = delta / ||w||^2
        w_norm_sq = torch.norm(w, p=2)**2
        lambda_val = delta / (w_norm_sq + 1e-8) # Avoid div by zero
        
        # v = lambda * w
        v = torch.outer(lambda_val, w) # (batch_size, input_dim)
        
        # Only apply if current_error > target_alpha
        mask = (current_error > target_alpha).float().unsqueeze(-1)
        v = v * mask
        
        return v

class MLPProbe(ErrorProbe):
    """
    MLP Probe.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h):
        return self.net(h)

    def get_steering_vector(self, h, target_alpha, method='taylor', **kwargs):
        if method == 'taylor':
            return self._steering_taylor(h, target_alpha)
        elif method == 'numerical':
            return self._steering_numerical(h, target_alpha, **kwargs)
        else:
            raise ValueError(f"Unknown method/Method not implemented: {method}")

    def _steering_taylor(self, h, target_alpha):
        """
        Method 1: Taylor Expansion.
        Linearize locally: g = grad(p(h))
        """
        h_detached = h.detach().clone().requires_grad_(True)
        output = self.forward(h_detached)
        output.sum().backward()
        g = h_detached.grad # (batch_size, input_dim)
        
        current_error = output.detach().squeeze(-1) # (batch_size,)
        delta = target_alpha - current_error
        
        g_norm_sq = torch.sum(g**2, dim=1) # (batch_size,)
        lambda_val = delta / (g_norm_sq + 1e-8)
        
        v = g * lambda_val.unsqueeze(-1)
        
        mask = (current_error > target_alpha).float().unsqueeze(-1)
        v = v * mask
        
        return v

class SteeringController:
    """
    Handles thresholds, decision logic, and penalty calculations.
    """
    def __init__(self, probe, alpha_steer, alpha_backtrack, penalty_type='amplified', penalty_beta=1.0):
        self.probe = probe
        self.alpha_steer = alpha_steer
        self.alpha_backtrack = alpha_backtrack
        self.penalty_type = penalty_type
        self.penalty_beta = penalty_beta

    def decide_steering(self, h):
        """
        Returns:
            v: Steering vector (or None if abstained/backtracked)
            status: 'abstained', 'steered', 'backtracked'
            h_new: The modified hidden state (if applicable)
        """
        # Handle dtype mismatch: Model might be BFloat16, Probe is Float32
        original_dtype = h.dtype
        probe_dtype = next(self.probe.parameters()).dtype
        
        # Perform all calculations in probe_dtype (usually Float32) for precision/compatibility
        h_compute = h.to(probe_dtype)
        
        # Convert alphas (probs) to logits
        # Clamp to avoid inf
        alpha_steer_clamped = max(1e-6, min(1.0 - 1e-6, self.alpha_steer))
        alpha_backtrack_clamped = max(1e-6, min(1.0 - 1e-6, self.alpha_backtrack))
        
        logit_steer = torch.logit(torch.tensor(alpha_steer_clamped, device=h.device, dtype=probe_dtype))
        logit_backtrack = torch.logit(torch.tensor(alpha_backtrack_clamped, device=h.device, dtype=probe_dtype))
        
        with torch.no_grad():
            # Error Probe outputs LOGITS
            error_logits = self.probe(h_compute).squeeze(-1)
        
        batch_size = h.shape[0]
        v = torch.zeros_like(h_compute)
        status = ['abstained'] * batch_size
        h_new = h_compute.clone()
        
        # 2. Steering: logit_steer <= error_logits < logit_backtrack
        mask_steer = (error_logits >= logit_steer) & (error_logits < logit_backtrack)
        
        # 3. Backtrack: error_logits >= logit_backtrack
        mask_backtrack = error_logits >= logit_backtrack
        
        # Apply Steering
        if mask_steer.any():
            h_to_steer = h_compute[mask_steer]
            # Steer towards logit_steer
            # Note: get_steering_vector expects 'target_alpha' in the same space as probe output (Logits)
            v_steer = self.probe.get_steering_vector(h_to_steer, target_alpha=logit_steer)
            v[mask_steer] = v_steer
            h_new[mask_steer] = h_to_steer + v_steer
            
            indices = torch.nonzero(mask_steer).view(-1).tolist()
            for idx in indices:
                status[int(idx)] = 'steered'

        # Apply Backtracking Penalties
        if mask_backtrack.any():
            h_to_backtrack = h_compute[mask_backtrack]
            
            if self.penalty_type == 'amplified':
                v_penalty = self.probe.get_steering_vector(h_to_backtrack, target_alpha=-1.0)
                h_new[mask_backtrack] = h_to_backtrack + v_penalty
            
            indices = torch.nonzero(mask_backtrack).view(-1).tolist()
            for idx in indices:
                status[int(idx)] = 'backtracked'
                
        # Cast back to original dtype for model compatibility
        return v.to(original_dtype), status, h_new.to(original_dtype)

class MERAHook:
    """
    PyTorch hook for Autoregressive Steering.
    """
    def __init__(self, controller, layer_idx):
        self.controller = controller
        self.layer_idx = layer_idx
        # Dictionary to store stats if needed?
        self.steering_stats = []

    def __call__(self, module, inputs, outputs):
        # outputs is usually a tuple (hidden_states,) or just hidden_states
        if isinstance(outputs, tuple):
            h = outputs[0]
            is_tuple = True
        else:
            h = outputs
            is_tuple = False
            
        # h: [Batch, Seq, Dim]
        # We process the LAST token only for autoregressive generation
        # because previous tokens are already fixed.
        seq_len = h.shape[1]
        h_last = h[:, -1, :] # [Batch, Dim]
        
        # Decide steering
        v, status, h_new_last = self.controller.decide_steering(h_last)
        
        # Record stats
        self.steering_stats.append(status)
        
        # Replace the last token in h
        # We need to clone to avoid inplace errors if any, though usually replacing slice is fine.
        # h is likely an activation tensor from the model.
        # h[:, -1, :] = h_new_last # This is inplace.
        
        # To be safe against autograd checks:
        if seq_len > 1:
            h_new = torch.cat([h[:, :-1, :], h_new_last.unsqueeze(1)], dim=1)
        else:
            h_new = h_new_last.unsqueeze(1)
            
        if is_tuple:
            return (h_new,) + outputs[1:]
        else:
            return h_new


class SimpleSteeringController:
    """
    Simple Steering Controller:
    1. Computes 10 average vectors from correct training samples (one per pred label 0-9)
    2. If probe predicts error, replaces hidden state with the nearest average vector
    """
    def __init__(self, probe, avg_vectors, device='cuda'):
        """
        Args:
            probe: Trained LinearProbe or MLPProbe
            avg_vectors: dict mapping digit label (0-9) to average hidden state tensor (Dim,)
        """
        self.probe = probe
        self.device = device
        
        # Convert avg_vectors dict to tensor for efficient distance computation
        # avg_vectors: {0: tensor, 1: tensor, ..., 9: tensor}
        self.labels = sorted(avg_vectors.keys())
        self.avg_tensor = torch.stack([avg_vectors[k].to(device) for k in self.labels], dim=0)  # (10, Dim)
        
    @staticmethod
    def compute_avg_vectors(X_train, y_train, preds_train, device='cuda'):
        """
        Compute average vectors for each pred label from correct samples.
        
        Args:
            X_train: numpy array (N, Dim) - hidden states
            y_train: numpy array (N,) - error labels (0=correct, 1=error)
            preds_train: numpy array (N,) - predicted characters as strings ('0'-'9')
            
        Returns:
            dict mapping digit (0-9) to average tensor
        """
        avg_vectors = {}
        
        # Filter correct samples (y_train == 0 means no error, i.e., correct)
        correct_mask = (y_train == 0)
        
        for digit in range(10):
            digit_str = str(digit)
            # Find correct samples with this pred label
            mask = correct_mask & (preds_train == digit_str)
            
            if mask.sum() > 0:
                avg_vec = X_train[mask].mean(axis=0)
                avg_vectors[digit] = torch.tensor(avg_vec, dtype=torch.float32, device=device)
            else:
                # If no samples for this digit, use zeros (fallback)
                print(f"Warning: No correct samples for digit {digit}. Using zero vector.")
                dim = X_train.shape[1]
                avg_vectors[digit] = torch.zeros(dim, dtype=torch.float32, device=device)
        
        return avg_vectors
    
    def decide_steering(self, h):
        """
        Decide whether to steer based on probe prediction.
        
        Returns:
            v: Steering vector (h_new - h)
            status: List of 'abstained' or 'replaced'
            h_new: The modified hidden state
        """
        original_dtype = h.dtype
        probe_dtype = next(self.probe.parameters()).dtype
        h_compute = h.to(probe_dtype)
        
        batch_size = h.shape[0]
        
        with torch.no_grad():
            # Error Probe outputs LOGITS
            error_logits = self.probe(h_compute).squeeze(-1)  # (batch_size,)
            error_probs = torch.sigmoid(error_logits)
            
            # Predict error if prob > 0.5
            is_error = (error_probs > 0.5)
        
        v = torch.zeros_like(h_compute)
        status = ['abstained'] * batch_size
        h_new = h_compute.clone()
        
        if is_error.any():
            h_error = h_compute[is_error]  # (num_errors, Dim)
            
            # Compute distances to all avg vectors
            # h_error: (num_errors, Dim), avg_tensor: (10, Dim)
            # distances: (num_errors, 10)
            distances = torch.cdist(h_error, self.avg_tensor.to(h_error.dtype))
            
            # Find nearest avg vector for each error sample
            nearest_idx = distances.argmin(dim=1)  # (num_errors,)
            nearest_vectors = self.avg_tensor[nearest_idx]  # (num_errors, Dim)
            
            # Replace hidden states
            h_new[is_error] = nearest_vectors.to(h_compute.dtype)
            v[is_error] = nearest_vectors.to(h_compute.dtype) - h_error
            
            # Update status
            error_indices = torch.nonzero(is_error).view(-1).tolist()
            for idx in error_indices:
                status[idx] = 'replaced'
        
        return v.to(original_dtype), status, h_new.to(original_dtype)


class SimpleHook:
    """
    PyTorch hook for Simple Steering (Replace with nearest avg vector).
    """
    def __init__(self, controller, layer_idx):
        self.controller = controller
        self.layer_idx = layer_idx
        self.steering_stats = []

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
            is_tuple = True
        else:
            h = outputs
            is_tuple = False
            
        seq_len = h.shape[1]
        h_last = h[:, -1, :]  # [Batch, Dim]
        
        # Decide steering
        v, status, h_new_last = self.controller.decide_steering(h_last)
        
        # Record stats
        self.steering_stats.append(status)
        
        # Replace the last token in h
        if seq_len > 1:
            h_new = torch.cat([h[:, :-1, :], h_new_last.unsqueeze(1)], dim=1)
        else:
            h_new = h_new_last.unsqueeze(1)
            
        if is_tuple:
            return (h_new,) + outputs[1:]
        else:
            return h_new