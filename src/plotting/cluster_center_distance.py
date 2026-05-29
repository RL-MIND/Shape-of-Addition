"""
Cluster center distance analysis script.

Loads samples from .h5 data files and labels them by (GT Digit, Pred Digit, GT In_carry).
The user specifies three labels; computes the mean center for each label's samples and pairwise center distances.

Supported distance metrics:
- euclidean: Euclidean distance
- cosine: Cosine distance (1 - cosine_similarity)

Usage:
    python cluster_center_distance.py
"""

from pathlib import Path as _Path
import sys as _sys

_SRC_DIR = _Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Error: please install h5py: pip install h5py")
    exit(1)

from utils.flow_utils import load_samples_meta


# CUDA_VISIBLE_DEVICES=2 python cluster_center_distance.py


# ==========================================
# ========== Configuration ==========
# ==========================================

# --- Data file path ---
DATA_FILE_PATH = '/home/wenliuyuan/activations/results/activations/plus_num3len10_Qwen3-4B_nocheckall_balance.h5'
# DATA_FILE_PATH = 'results/activations/plus_num4len10_Qwen3-4B'

# --- Position selection (which position's data to use) ---
POSITION_SELECT = 4

# --- Distance metric: 'euclidean' or 'cosine' ---
DISTANCE_METRIC = 'cosine'  # 'euclidean', 'cosine'

# --- Three label definitions (GT Digit, Pred Digit, GT In_carry) ---
# Each label is a triple (gt_digit, pred_digit, in_carry)
# gt_digit: ground-truth digit (0-9)
# pred_digit: predicted digit (0-9)
# in_carry: input carry (0 or 1 for addition)

RAW_SUM = 8
# Note: The "reference center" for the Distance vs C_potential plot comes from PLOT_SAMPLE_LABELS[0].
# Label order matches the activation-analysis script for consistent plots.
_A = RAW_SUM
_B = (RAW_SUM + 1)%10
_C = (RAW_SUM + 2)%10

LABEL_1 = (_A, _A, 0)  # Label 1: gt=0, pred=0, in_carry=0
LABEL_2 = (_B, _B, 1)  # Label 2: gt=1, pred=1, in_carry=1
LABEL_3 = (_C, _C, 2)  # Label 3: gt=2, pred=2, in_carry=2


# --- Wrong-prediction sample labels ---
# Each tuple is a wrong sample (gt_digit, pred_digit, in_carry)
# Requirements: 1) gt_digit != pred_digit (must be a wrong sample)
#               2) gt_digit and pred_digit must appear among LABEL_1/2/3 gt_digits
# Each wrong sample gets distances to the corresponding correct centers
# e.g. (4,3,0) computes distance to (4,4,?) center and (3,3,?) center

# WRONG_SAMPLE_LABELS = [
#     (4, 3, 0), (4, 3, 1),  # gt=4 predicted as 3
#     (3, 4, 0), (3, 4, 1),  # gt=3 predicted as 4
#     (5, 4, 1), (5, 4, 2),  # gt=5 predicted as 4
#     (4, 5, 1), (4, 5, 2),  # gt=4 predicted as 5
#     (3,4,3), (5,4,0), (4,5,3)
# ]

WRONG_SAMPLE_LABELS = [
    (_A, _B, 0),(_A, _B, 1),
    (_B, _A, 1),(_B, _A, 0),
    (_B, _C, 1),(_B, _C, 2),
    (_C, _B, 1),(_C, _B, 2),
    (_A, _B, 3),
    (_C, _A, 0),
    (_B, _C, 3),
]



PLOT_SAMPLE_LABELS = [
    (_B, _B, 1),
    (_C, _B, 2),
    (_B, _C, 1),
    (_C, _C, 2),
    (_A, _B, 0),
    (_B, _A, 1),
    (_A, _A, 0),
]


# --- Layer selection ---
# None: use all layers (flattened to 1D vector)
# int: use a specific layer (0, 1, 2, ...)
SPECIFIC_LAYER_INDEX = 36  # None, 0, 1, 2, ...

# --- Model path and norm settings ---
MODEL_PATH = "Qwen/Qwen3-4B"
APPLY_MODEL_NORM = True  # True/False: load model and apply final norm to hidden states

# --- Sample filter mode ---
# 'correct': only correctly predicted samples (gt == pred)
# 'incorrect': only incorrectly predicted samples (gt != pred)
# 'all': all samples
SAMPLE_FILTER_MODE = 'all'  # 'correct', 'incorrect', 'all'

# --- Intermediate vector token visualization ---
# If enabled, decode tokens at evenly spaced points on the line between any two label centers
COMPUTE_INTERMEDIATE_TOKENS = True  # True/False: compute intermediate tokens along label center lines
INTERMEDIATE_STEPS = 50  # Number of evenly spaced points on the line (endpoints excluded)

# --- Save scatter plot data ---
# If enabled, save all plot data as CSV in the same folder as plots
SAVE_PLOT_DATA = True  # True, False

# ==========================================
# ========== Utility functions ==========
# ==========================================

def decode_array(arr: List[Any]) -> List[str]:
    """Decode bytes array to string list."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def compute_euclidean_distance(center1: np.ndarray, center2: np.ndarray) -> float:
    """Compute Euclidean distance between two centers."""
    return float(np.linalg.norm(center1 - center2))


def compute_cosine_distance(center1: np.ndarray, center2: np.ndarray) -> Tuple[float, float]:
    """Compute cosine distance (1 - cosine_similarity) and angle theta between two centers."""
    norm1 = np.linalg.norm(center1)
    norm2 = np.linalg.norm(center2)
    if norm1 == 0 or norm2 == 0:
        return 1.0, 90.0  # If either vector is zero, distance=1, angle=90°
    cosine_sim = np.dot(center1, center2) / (norm1 * norm2)
    # Clip cosine_sim to [-1, 1] to avoid arccos errors from floating-point noise
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    theta_rad = np.arccos(cosine_sim)  # radians
    theta_deg = np.degrees(theta_rad)  # degrees
    return float(1 - cosine_sim), float(theta_deg)


def compute_distance(center1: np.ndarray, center2: np.ndarray, metric: str) -> Union[float, Tuple[float, float]]:
    """Compute distance between two vectors using the specified metric."""
    if metric == 'euclidean':
        return compute_euclidean_distance(center1, center2)
    elif metric == 'cosine':
        return compute_cosine_distance(center1, center2)  # returns (distance, theta_deg)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}. Use 'euclidean' or 'cosine'.")


def compute_raw_sum(question: str, sum_pos: int) -> int:
    """
    Compute the raw digit sum at the given position (sum of operand digits at that position, no mod).
    
    Args:
        question: Expression string, e.g. "123 + 456 + 789"
        sum_pos: Sum position index (from most significant digit, 0 = MSB)
    
    Returns:
        int: Raw sum (>=0), or -1 if it cannot be computed
    """
    try:
        # Parse operands
        operands = []
        for part in question.split('+'):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))
        
        if not operands:
            return -1
        
        # Compute sum
        result = sum(operands)
        result_str = str(result)
        result_len = len(result_str)
        
        # Max operand digit count
        max_operand_len = max(len(str(op)) for op in operands)
        
        # Digit offset (sum may have one more digit than operands)
        extra_digits = result_len - max_operand_len
        
        # Corresponding position in operands
        operand_pos = sum_pos - extra_digits
        
        # operand_pos < 0 means no corresponding digit in operands at this sum position
        if operand_pos < 0:
            return 0
        
        # Sum digit values at this position across operands
        digit_sum = 0
        for op in operands:
            op_str = str(op)
            op_len = len(op_str)
            right_idx = result_len - 1 - sum_pos  # index from the right
            op_digit_idx = op_len - 1 - right_idx  # index from the left in operand
            
            if 0 <= op_digit_idx < op_len:
                digit_sum += int(op_str[op_digit_idx])
        
        return digit_sum
    except Exception:
        return -1


def compute_c_potential(question: str, current_pos: int) -> float:
    """
    Compute C_potential (Potential of Truth).
    Formula: C_potential(pos) = sum( raw_sum(k) / 10^(k-pos) ) for k = pos+1 to end
    
    Args:
        question: Expression string
        current_pos: Current position index (from MSB, 0 = MSB)
    
    Returns:
        float: C_potential value
    """
    try:
        operands = []
        for part in question.split('+'):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))
        if not operands:
            return 0.0
        
        # Estimated total digit count of the result
        result_len = len(str(sum(operands)))
        
        # Sum k = current_pos+1 .. result_len-1; middle columns may have raw_sum 0, do not break early on small terms
        c_potential = 0.0
        for k in range(current_pos + 1, result_len):
            raw_sum_k = compute_raw_sum(question, k)
            if raw_sum_k == -1:
                break
            exponent = k - current_pos
            c_potential += raw_sum_k / (10 ** exponent)

        return c_potential
    except Exception:
        return 0.0


# ==========================================
# ========== Data loading ==========
# ==========================================

def load_model_and_tokenizer():
    """
    Load model and tokenizer.
    
    Returns:
        (model_obj, tokenizer, norm_layer): model, tokenizer, and norm layer; (None, None, None) on failure
    """
    print(f"\n[Loading model and tokenizer]")
    print(f"  Model path: {MODEL_PATH}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        model_obj = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        model_obj.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"  Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"  [Error] Model loading failed: {e}")
        return None, None, None
    
    # Find norm layer
    norm_layer = None
    if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'norm'):
        norm_layer = model_obj.model.norm
    elif hasattr(model_obj, 'transformer') and hasattr(model_obj.transformer, 'ln_f'):
        norm_layer = model_obj.transformer.ln_f
    
    if norm_layer is not None:
        print(f"  Found norm layer: {type(norm_layer).__name__}")
    
    return model_obj, tokenizer, norm_layer


def apply_model_norm(flows_list: List[np.ndarray], layer_index: Union[int, None] = None,
                     model_obj=None, norm_layer=None) -> Tuple[List[np.ndarray], Any, Any]:
    """
    Load model and apply final norm to hidden states.
    
    Args:
        flows_list: List of flow features, each shape (seq_len, feature_dim)
        layer_index: Layer index; if set, norm is applied only to that layer
        model_obj: Optional pre-loaded model; reused if provided
        norm_layer: Optional pre-loaded norm layer
    
    Returns:
        (normed_flows_list, model_obj, norm_layer): normed flows, model, and norm layer
    """
    if not APPLY_MODEL_NORM:
        return flows_list, None, None
    
    # Reload model if not passed in
    if model_obj is None:
        print(f"\n[Applying model norm]")
        print(f"  Loading model: {MODEL_PATH}")
        
        from transformers import AutoModelForCausalLM
        
        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
            )
            model_obj.eval()
        except Exception as e:
            print(f"  [Error] Model loading failed: {e}")
            print(f"  Continuing with raw hidden states.")
            return flows_list, None, None
    
    # Find norm layer (if not passed in)
    if norm_layer is None:
        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'norm'):
            norm_layer = model_obj.model.norm
        elif hasattr(model_obj, 'transformer') and hasattr(model_obj.transformer, 'ln_f'):
            norm_layer = model_obj.transformer.ln_f
    
    if norm_layer is None:
        print("  [Error] Could not find model final norm layer. Using raw data.")
        return flows_list, model_obj, None
    
    print(f"  Found norm layer: {type(norm_layer).__name__}")
    
    # Convert to tensor and apply norm
    flows_array = np.stack(flows_list, axis=0)  # (N, seq_len, feature_dim)
    N, seq_len, feature_dim = flows_array.shape
    
    if layer_index is not None:
        # Apply norm only to the specified layer
        if layer_index < 0 or layer_index >= seq_len:
            raise ValueError(f"Layer index {layer_index} out of range [0, {seq_len-1}]")
        
        layer_data = flows_array[:, layer_index, :]  # (N, feature_dim)
        X_tensor = torch.from_numpy(layer_data).to(model_obj.device).to(model_obj.dtype)
        
        print(f"  Applying norm to layer {layer_index}, {N} samples...")
        
        normed_batches = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_in = X_tensor[i:i+batch_size]
                batch_out = norm_layer(batch_in)
                normed_batches.append(batch_out.cpu().float().numpy())
        
        normed_layer = np.concatenate(normed_batches, axis=0)  # (N, feature_dim)
        
        # Update the specified layer in flows_array
        flows_array[:, layer_index, :] = normed_layer
    else:
        # Apply norm to all layers
        X_tensor = torch.from_numpy(flows_array).to(model_obj.device).to(model_obj.dtype)
        
        print(f"  Applying norm to all {seq_len} layers, {N} samples...")
        
        normed_batches = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_in = X_tensor[i:i+batch_size]  # (batch, seq_len, feature_dim)
                batch_out = norm_layer(batch_in)
                normed_batches.append(batch_out.cpu().float().numpy())
        
        flows_array = np.concatenate(normed_batches, axis=0)  # (N, seq_len, feature_dim)
    
    print(f"  Norm application complete.")
    
    # Convert back to list
    normed_flows_list = [flows_array[i] for i in range(N)]
    return normed_flows_list, model_obj, norm_layer


def load_samples_with_labels(file_path: Union[str, Path], position_select: int) -> Tuple[
    List[np.ndarray],  # flows
    List[Tuple[int, int, int]],  # (gt_digit, pred_digit, in_carry)
    List[int],  # sample_indices
]:
    """
    Load data and label each sample (GT Digit, Pred Digit, GT In_carry).
    
    Args:
        file_path: Data file path
        position_select: Position index (int)
    
    Returns:
        flows_list: Flow features per sample
        labels_list: Label triples (gt_digit, pred_digit, in_carry) per sample
        sample_idx_list: sample_idx per sample
    """
    file_path = Path(file_path)
    
    # Auto-append .h5 extension if missing
    if file_path.suffix == "":
        h5_path = file_path.with_suffix(".h5")
        if h5_path.exists():
            file_path = h5_path
        else:
            raise FileNotFoundError(f"Data file not found: {file_path} or {h5_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")
    
    print(f"Loading data file: {file_path}")
    print(f"Selected position: {position_select}")
    
    flows_list: List[np.ndarray] = []
    labels_list: List[Tuple[int, int, int]] = []
    sample_idx_list: List[int] = []
    
    with h5py.File(file_path, "r") as hf:
        # Read data for the selected position from all_token_results
        pos_name = f"pos_{position_select}"
        positions_group = hf.get("all_token_results")
        
        if positions_group is None:
            raise ValueError("Data file has no all_token_results group")
        
        if pos_name not in positions_group:
            raise ValueError(f"Data file has no position {position_select}")
        
        pos_group = positions_group[pos_name]
        
        # Read flows
        if "flows" not in pos_group:
            raise ValueError(f"Position {position_select} has no flows data")
        flows_array = pos_group["flows"][:]
        
        # Read preds (predicted digits)
        if "preds" not in pos_group:
            raise ValueError(f"Position {position_select} has no preds data")
        preds = decode_array(pos_group["preds"][:])
        
        # Read gt_chars (ground-truth digits)
        if "gt_chars" not in pos_group:
            raise ValueError(f"Position {position_select} has no gt_chars data")
        gt_chars = decode_array(pos_group["gt_chars"][:])
        
        # Read incoming_carries (input carry)
        if "incoming_carries" not in pos_group:
            raise ValueError(f"Position {position_select} has no incoming_carries data")
        incoming_carries = list(pos_group["incoming_carries"][:])
        
        # Read sample_indices
        if "sample_indices" in pos_group:
            sample_indices = list(pos_group["sample_indices"][:])
        else:
            sample_indices = list(range(len(flows_array)))
        
        print(f"  Total samples: {len(flows_array)}")
        
        # Build flows and labels
        for i in range(len(flows_array)):
            flow = flows_array[i]
            pred_char = preds[i]
            gt_char = gt_chars[i]
            in_carry = incoming_carries[i]
            sample_idx = sample_indices[i]
            
            # Parse gt_digit and pred_digit
            try:
                gt_digit = int(gt_char)
                pred_digit = int(pred_char)
            except (ValueError, TypeError):
                # Skip unparseable samples (e.g. extra)
                continue
            
            # Validate in_carry (for addition, carry may be 0, 1, 2, ...)
            if in_carry < 0:
                continue
            
            # Filter by SAMPLE_FILTER_MODE
            is_correct = (gt_digit == pred_digit)
            if SAMPLE_FILTER_MODE == 'correct' and not is_correct:
                continue
            elif SAMPLE_FILTER_MODE == 'incorrect' and is_correct:
                continue
            # 'all' mode: no filtering
            
            flows_list.append(np.asarray(flow))
            labels_list.append((gt_digit, pred_digit, int(in_carry)))
            sample_idx_list.append(int(sample_idx))
    
    print(f"  Valid samples: {len(flows_list)}")
    return flows_list, labels_list, sample_idx_list


def filter_by_label(
    flows_list: List[np.ndarray],
    labels_list: List[Tuple[int, int, int]],
    target_label: Tuple[int, int, int],
    sample_idx_list: List[int] = None
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
    """
    Filter samples matching the target label.
    
    Args:
        flows_list: List of flow features
        labels_list: List of label triples
        target_label: Target label (gt_digit, pred_digit, in_carry)
        sample_idx_list: Optional sample index list
    
    Returns:
        filtered_flows: Matching flow features
        filtered_sample_indices: Matching sample indices (if sample_idx_list was provided)
    """
    filtered_flows = []
    filtered_sample_indices = []
    
    if sample_idx_list is None:
        for flow, label in zip(flows_list, labels_list):
            if label == target_label:
                filtered_flows.append(flow)
        return filtered_flows
    
    for flow, label, s_idx in zip(flows_list, labels_list, sample_idx_list):
        if label == target_label:
            filtered_flows.append(flow)
            filtered_sample_indices.append(s_idx)
    return filtered_flows, filtered_sample_indices


def compute_center(flows: List[np.ndarray], layer_index: Union[int, None] = None) -> np.ndarray:
    """
    Compute the mean center of a group of samples.
    
    Args:
        flows: List of sample flows, each shape (seq_len, feature_dim)
        layer_index: Layer index; None flattens all layers, int selects one layer
    
    Returns:
        center: Mean center vector
    """
    if not flows:
        raise ValueError("No samples available to compute center")
    
    flows_array = np.stack(flows, axis=0)  # (N, seq_len, feature_dim)
    
    if layer_index is not None:
        # Select specific layer
        if layer_index < 0 or layer_index >= flows_array.shape[1]:
            raise ValueError(f"Layer index {layer_index} out of range [0, {flows_array.shape[1]-1}]")
        flows_array = flows_array[:, layer_index, :]  # (N, feature_dim)
    
    # Flatten flows to 1D and compute mean
    flows_flat = flows_array.reshape(len(flows), -1)  # (N, D)
    center = np.mean(flows_flat, axis=0)  # (D,)
    
    return center


# ==========================================
# ========== Intermediate vector token visualization ==========
# ==========================================

def decode_vector_to_token(
    vector: np.ndarray, 
    model_obj, 
    tokenizer
) -> str:
    """
    Decode a single vector to a token via lm_head.
    
    Args:
        vector: Hidden state vector, shape (feature_dim,)
        model_obj: Language model (for lm_head)
        tokenizer: Tokenizer
    
    Returns:
        Decoded token string
    """
    with torch.no_grad():
        vec_tensor = torch.tensor(vector, dtype=model_obj.dtype).unsqueeze(0).to(model_obj.device)
        logits = model_obj.lm_head(vec_tensor)
        token_id = logits.argmax(dim=-1).item()
        token = tokenizer.decode([token_id])
    return token.strip()


def compute_intermediate_tokens(
    center_1: np.ndarray,
    center_2: np.ndarray,
    model_obj,
    tokenizer,
    num_steps: int = 10
) -> List[Tuple[float, str, np.ndarray]]:
    """
    Compute tokens at evenly spaced points on the line between two center vectors.
    
    Args:
        center_1: First center vector
        center_2: Second center vector
        model_obj: Language model
        tokenizer: Tokenizer
        num_steps: Number of evenly spaced points (endpoints excluded)
    
    Returns:
        list of (t, token, vector): t is interpolation factor (0 to 1), token is decoded string, vector is intermediate vector
    """
    results = []
    
    # Evenly spaced points on the line (endpoints excluded)
    for i in range(1, num_steps + 1):
        t = i / (num_steps + 1)  # t ∈ (0, 1)
        intermediate = center_1 * (1 - t) + center_2 * t
        token = decode_vector_to_token(intermediate, model_obj, tokenizer)
        results.append((t, token, intermediate))
    
    return results


def plot_intermediate_token_paths(
    centers: List[np.ndarray],
    labels: List[str],
    target_labels: List[Tuple[int, int, int]],
    intermediate_results: Dict[Tuple[int, int], List[Tuple[float, str, np.ndarray]]],
    filtered_flows_list: List[List[np.ndarray]],
    layer_index: Union[int, None],
    wrong_sample_data: Dict[Tuple[int, int, int], Tuple[List[np.ndarray], int, int]] = None,
    save_dir: str = "plots_cluster",
    position: int = 0
):
    """
    Plot 2D token paths between the three labels pairwise.
    
    Args:
        centers: List of three center vectors
        labels: Label names ["Label 1", "Label 2", "Label 3"]
        target_labels: List of label triples
        intermediate_results: Intermediate tokens between label pairs {(i,j): [(t, token, vec), ...]}
        filtered_flows_list: Sample flows for the three labels
        layer_index: Layer index for extracting a specific layer from samples
        wrong_sample_data: Wrong samples {wrong_label: (flows_list, gt_center_idx, pred_center_idx)}
        save_dir: Output directory
        position: Position index
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Collect all vectors for PCA
    all_vectors = []
    vector_info = []  # (type, label_or_pair, index, extra_info)
    
    # Add all sample points matching the three labels
    for label_idx, (flows, target_label) in enumerate(zip(filtered_flows_list, target_labels)):
        if flows is not None and centers[label_idx] is not None:
            center = centers[label_idx]
            for flow in flows:
                # Extract vector at the specified layer
                if layer_index is not None:
                    vec = flow[layer_index, :]
                else:
                    vec = flow.flatten()
                all_vectors.append(vec)
                # Distance from this point to its center
                dist_result = compute_distance(vec, center, DISTANCE_METRIC)
                if DISTANCE_METRIC == 'cosine':
                    dist_val = dist_result[0]  # cosine returns (distance, theta)
                else:
                    dist_val = dist_result
                # Marker: label + distance without spaces, e.g. "(3,3,0,0.12)"
                marker_text = f"({target_label[0]},{target_label[1]},{target_label[2]},{dist_val:.2f})"
                vector_info.append(('sample', label_idx, None, marker_text))
    
    # Add wrong-prediction sample points
    if wrong_sample_data:
        for wrong_label, (wrong_flows, gt_center_idx, pred_center_idx) in wrong_sample_data.items():
            gt_center = centers[gt_center_idx]
            pred_center = centers[pred_center_idx]
            if gt_center is None or pred_center is None:
                continue
            for flow in wrong_flows:
                # Extract vector at the specified layer
                if layer_index is not None:
                    vec = flow[layer_index, :]
                else:
                    vec = flow.flatten()
                all_vectors.append(vec)
                # Distance to GT center
                dist_gt_result = compute_distance(vec, gt_center, DISTANCE_METRIC)
                if DISTANCE_METRIC == 'cosine':
                    dist_gt = dist_gt_result[0]
                else:
                    dist_gt = dist_gt_result
                # Distance to Pred center
                dist_pred_result = compute_distance(vec, pred_center, DISTANCE_METRIC)
                if DISTANCE_METRIC == 'cosine':
                    dist_pred = dist_pred_result[0]
                else:
                    dist_pred = dist_pred_result
                # Marker format: "(4,3,0,dis_to_4,dis_to_3)"
                gt_digit, pred_digit, carry = wrong_label
                marker_text = f"({gt_digit},{pred_digit},{carry},{dist_gt:.2f},{dist_pred:.2f})"
                vector_info.append(('wrong_sample', wrong_label, None, marker_text))
    
    # Add the three center points
    for i, center in enumerate(centers):
        if center is not None:
            all_vectors.append(center)
            vector_info.append(('center', i, None, None))
    
    # Add intermediate points
    for (i, j), results in intermediate_results.items():
        for idx, (t, token, vec) in enumerate(results):
            all_vectors.append(vec)
            vector_info.append(('intermediate', (i, j), idx, None))
    
    if len(all_vectors) < 2:
        print("Not enough vectors for PCA dimensionality reduction")
        return
    
    # PCA to 2D
    all_vectors_array = np.stack(all_vectors, axis=0)
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_vectors_array)
    
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Extract coordinates
    sample_coords = {0: [], 1: [], 2: []}  # label_idx -> [(coord, marker_text), ...]
    wrong_sample_coords = []  # [(coord, marker_text), ...]
    center_coords = {}
    intermediate_coords = {}
    
    for idx, (vtype, info, sub_idx, extra) in enumerate(vector_info):
        if vtype == 'sample':
            label_idx = info
            sample_coords[label_idx].append((coords_2d[idx], extra))  # (coord, marker_text)
        elif vtype == 'wrong_sample':
            wrong_sample_coords.append((coords_2d[idx], extra))  # (coord, marker_text)
        elif vtype == 'center':
            center_coords[info] = coords_2d[idx]
        else:
            pair = info
            if pair not in intermediate_coords:
                intermediate_coords[pair] = []
            intermediate_coords[pair].append((coords_2d[idx], intermediate_results[pair][sub_idx][1]))  # (coord, token)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color scheme
    center_colors = ['#e74c3c', '#3498db', '#2ecc71']  # red, blue, green
    sample_colors = ['#e74c3c', '#3498db', '#2ecc71']  # match centers
    wrong_sample_color = '#7f8c8d'  # gray for wrong samples
    line_colors = {
        (0, 1): '#9b59b6',  # purple
        (0, 2): '#f39c12',  # orange
        (1, 2): '#1abc9c',  # teal
    }
    
    # Plot all correct sample points (label text as marker)
    for label_idx, coords_markers in sample_coords.items():
        color = sample_colors[label_idx]
        for coord, marker_text in coords_markers:
            ax.text(
                coord[0], coord[1], marker_text,
                fontsize=7,
                color=color,
                alpha=0.6,
                ha='center',
                va='center',
                zorder=1
            )
    
    # Plot all wrong sample points (gray)
    for coord, marker_text in wrong_sample_coords:
        ax.text(
            coord[0], coord[1], marker_text,
            fontsize=7,
            color=wrong_sample_color,
            alpha=0.6,
            ha='center',
            va='center',
            zorder=1
        )
    
    # Draw connecting lines (dashed) — zorder above sample points
    for (i, j), color in line_colors.items():
        if i in center_coords and j in center_coords:
            x = [center_coords[i][0], center_coords[j][0]]
            y = [center_coords[i][1], center_coords[j][1]]
            ax.plot(x, y, '--', color=color, alpha=0.3, linewidth=1.5, zorder=2)
    
    # Plot intermediate points and token labels
    for (i, j), coords_tokens in intermediate_coords.items():
        color = line_colors.get((i, j), 'gray')
        for coord, token in coords_tokens:
            ax.plot(coord[0], coord[1], 'o', color=color, markersize=6, alpha=0.7, zorder=3)
            ax.annotate(
                token,
                (coord[0], coord[1]),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=9,
                color=color,
                fontweight='bold',
                zorder=4
            )
    
    # Plot the three center points
    for i, (name, target_label) in enumerate(zip(labels, target_labels)):
        if i in center_coords:
            coord = center_coords[i]
            ax.plot(coord[0], coord[1], 'o', color=center_colors[i], markersize=15, 
                   markeredgecolor='black', markeredgewidth=2, zorder=10)
            # Decode center point token
            center_token = decode_vector_to_token(centers[i], 
                                                   intermediate_results[(0,1)][0][2].__class__.__bases__[0], 
                                                   None) if False else ""
            label_text = f"{target_label}"
            ax.annotate(
                label_text,
                (coord[0], coord[1]),
                textcoords="offset points",
                xytext=(15, -10),
                ha='left',
                fontsize=10,
                color=center_colors[i],
                fontweight='bold',
                # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=center_colors[i]),
                zorder=11
            )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, (name, target_label) in enumerate(zip(labels, target_labels)):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{target_label}',
                                       markerfacecolor=center_colors[i], markersize=10, markeredgecolor='black'))
    for (i, j), color in line_colors.items():
        legend_elements.append(Line2D([0], [0], linestyle='--', color=color, 
                                       label=f'{target_labels[i]} ↔ {target_labels[j]} Path'))
    
    ax.legend(handles=legend_elements, fontsize=9)
    
    # Title and axis labels
    ax.set_title(f'Intermediate Token Paths between Labels (PCA 2D)\nPosition: {position}, Steps: {INTERMEDIATE_STEPS}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('PCA Component 1', fontsize=10)
    ax.set_ylabel('PCA Component 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'intermediate_tokens_pos{position}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {save_path}")
    plt.close()


# ==========================================
# ========== Distance vs C_potential scatter plot ==========
# ==========================================

def plot_distance_vs_c_potential(
    flows_list: List[np.ndarray],
    labels_list: List[Tuple[int, int, int]],
    sample_idx_list: List[int],
    samples_meta: Dict[int, Dict[str, str]],
    reference_center: np.ndarray,
    reference_label: Tuple[int, int, int],
    plot_sample_labels: List[Tuple[int, int, int]],
    layer_index: Union[int, None],
    distance_metric: str,
    position: int,
    save_dir: str = "plots_cluster"
):
    """
    Plot distance vs c_potential scatter.
    
    Args:
        flows_list: All sample flow features
        labels_list: All sample label triples
        sample_idx_list: All sample_idx values
        samples_meta: Sample metadata with question/pred/gt
        reference_center: Reference cluster center (center of PLOT_SAMPLE_LABELS[0])
        reference_label: Reference label (PLOT_SAMPLE_LABELS[0])
        plot_sample_labels: Labels to include in the plot
        layer_index: Layer index
        distance_metric: Distance metric
        position: Position index
        save_dir: Output directory
    """
    import matplotlib.pyplot as plt
    
    print(f"\n[Plotting Distance vs C_potential scatter]")
    print(f"  Reference center: {reference_label}")
    print(f"  Plot labels: {plot_sample_labels}")
    
    # Collect distance and c_potential for samples in PLOT_SAMPLE_LABELS
    distances = []
    c_potentials = []
    sample_labels_for_plot = []  # for color grouping
    
    for flow, label, s_idx in zip(flows_list, labels_list, sample_idx_list):
        if label not in plot_sample_labels:
            continue
        
        # Extract vector at the specified layer
        if layer_index is not None:
            vec = flow[layer_index, :]
        else:
            vec = flow.flatten()
        
        # Compute distance
        dist_result = compute_distance(vec, reference_center, distance_metric)
        if distance_metric == 'cosine':
            dist_val = dist_result[0]  # cosine returns (distance, theta)
        else:
            dist_val = dist_result
        
        # Get question and compute c_potential
        if s_idx not in samples_meta:
            continue
        question = samples_meta[s_idx].get("question", "")
        if not question:
            continue
        
        c_pot = compute_c_potential(question, position)
        
        distances.append(dist_val)
        c_potentials.append(c_pot)
        sample_labels_for_plot.append(label)
    
    if not distances:
        print("  Warning: no matching samples found, skipping plot")
        return
    
    print(f"  Matching samples: {len(distances)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color scheme by label
    unique_labels = sorted(list(set(sample_labels_for_plot)), key=lambda x: (x[0], x[1]))
    
    # Split correct vs incorrect labels
    correct_labels = [l for l in unique_labels if l[0] == l[1]]
    incorrect_labels = [l for l in unique_labels if l[0] != l[1]]
    
    print(f"  Correct labels: {correct_labels}")
    print(f"  Incorrect labels: {incorrect_labels}")
    
    # Primary colors (R, G, B) for correct labels; fallback if more than 3
    base_colors = ['#ff0000', '#00ff00', '#0000ff']  # RGB
    label_to_color = {}
    
    for i, lbl in enumerate(correct_labels):
        if i < len(base_colors):
            label_to_color[lbl] = base_colors[i]
        else:
            # Fallback colors
            label_to_color[lbl] = plt.cm.tab10((i + 3) % 10)

    # Colors for incorrect labels (tab10-like, avoid clashing with RGB)
    # tab10: 0=blue, 1=orange, 2=green, 3=red, 4=purple, 5=brown, 6=pink, 7=gray, ...
    # Prefer orange, purple, brown, pink, etc.
    incorrect_palette = ['#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, lbl in enumerate(incorrect_labels):
        label_to_color[lbl] = incorrect_palette[i % len(incorrect_palette)]
    
    # Plot by label group
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(sample_labels_for_plot) if l == lbl]
        x = [distances[i] for i in indices]
        y = [c_potentials[i] for i in indices]
        
        color = label_to_color[lbl]
        
        # Choose marker
        if lbl[0] == lbl[1]:
            marker = 'o'  # correct samples: circle
            # alpha = 0.6
        else:
            marker = 's'  # wrong samples: square
            # alpha = 0.8
            
        ax.scatter(x, y, c=[color], label=f"{lbl}", alpha=0.7, s=40, marker=marker, edgecolors='white', linewidth=0.5)
    
    # Title and axis labels
    ax.set_xlabel(f'Distance to {reference_label} Center ({distance_metric})', fontsize=11)
    ax.set_ylabel(f'C_potential (pos={position})', fontsize=11)
    ax.set_title(f'Distance to Cluster Center vs C_potential\nRef: {reference_label}, Position: {position}',
                fontsize=12, fontweight='bold')
    
    ax.legend(fontsize=9, title="Sample Labels")
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'distance_vs_c_potential_pos{position}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {save_path}")
    plt.close()
    
    # Save plot data
    if SAVE_PLOT_DATA:
        import csv
        data_save_path = os.path.join(save_dir, f'distance_vs_c_potential_data_pos{position}.csv')
        print(f"  Saving plot data to: {data_save_path}")
        
        try:
            with open(data_save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(['gt_digit', 'pred_digit', 'in_carry', 'distance', 'c_potential'])
                
                # Rows
                for dist, c_pot, label in zip(distances, c_potentials, sample_labels_for_plot):
                    writer.writerow([label[0], label[1], label[2], dist, c_pot])
            print(f"  Data saved successfully")
        except Exception as e:
            print(f"  [Error] Failed to save data: {e}")


# ==========================================
# ========== Main ==========
# ==========================================

def main():
    print("=" * 60)
    print("Cluster center distance analysis")
    print("=" * 60)
    
    # Show configuration
    print(f"\n[Configuration]")
    print(f"  Data file: {DATA_FILE_PATH}")
    print(f"  Position: {POSITION_SELECT}")
    print(f"  Layer: {SPECIFIC_LAYER_INDEX if SPECIFIC_LAYER_INDEX is not None else 'all layers (flattened)'}")
    print(f"  Apply norm: {APPLY_MODEL_NORM}")
    print(f"  Sample filter: {SAMPLE_FILTER_MODE}")
    print(f"  Distance metric: {DISTANCE_METRIC}")
    print(f"  Label 1 (GT, Pred, In_carry): {LABEL_1}")
    print(f"  Label 2 (GT, Pred, In_carry): {LABEL_2}")
    print(f"  Label 3 (GT, Pred, In_carry): {LABEL_3}")
    print(f"  Intermediate token viz: {COMPUTE_INTERMEDIATE_TOKENS}")
    if COMPUTE_INTERMEDIATE_TOKENS:
        print(f"  Intermediate steps: {INTERMEDIATE_STEPS}")
    
    # Load data
    print(f"\n[Loading data]")
    flows_list, labels_list, sample_idx_list = load_samples_with_labels(DATA_FILE_PATH, POSITION_SELECT)
    
    # Load sample metadata (for question / c_potential)
    samples_meta = load_samples_meta(DATA_FILE_PATH)
    
    # Initialize model variables
    model_obj = None
    tokenizer = None
    norm_layer = None
    
    # Apply model norm if enabled
    if APPLY_MODEL_NORM:
        flows_list, model_obj, norm_layer = apply_model_norm(flows_list, SPECIFIC_LAYER_INDEX)
    
    # Load model for intermediate tokens if needed and not already loaded
    if COMPUTE_INTERMEDIATE_TOKENS and model_obj is None:
        model_obj, tokenizer, norm_layer = load_model_and_tokenizer()
    elif COMPUTE_INTERMEDIATE_TOKENS and tokenizer is None:
        # Model loaded; load tokenizer only
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"  Tokenizer loaded")
    
    # Count samples per label
    label_counts: Dict[Tuple[int, int, int], int] = {}
    for label in labels_list:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n[Label distribution] (top 20)")
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:20]
    for label, count in sorted_labels:
        print(f"  {label}: {count} samples")
    
    # Filter samples for the three labels
    print(f"\n[Filtering samples]")
    label_names = ["Label 1", "Label 2", "Label 3"]
    target_labels = [LABEL_1, LABEL_2, LABEL_3]
    
    # LABEL_1/2/3 must be correct samples (gt == pred)
    for name, target_label in zip(label_names, target_labels):
        gt, pred, carry = target_label
        if gt != pred:
            raise ValueError(f"Error: {name} {target_label} is not a correct sample (GT Digit != Pred Digit). LABEL_1/2/3 must be correct samples.")
    
    # Map gt_digit -> label for correct samples
    correct_digit_to_label = {}  # gt_digit -> (label_index, target_label)
    for i, target_label in enumerate(target_labels):
        gt_digit = target_label[0]
        correct_digit_to_label[gt_digit] = (i, target_label)
    
    # Validate WRONG_SAMPLE_LABELS
    print(f"\n[Validating wrong-sample labels]")
    print(f"  WRONG_SAMPLE_LABELS: {WRONG_SAMPLE_LABELS}")
    if WRONG_SAMPLE_LABELS:
        valid_gt_digits = set(correct_digit_to_label.keys())
        for wrong_label in WRONG_SAMPLE_LABELS:
            gt, pred, carry = wrong_label
            # Check 1: must be wrong samples
            if gt == pred:
                raise ValueError(f"Error: {wrong_label} in WRONG_SAMPLE_LABELS is a correct sample (GT Digit == Pred Digit). List must contain only wrong samples.")
            # Check 2: gt_digit must appear in LABEL_1/2/3
            if gt not in valid_gt_digits:
                raise ValueError(f"Error: GT Digit={gt} in WRONG_SAMPLE_LABELS {wrong_label} not in LABEL_1/2/3 GT digits {valid_gt_digits}.")
            # Check 3: pred_digit must appear in LABEL_1/2/3
            if pred not in valid_gt_digits:
                raise ValueError(f"Error: Pred Digit={pred} in WRONG_SAMPLE_LABELS {wrong_label} not in LABEL_1/2/3 GT digits {valid_gt_digits}.")
        print(f"  Validation passed")
    
    filtered_flows_list = []
    centers = []
    
    for name, target_label in zip(label_names, target_labels):
        filtered = filter_by_label(flows_list, labels_list, target_label)
        print(f"  {name} {target_label}: {len(filtered)} samples")
        
        if len(filtered) == 0:
            print(f"    Warning: no samples found for {name}!")
            filtered_flows_list.append(None)
            centers.append(None)
        else:
            filtered_flows_list.append(filtered)
            center = compute_center(filtered, SPECIFIC_LAYER_INDEX)
            centers.append(center)
            print(f"    Center vector shape: {center.shape}")
    
    # Pairwise center distances
    print(f"\n[Pairwise center distances ({DISTANCE_METRIC})]")
    for i in range(3):
        for j in range(i + 1, 3):
            if centers[i] is None or centers[j] is None:
                print(f"  {label_names[i]} <-> {label_names[j]}: cannot compute (missing samples)")
            else:
                result = compute_distance(centers[i], centers[j], DISTANCE_METRIC)
                if DISTANCE_METRIC == 'cosine':
                    dist, theta_deg = result
                    print(f"  {label_names[i]} {target_labels[i]} <-> {label_names[j]} {target_labels[j]}: dist={dist:.6f}, theta={theta_deg:.2f}°")
                else:
                    print(f"  {label_names[i]} {target_labels[i]} <-> {label_names[j]} {target_labels[j]}: {result:.6f}")
    
    # Vector norms
    print(f"\n[Vector norms]")
    for i, (name, target_label) in enumerate(zip(label_names, target_labels)):
        if centers[i] is not None:
            norm = np.linalg.norm(centers[i])
            print(f"  {name} {target_label}: {norm:.6f}")
    
    # Intermediate vector tokens (if enabled)
    if COMPUTE_INTERMEDIATE_TOKENS:
        print(f"\n[Computing intermediate vector tokens]")
        
        if model_obj is None or tokenizer is None:
            print("  [Error] Model or tokenizer not loaded; cannot compute intermediate tokens")
        else:
            # Intermediate tokens between the three label center pairs
            intermediate_results: Dict[Tuple[int, int], List[Tuple[float, str, np.ndarray]]] = {}
            
            # Decode tokens at the three centers first
            print(f"\n  [Center point tokens]")
            center_tokens = []
            for i, (name, target_label) in enumerate(zip(label_names, target_labels)):
                if centers[i] is not None:
                    token = decode_vector_to_token(centers[i], model_obj, tokenizer)
                    center_tokens.append(token)
                    print(f"    {name} {target_label}: '{token}'")
                else:
                    center_tokens.append(None)
            
            # Pairwise intermediate tokens
            pairs = [(0, 1), (0, 2), (1, 2)]
            for (i, j) in pairs:
                if centers[i] is not None and centers[j] is not None:
                    # print(f"\n  [{label_names[i]} <-> {label_names[j]} tokens on segment]")
                    results = compute_intermediate_tokens(
                        centers[i], centers[j], model_obj, tokenizer, INTERMEDIATE_STEPS
                    )
                    intermediate_results[(i, j)] = results
                    
                    # Print results
                    # for t, token, _ in results:
                    #     print(f"    t={t:.3f}: '{token}'")
                else:
                    print(f"\n  [{label_names[i]} <-> {label_names[j]}]: skipped (missing center vectors)")
            
            # 2D path visualization
            if intermediate_results:
                print(f"\n[Plotting token path 2D visualization]")
                
                # Wrong-sample data for overlay
                wrong_sample_data = {}
                if WRONG_SAMPLE_LABELS:
                    print(f"\n[Filtering wrong samples]")
                    for wrong_label in WRONG_SAMPLE_LABELS:
                        gt_digit, pred_digit, carry = wrong_label
                        # Indices of correct centers for gt_digit and pred_digit
                        gt_center_idx = None
                        pred_center_idx = None
                        for idx, tl in enumerate(target_labels):
                            if tl[0] == gt_digit:  # match gt_digit
                                gt_center_idx = idx
                            if tl[0] == pred_digit:  # match pred_digit
                                pred_center_idx = idx
                        
                        if gt_center_idx is None or pred_center_idx is None:
                            print(f"  Warning: {wrong_label} has no matching center index, skipping")
                            continue
                        
                        # Samples matching this wrong label
                        wrong_flows = filter_by_label(flows_list, labels_list, wrong_label)
                        print(f"  {wrong_label}: {len(wrong_flows)} samples (GT center: Label {gt_center_idx+1}, Pred center: Label {pred_center_idx+1})")
                        
                        if wrong_flows:
                            wrong_sample_data[wrong_label] = (wrong_flows, gt_center_idx, pred_center_idx)
                
                plot_intermediate_token_paths(
                    centers=centers,
                    labels=label_names,
                    target_labels=target_labels,
                    intermediate_results=intermediate_results,
                    filtered_flows_list=filtered_flows_list,
                    layer_index=SPECIFIC_LAYER_INDEX,
                    wrong_sample_data=wrong_sample_data if WRONG_SAMPLE_LABELS else None,
                    save_dir=f"/home/wenliuyuan/huanglihao/guestwork/plots_cluster/{RAW_SUM}",
                    position=POSITION_SELECT
                )
    
    # Distance vs C_potential scatter (PLOT_SAMPLE_LABELS)
    if PLOT_SAMPLE_LABELS:
        print(f"\n[Processing PLOT_SAMPLE_LABELS]")
        print(f"  PLOT_SAMPLE_LABELS: {PLOT_SAMPLE_LABELS}")
        
        # Reference center = cluster center of PLOT_SAMPLE_LABELS[0]
        reference_label = PLOT_SAMPLE_LABELS[0]
        ref_flows, ref_sample_indices = filter_by_label(
            flows_list, labels_list, reference_label, sample_idx_list
        )
        print(f"  Reference label {reference_label}: {len(ref_flows)} samples")
        
        if len(ref_flows) > 0:
            reference_center = compute_center(ref_flows, SPECIFIC_LAYER_INDEX)
            print(f"  Reference center shape: {reference_center.shape}")
            
            # Scatter plot
            plot_distance_vs_c_potential(
                flows_list=flows_list,
                labels_list=labels_list,
                sample_idx_list=sample_idx_list,
                samples_meta=samples_meta,
                reference_center=reference_center,
                reference_label=reference_label,
                plot_sample_labels=PLOT_SAMPLE_LABELS,
                layer_index=SPECIFIC_LAYER_INDEX,
                distance_metric=DISTANCE_METRIC,
                position=POSITION_SELECT,
                save_dir=f"/home/wenliuyuan/huanglihao/guestwork/plots_cluster/{RAW_SUM}"
            )
        else:
            print(f"  Warning: no samples for reference label {reference_label}; cannot compute reference center")
    
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

