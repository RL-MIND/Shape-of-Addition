from pathlib import Path as _Path
import sys as _sys

_SRC_DIR = _Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import os
import argparse
from utils.cli import parse_position_arg, str_to_bool
# Set thread-control env vars before importing other libs (must be set before import to take effect)
# These env vars affect NumPy, pynndescent, and other libs using OpenMP/BLAS
_UMAP_N_JOBS = os.environ.get('UMAP_N_JOBS', '16')
os.environ['OMP_NUM_THREADS'] = _UMAP_N_JOBS
os.environ['MKL_NUM_THREADS'] = _UMAP_N_JOBS
os.environ['OPENBLAS_NUM_THREADS'] = _UMAP_N_JOBS
os.environ['NUMEXPR_NUM_THREADS'] = _UMAP_N_JOBS

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import trustworthiness
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
try:
    import numba
    import umap
    from umap.aligned_umap import AlignedUMAP
    UMAP_IMPORT_ERROR = None
except ImportError as exc:
    numba = None
    umap = None
    AlignedUMAP = None
    UMAP_IMPORT_ERROR = exc
from utils.flow_utils import load_and_process_data
from utils.flow_utils import (
    load_token_meta_aligned,
    balance_indices_numpy,
    balance_indices_per_position_numpy,
    load_all_token_results,
    select_positions_extended,
    load_samples_meta,
    compute_raw_sum,
    compute_c_potential,
    is_addition_dataset_path,
)



# ========== Switches and parameter settings ==========
# Performance and seed settings
USE_SEED = False  # True, False: whether to fix seed. If False, UMAP uses multithreading (faster but non-deterministic).
# === Two-level parallelism control ===
# Level 1: script-level parallelism (process multiple positions in parallel)
SCRIPT_PARALLEL_WORKERS = 5  # How many positions to process at once; 1 disables script-level parallelism
# Level 2: UMAP internal parallelism (threads per UMAP run)
UMAP_N_JOBS = 8   # Threads per UMAP; -1 uses all cores, positive int (e.g. 4, 8) caps threads
# Total threads ~ SCRIPT_PARALLEL_WORKERS * UMAP_N_JOBS; keep within CPU core count

# Update env vars to match UMAP_N_JOBS (affects pynndescent, etc.)
os.environ['OMP_NUM_THREADS'] = str(UMAP_N_JOBS) if UMAP_N_JOBS > 0 else str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(UMAP_N_JOBS) if UMAP_N_JOBS > 0 else str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(UMAP_N_JOBS) if UMAP_N_JOBS > 0 else str(os.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(UMAP_N_JOBS) if UMAP_N_JOBS > 0 else str(os.cpu_count())
# Sync Numba thread count (affects UMAP embedding optimization)
if numba is not None:
    numba.set_num_threads(UMAP_N_JOBS if UMAP_N_JOBS > 0 else numba.get_num_threads())


def require_umap_dependencies() -> None:
    """Raise a clear error when UMAP/numba cannot be imported in the current environment."""
    if UMAP_IMPORT_ERROR is not None:
        raise ImportError(
            "UMAP plotting requires compatible 'umap-learn' and 'numba' packages. "
            f"The current environment failed to import them: {UMAP_IMPORT_ERROR}"
        ) from UMAP_IMPORT_ERROR

# UMAP evolution continuity control
# 'independent': run each layer independently (default)
# 'sequential': use previous layer embedding as init for next (better visual continuity)
# 'aligned': Aligned UMAP optimizes all layers jointly (strong alignment, higher memory)
UMAP_ALIGNMENT_MODE = 'independent'
# aligned mode .pkl save/load
LOAD_SAVED_DATA = False  # True: load saved .pkl and plot only; False: run normally and save data

# Whether to compute all pos/layer combinations and save uniformly
COMPUTE_ALL_COMBINATIONS = False   # True, False
SKIP_LAYER_0 = True   # True, False: skip layer 0 (when computing all layers; independent of COMPUTE_ALL_COMBINATIONS)

# Model path and norm settings
# MODEL_BACKEND = "quanta"  # "hf" | "quanta"
# MODEL_PATH = "/home/wenliuyuan/activations/quanta_maths/models/add_d10_l2_h3_t40K_s572091"
# MODEL_PATH = "/home/wenliuyuan/activations/quanta_maths/models/add_d20_l2_h3_t80K_s572091"
# MODEL_PATH = "/home/wenliuyuan/activations/quanta_maths/models/mix_d13_l3_h4_t85K_s572091"

MODEL_BACKEND = "hf"
MODEL_PATH = "Qwen/Qwen3-4B"

# MODEL_PATH = "path/to/your/model"

APPLY_MODEL_NORM = True  # True, False: load model and apply final norm to hidden states

# Data path and loading settings
DATA_PATH = "results/activations/plus_num3len10_Qwen3-4B_nocheckall_balance_both"
# DATA_PATH = "results/activations/plus_num3len12_Qwen3-8B_nocheckall_balance"

# DATA_PATH = "results/activations/plus_num3len10_phi-3-mini-4k-instruct_nocheckall_balance"

# DATA_PATH = "results/activations/plus_num3len10_gemma-3-4b-it_nocheckall_balance"
# DATA_PATH = "results/activations/plus_num3len10_gemma-2-2b-it_nocheckall_balance"

# DATA_PATH = "results/activations/plus_num4len10_Qwen3-4B_nocheckall_balance_trainval.h5"

# DATA_PATH = "results/quanta_add_d10_l2_h3_num2len10_n10000"
# DATA_PATH = "results/quanta_add_d20_l2_h3_num2len10_n10000"
# DATA_PATH = "results/quanta_add_d20_l2_h3_num2len20_n10000"
# DATA_PATH = "results/quanta_mix_d13_l3_h4_num2len13_n10000"

BALANCE_MODE = 'none'  # 'none', 'normal', 'strong'
POSITION_SELECT = 4                        # Position select: 'all'/'all_no_extra'/'extra' or int/list
FEATURE_TYPE = "post_ffn"                    # 'post_attn' | 'post_ffn'
POOLING = 'none'                           # None | 'avg' | 'max'
USE_PCA = False                            # Whether to run PCA first
PCA_DIM = 100
PCA_MODE = "per_layer"                    # 'per_layer' or 'concat_first'

# Save settings
SAVE_PLOT = True    # True, False
SAVE_PLOT_DATA = True # True, False
SAVE_DIR = "results/figures/umap"

# UMAP core parameters
UMAP_SELECT_SPECIFIC_LAYER = True   # True, False: use only specified layer (only when COMPUTE_ALL_COMBINATIONS is False)
UMAP_LAYER_INDEX = 36   # quanta flow axis: [embed, layer0, layer1]
# Marker mode settings
# "point": plain dot
# "pred": use predicted digits (pred_digits)
# "input": use input digits (input_digits / input_tokens_list)
# "in_carry": incoming carry (from lower position)
# "out_carry": outgoing carry (to higher position)
# "raw_sum": digit sum at position mod 10 (addition datasets only)
# "gt_carry": show (GT, In_Carry), e.g. "(4, 1)"
# "gt_pred_carry": show (GT, Pred, In_Carry), e.g. "(4, 3, 1)"
# "gt_pred_carry_potential": like gt_pred_carry plus C_potential, e.g. "(4,3,1,0.52)"
# "next_raw_sum": raw_sum at next position (e.g. pos=4 shows pos=5 raw_sum)
# "next_pred": pred at next position (e.g. pos=4 shows pos=5 pred)
# "next_incoming_carry": incoming carry at next position
UMAP_MARKER_MODE = "gt_carry"
# Color mode (color only, does not affect marker)
# "consistent": color by correctness at current pos (blue=correct, red=wrong)
# "prefix_is_correct": color by prefix correctness (blue=all correct, red=any error)
UMAP_COLOR_MODE = "consistent"  # "consistent", "prefix_is_correct"
# Pred filter: filter samples by predicted digit before UMAP
# 'all': no filter (use all samples)
# int: keep only samples with that pred (e.g. 5 keeps pred=5)
# list[int]: keep preds in list (e.g. [0,1,2] keeps pred 0/1/2)
UMAP_PRED_FILTER = 'all'  # 'all', int, or list[int]    [3,4,5]
UMAP_DIM = 2    # 2, 3
UMAP_MAX_POINTS = 1000
# Max positive/negative sample counts (None defaults to UMAP_MAX_POINTS)
UMAP_MAX_POS_POINTS = 400  # e.g. 1000
UMAP_MAX_NEG_POINTS = 200  # e.g. 1000
# Balance in_carry counts when sampling with UMAP_MAX_POINTS
# (only when COMPUTE_ALL_COMBINATIONS = False)
BALANCE_IN_CARRY_SAMPLING = False  # True: balance across carry values; False: no balancing
TRUST_N_NEIGHBORS = 50
UMAP_N_NEIGHBORS = 500
UMAP_MIN_DIST = 0.3 # 0.1, 0.3, 0.5
UMAP_METRIC = "cosine"  # "cosine" | "euclidean"
UMAP_SUPERVISED = False  # True=supervised, False=unsupervised
# UMAP grid inverse-transform decode (only when APPLY_MODEL_NORM=True)
# Grid-sample UMAP space, inverse-transform each point and decode to token
UMAP_GRID_DECODE_2D = False  # True: enable grid decode when UMAP_DIM=2; False: disable
UMAP_GRID_DECODE_3D = False  # True: enable grid decode when UMAP_DIM=3; False: disable
UMAP_GRID_SIZE = 100       # Grid resolution (20 => 20x20 = 400 points)
UMAP_GRID_WORKERS = 8    # Threads for parallel grid processing
UMAP_GRID_USE_KNN = True  # True: KNN regressor for high-dim vectors; False: UMAP inverse (slow)
UMAP_GRID_KNN_K = 5     # KNN K (number of neighbors)
# Interactive HTML export (only when COMPUTE_ALL_COMBINATIONS is False)
EXPORT_INTERACTIVE_HTML = True  # True: interactive HTML (hover Q, GT, Pred); False: static PNG only

# --- load_and_process_data filter/balance config ---
# Sample filter mode
SAMPLE_FILTER_MODE = 'all'  # 'correct', 'incorrect', 'all'
# Source / INPUT_POS filters
ALLOWED_INPUT_DIGITS = None   # Allowed source input digits; None disables
ALLOWED_OUTPUT_DIGITS = None # [1, 2, 3, 4, 5, 6, 7, 8]   # Allowed source output digits; None disables
ALLOWED_INCOMING_CARRIES = None   # Allowed source incoming carries; None disables

# Target / TARGET_POS filters
ALLOWED_TARGET_INPUT_DIGITS = None   # Allowed target input digits; None disables
ALLOWED_TARGET_OUTPUT_DIGITS = None   # Allowed target output digits; None disables
ALLOWED_TARGET_INCOMING_CARRIES = None   # Allowed target incoming carries; None disables
# Result-length filter switch (default True)
FILTER_BY_RESULT_LEN = False  # True, False
# Data balance mode
BALANCE_MODE = 'none'  # 'none', 'normal', 'strong'
BALANCE_TARGET_CLASSES = False  # True, False
# raw_sum_mod_10
RAW_SUM_MOD_10 = False   # True, False
# SEED
SEED = 42   # Random seed

# Load pretrained classifier (.pt from classifier.py)
# When True, load .pt from SAVED_MODEL_PATH,
# use saved filter params to override local config above,
# run inference and highlight wrong predictions in green
# (only when COMPUTE_ALL_COMBINATIONS is False)
LOAD_PRETRAINED_MODEL = False  # True, False

# Pretrained model path (.pt; only when LOAD_PRETRAINED_MODEL = True)
# SAVED_MODEL_PATH = 'results/checkpoints/mlp_plus_num3len10_Qwen3-4B_checkall_balance_4_none_layer36_is_correct.pt'
# SAVED_MODEL_PATH = 'results/checkpoints/mlp_plus_num3len10_Qwen3-4B_checkall_balance_5_none_layer36_last_is_correct.pt'
# SAVED_MODEL_PATH = 'results/checkpoints/mlp_plus_num3len10_Qwen3-4B_checkall_balance_3_none_layer36_next_is_correct.pt'
SAVED_MODEL_PATH = 'results/checkpoints/mlp_plus_num3len10_Qwen3-4B_checkall_balance_consistent_tgt4_none_layer36_prefix_is_correct.pt'

# Plot UMAP on validation set only (only when LOAD_PRETRAINED_MODEL = True)
# True: use val_indices from .pt as plot samples
# False: filter all data then sample with get_sampled_indices
USE_VAL_ONLY = True  # True, False

# GIF export (aligned mode only)
EXPORT_GIF = True   # True, False: export GIF animation
GIF_FPS = 3         # GIF frame rate (frames per second)
GIF_INTERVAL = 500  # Interval per frame (ms)

# ========== Parameter validation ==========

def get_model_embeddings(model):
    """
    Automatically locate the model Embeddings layer.
    Supports:
    - Qwen/Llama (model.model.embed_tokens)
    - Gemma 3 (model.model.language_model.embed_tokens)
    - Others (model.transformer.wte, etc.; add as needed)
    """
    # 1. Gemma 3
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
        return model.model.language_model.embed_tokens
    
    # 2. Qwen / Llama
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens

    # 3. Fallback
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
        
    raise AttributeError("Could not automatically locate Embeddings layer")

def validate_parameters():
    """Validate parameter configuration compatibility."""
    require_umap_dependencies()
    if MODEL_BACKEND not in {"hf", "quanta"}:
        raise ValueError(
            f"Invalid MODEL_BACKEND: {MODEL_BACKEND}. "
            "Valid values: 'hf', 'quanta'"
        )
    if UMAP_COLOR_MODE not in {"consistent", "prefix_is_correct"}:
        raise ValueError(
            f"Invalid UMAP_COLOR_MODE: {UMAP_COLOR_MODE}. "
            f"Valid values: 'consistent', 'prefix_is_correct'"
        )
    if (
        UMAP_COLOR_MODE == "prefix_is_correct"
        and not COMPUTE_ALL_COMBINATIONS
        and not isinstance(POSITION_SELECT, (int, np.integer))
    ):
        raise ValueError(
            f"UMAP_COLOR_MODE='prefix_is_correct' requires a single numeric POSITION_SELECT. "
            f"Current POSITION_SELECT={POSITION_SELECT}"
        )

    if EXPORT_INTERACTIVE_HTML and COMPUTE_ALL_COMBINATIONS:
        raise ValueError("EXPORT_INTERACTIVE_HTML requires COMPUTE_ALL_COMBINATIONS=False.")

    if MODEL_BACKEND == "quanta" and (UMAP_GRID_DECODE_2D or UMAP_GRID_DECODE_3D):
        raise ValueError("The quanta backend does not support UMAP grid inverse decode.")
    
    if LOAD_PRETRAINED_MODEL and UMAP_ALIGNMENT_MODE in ['sequential', 'aligned']:
        print("=" * 60)
        print("Error: LOAD_PRETRAINED_MODEL does not support UMAP_ALIGNMENT_MODE 'sequential' or 'aligned' yet")
        print(f"Current setting: LOAD_PRETRAINED_MODEL = {LOAD_PRETRAINED_MODEL}")
        print(f"Current setting: UMAP_ALIGNMENT_MODE = {UMAP_ALIGNMENT_MODE}")
        print("=" * 60)
        print("Set UMAP_ALIGNMENT_MODE to 'independent', or disable LOAD_PRETRAINED_MODEL")
        raise ValueError(
            f"LOAD_PRETRAINED_MODEL={LOAD_PRETRAINED_MODEL} "
            f"incompatible with UMAP_ALIGNMENT_MODE={UMAP_ALIGNMENT_MODE}. "
            f"Use UMAP_ALIGNMENT_MODE='independent'"
        )
    
    # Validate pretrained model path
    if LOAD_PRETRAINED_MODEL:
        if not os.path.isfile(SAVED_MODEL_PATH):
            raise FileNotFoundError(
                f"Pretrained model file not found: {SAVED_MODEL_PATH}\n"
                f"Check SAVED_MODEL_PATH or set LOAD_PRETRAINED_MODEL to False"
            )


def validate_batch_mode_capabilities():
    """Validate batch mode does not enable unsupported heavy features."""
    if not COMPUTE_ALL_COMBINATIONS:
        return

    blocked_features = []
    if LOAD_PRETRAINED_MODEL:
        blocked_features.append("LOAD_PRETRAINED_MODEL=True")
    if UMAP_DIM == 2 and UMAP_GRID_DECODE_2D:
        blocked_features.append("UMAP_GRID_DECODE_2D=True")
    if UMAP_DIM == 3 and UMAP_GRID_DECODE_3D:
        blocked_features.append("UMAP_GRID_DECODE_3D=True")

    if blocked_features:
        blocked_text = ", ".join(blocked_features)
        raise ValueError(
            "COMPUTE_ALL_COMBINATIONS=True does not support: "
            f"{blocked_text}. Disable these before batch mode."
        )


def validate_position_consistency(saved_config, umap_position_select):
    """
    Verify POSITION_SELECT matches the true label position from .pt position_select + train_target.
    
    For next_* targets: activations at position_select=N, labels from pos=N+1,
                     so UMAP POSITION_SELECT should be N+1.
    For last_* targets: activations at N, labels from N-1,
                     so POSITION_SELECT should be N-1.
    Other targets: POSITION_SELECT should equal position_select.
    
    Args:
        saved_config: config dict saved in .pt
        umap_position_select: current script POSITION_SELECT
    
    Raises:
        ValueError: when positions do not match
    """
    train_target = saved_config.get('train_target', 'is_correct')
    model_position_select = saved_config.get('position_select')
    
    if model_position_select is None or not isinstance(model_position_select, int):
        raise ValueError(
            f"position_select/target_pos unavailable in .pt; parsed value {model_position_select}; cannot validate"
        )
    if not isinstance(umap_position_select, int):
        raise ValueError(
            f"POSITION_SELECT is {umap_position_select}; cannot validate"
        )
    # Compute true label position
    if train_target.startswith('next_'):
        true_label_pos = model_position_select + 1
    elif train_target.startswith('last_'):
        true_label_pos = model_position_select - 1
    else:
        true_label_pos = model_position_select
    
    if true_label_pos != umap_position_select:
        raise ValueError(
            f"Position mismatch!\n"
            f"  .pt model position_select={model_position_select}, train_target='{train_target}'\n"
            f"  True label position = {true_label_pos}\n"
            f"  But current POSITION_SELECT = {umap_position_select}\n"
            f"  Set POSITION_SELECT to {true_label_pos}"
        )
    
    print(f"  [Position check OK] .pt position_select={model_position_select}, "
          f"train_target='{train_target}', true label position={true_label_pos}, "
          f"POSITION_SELECT={umap_position_select}")


def load_pretrained_and_get_val_indices(saved_model_path):
    """
    Load pretrained .pt, extract saved config and val_indices.
    
    Args:
        saved_model_path: path to .pt file
    
    Returns:
        config_dict: saved filter/training params
        val_indices: validation indices (numpy array)
        model_state: model weight state_dict
    """
    print(f"\n====== Loading pretrained model: {saved_model_path} ======")
    checkpoint = torch.load(saved_model_path, map_location='cpu', weights_only=False)

    # Position field compatibility:
    # - Legacy: position_select
    # - classifier.py: target_pos / input_pos
    # - Per-layer/position eval: position
    position_select_compat = checkpoint.get('position_select')
    if position_select_compat is None:
        position_select_compat = checkpoint.get('target_pos')
    if position_select_compat is None:
        position_select_compat = checkpoint.get('position')

    def resolve_allowed_filter(allowed_key, legacy_flag_key, legacy_default):
        """
        Compatible with old/new checkpoints:
        - New: only allowed_*; None means disabled
        - Old: filter_by_* and allowed_*; filter_by_=False means disabled
        """
        allowed_val = checkpoint.get(allowed_key, None)
        if legacy_flag_key in checkpoint:
            if checkpoint.get(legacy_flag_key, False):
                return allowed_val if allowed_val is not None else legacy_default
            return None
        return allowed_val
    
    # Extract config parameters
    config_dict = {
        'model_type': checkpoint.get('model_type', 'mlp'),
        'feature_type': checkpoint.get('feature_type', 'flows'),
        'train_target': checkpoint.get('train_target', 'is_correct'),
        'num_classes': checkpoint.get('num_classes', 2),
        'position_select': position_select_compat,
        'target_pos': checkpoint.get('target_pos'),
        'input_pos': checkpoint.get('input_pos'),
        'pooling_type': checkpoint.get('pooling_type', 'none'),
        'use_pca': checkpoint.get('use_pca', False),
        'pca_dim': checkpoint.get('pca_dim', 512),
        'seq_len': checkpoint.get('seq_len'),
        'feature_dim': checkpoint.get('feature_dim'),
        'best_val_auc': checkpoint.get('best_val_auc'),
        'best_epoch': checkpoint.get('best_epoch'),
        'total_samples': checkpoint.get('total_samples'),
        # Filter parameters
        'sample_filter_mode': checkpoint.get('sample_filter_mode', 'all'),
        'filter_by_result_len': checkpoint.get('filter_by_result_len', True),
        'allowed_input_digits': resolve_allowed_filter('allowed_input_digits', 'filter_by_input_digits', [3]),
        'allowed_output_digits': resolve_allowed_filter('allowed_output_digits', 'filter_by_output_digits', [4]),
        'allowed_incoming_carries': resolve_allowed_filter('allowed_incoming_carries', 'filter_by_incoming_carry', [1]),
        'allowed_target_input_digits': resolve_allowed_filter('allowed_target_input_digits', 'filter_by_target_input_digits', [3]),
        'allowed_target_output_digits': resolve_allowed_filter('allowed_target_output_digits', 'filter_by_target_output_digits', [4]),
        'allowed_target_incoming_carries': resolve_allowed_filter('allowed_target_incoming_carries', 'filter_by_target_incoming_carry', [1]),
        'balance_mode': checkpoint.get('balance_mode', 'none'),
        'balance_target_classes': checkpoint.get('balance_target_classes', False),
        'apply_model_norm': checkpoint.get('apply_model_norm', True),
        'raw_sum_mod_10': checkpoint.get('raw_sum_mod_10', False),
        'seed': checkpoint.get('seed', 42),
        # Layer/position info
        'layer': checkpoint.get('layer'),
        'position': checkpoint.get('position'),
        'cumulative_layers': checkpoint.get('cumulative_layers', False),
    }
    
    # Extract val_indices
    val_indices = checkpoint.get('val_indices')
    if val_indices is not None:
        if isinstance(val_indices, torch.Tensor):
            val_indices = val_indices.cpu().numpy()
        elif not isinstance(val_indices, np.ndarray):
            val_indices = np.array(val_indices)
    
    # Extract train_indices
    train_indices = checkpoint.get('train_indices')
    if train_indices is not None:
        if isinstance(train_indices, torch.Tensor):
            train_indices = train_indices.cpu().numpy()
        elif not isinstance(train_indices, np.ndarray):
            train_indices = np.array(train_indices)
    config_dict['train_indices'] = train_indices
    
    model_state = checkpoint.get('model_state')
    
    # Print loaded config
    print(f"  Model type: {config_dict['model_type']}")
    print(f"  Train target: {config_dict['train_target']}")
    print(f"  Num classes: {config_dict['num_classes']}")
    print(f"  Best val AUC: {config_dict['best_val_auc']}")
    print(f"  Best epoch: {config_dict['best_epoch']}")
    print(f"  Total samples: {config_dict['total_samples']}")
    if val_indices is not None:
        print(f"  Val set size: {len(val_indices)}")
    if train_indices is not None:
        print(f"  Train set size: {len(train_indices)}")
    print(f"  Filter params:")
    print(f"    sample_filter_mode: {config_dict['sample_filter_mode']}")
    print(f"    filter_by_result_len: {config_dict['filter_by_result_len']}")
    print(f"    balance_mode: {config_dict['balance_mode']}")
    print(f"    apply_model_norm: {config_dict['apply_model_norm']}")
    if config_dict.get('layer') is not None:
        print(f"  Layer index: {config_dict['layer']}")
    if config_dict.get('position') is not None:
        print(f"  Position: {config_dict['position']}")
    print("=" * 60)
    
    return config_dict, val_indices, model_state

def evaluate_pretrained_model(X_all_raw, y_binary, saved_config, saved_model_state, saved_val_indices, seq_len, feature_dim,
                               current_umap_position=None, data_path=None, load_kwargs=None):
    """
    Load pretrained classifier and run inference/eval on full data.

    For next_* or last_* train_target, classifier activations come from
    a position offset ±1 from the UMAP plot position. This function detects offset
    and reloads activations from the correct position.

    Args:
        X_all_raw: feature matrix at current UMAP position
        y_binary: binary labels
        saved_config: config dict from .pt
        saved_model_state: model state_dict
        saved_val_indices: validation indices
        seq_len: sequence length (num layers)
        feature_dim: feature dimension
        current_umap_position: current UMAP plot position (int)
        data_path: data path for reload when needed
        load_kwargs: extra args for load_and_process_data
    """
    from models import create_model
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
    import torch
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Detect position offset ---
    train_target = saved_config.get('train_target', 'is_correct')
    saved_position = saved_config.get('position_select')
    
    # next_*: activations at pos predict attributes at pos+1
    # last_*: activations at pos predict attributes at pos-1
    # If saved position_select differs from current UMAP position,
    # reload activations from saved_position
    need_reload = False
    classifier_position = current_umap_position  # default: use current UMAP position data
    
    if current_umap_position is not None and saved_position is not None and isinstance(current_umap_position, int) and isinstance(saved_position, int):
        if train_target.startswith('next_'):
            # next_*: classifier trained at N, predicts N+1
            # UMAP at N+1 needs activations at N
            if current_umap_position != saved_position:
                classifier_position = saved_position
                need_reload = True
                print(f"  [Position offset] train_target='{train_target}': classifier trained at pos={saved_position}, "
                      f"UMAP at {current_umap_position}, reloading activations at pos={saved_position}")
        elif train_target.startswith('last_'):
            # last_*: classifier at N, predicts N-1
            # UMAP at N-1 needs activations at N
            if current_umap_position != saved_position:
                classifier_position = saved_position
                need_reload = True
                print(f"  [Position offset] train_target='{train_target}': classifier trained at pos={saved_position}, "
                      f"UMAP at {current_umap_position}, reloading activations at pos={saved_position}")
    
    if need_reload and data_path is not None:
        import io
        import contextlib
        from utils.flow_utils import load_and_process_data
        reload_kwargs = dict(load_kwargs) if load_kwargs else {}
        # Suppress verbose output (summary only)
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            X_clf, y_all_clf, _, _, _, seq_len_clf, feature_dim_clf, _, _, _ = load_and_process_data(
                data_path, position_select=classifier_position, feature_type=saved_config.get('feature_type', FEATURE_TYPE),
                **reload_kwargs
            )
        print(f"  [Reload done] pos={classifier_position}, samples={len(X_clf)}")
        X_np = X_clf.numpy() if isinstance(X_clf, torch.Tensor) else np.asarray(X_clf)
        seq_len = seq_len_clf
        feature_dim = feature_dim_clf
        # Use y_all (train_target labels) not y_binary (raw is_correct)
        # Classifier was trained on train_target labels
        # e.g. last_is_correct: y_all = is_correct at pos-1, y_binary = at current pos
        y_binary = y_all_clf
    else:
        X_np = X_all_raw.numpy() if isinstance(X_all_raw, torch.Tensor) else np.asarray(X_all_raw)
    
    X_np = X_np.reshape(len(X_np), seq_len, feature_dim)
    
    layer_idx = saved_config.get('layer')
    if layer_idx is not None and not saved_config.get('cumulative_layers', False):
        if layer_idx < seq_len:
            X_layer = X_np[:, layer_idx, :]
            print(f"  Using Layer {layer_idx} data for inference")
        else:
            X_layer = X_np[:, -1, :]
            print(f"  Warning: layer_idx ({layer_idx}) out of range, using last layer")
    elif UMAP_SELECT_SPECIFIC_LAYER:
        layer_idx = UMAP_LAYER_INDEX
        if layer_idx >= seq_len:
            layer_idx = 0
        X_layer = X_np[:, layer_idx, :]
        print(f"  Using Layer {layer_idx} data for inference")
    else:
        X_layer = X_np.reshape(len(X_np), -1)
        print(f"  Using all layers (flattened) for inference")
    
    input_dim = X_layer.shape[1]
    num_classes = saved_config['num_classes']
    
    clf_model = create_model(
        model_type=saved_config['model_type'],
        input_dim=input_dim,
        seq_len=seq_len,
        feature_dim=feature_dim,
        num_classes=num_classes,
        device=device,
    )
    clf_model.load_state_dict(saved_model_state)
    clf_model.eval()
    clf_model = clf_model.to(device)
    
    X_tensor = torch.tensor(X_layer, dtype=torch.float32).to(device)
    y_binary_np = y_binary.numpy() if isinstance(y_binary, torch.Tensor) else np.asarray(y_binary)
    
    all_preds = []
    all_probs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = clf_model(batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    preds_all_cpu = np.array(all_preds)
    probs_all_cpu = np.array(all_probs)
    correctness = (preds_all_cpu == y_binary_np)
    
    # --- Full-data evaluation ---
    acc_all = accuracy_score(y_binary_np, preds_all_cpu)
    print(f"  Inference done: full Acc={acc_all:.4f}, correct={np.sum(correctness)}, wrong={np.sum(~correctness)}")
    
    # --- Validation evaluation ---
    if saved_val_indices is not None:
        y_val = y_binary_np[saved_val_indices]
        preds_val = preds_all_cpu[saved_val_indices]
        probs_val = probs_all_cpu[saved_val_indices]
        
        if len(y_val) > 0:
            acc_val = accuracy_score(y_val, preds_val)
            f1_val = f1_score(y_val, preds_val, average='binary')
            
            try:
                auc_val = roc_auc_score(y_val, probs_val)
            except ValueError:
                auc_val = float('nan')
                
            try:
                tn, fp, fn, tp = confusion_matrix(y_val, preds_val, labels=[0, 1]).ravel()
                print(f"    Validation (N={len(y_val)}): Acc={acc_val:.4f}, AUC={auc_val:.4f}, F1={f1_val:.4f}")
                print(f"    Val details: TP={tp}, TPR={tp/(tp+fn)*100:.2f}%, TN={tn}, TNR={tn/(tn+fp)*100:.2f}%")
            except ValueError:
                pass
    
    del clf_model, X_tensor
    torch.cuda.empty_cache()
    
    return correctness


def filter_by_pred(pred_digits, *arrays):
    """
    Filter samples by UMAP_PRED_FILTER.
    
    Args:
        pred_digits: predicted digit array
        *arrays: other arrays to filter in sync (X_all_raw, y_all, gt_digits, etc.)
    
    Returns:
        filtered_indices and filtered arrays (same order as input arrays)
        If UMAP_PRED_FILTER = 'all', returns all samples
    """
    if UMAP_PRED_FILTER == 'all':
        # No filter; return all indices
        filter_mask = np.ones(len(pred_digits), dtype=bool)
        print(f"  UMAP_PRED_FILTER = 'all': no filter, using all {len(pred_digits)} samples")
    elif isinstance(UMAP_PRED_FILTER, int):
        # Filter single digit
        filter_mask = (pred_digits == UMAP_PRED_FILTER)
        count = np.sum(filter_mask)
        print(f"  UMAP_PRED_FILTER = {UMAP_PRED_FILTER}: pred={UMAP_PRED_FILTER}, {count}/{len(pred_digits)} samples")
    elif isinstance(UMAP_PRED_FILTER, (list, tuple)):
        # Filter multiple digits
        filter_mask = np.isin(pred_digits, UMAP_PRED_FILTER)
        count = np.sum(filter_mask)
        print(f"  UMAP_PRED_FILTER = {UMAP_PRED_FILTER}: pred in {UMAP_PRED_FILTER}, {count}/{len(pred_digits)} samples")
    else:
        raise ValueError(f"Invalid UMAP_PRED_FILTER: {UMAP_PRED_FILTER}; must be 'all', int, or list[int]")
    
    if np.sum(filter_mask) == 0:
        raise ValueError(f"UMAP_PRED_FILTER = {UMAP_PRED_FILTER} left no samples; check settings")
    
    # Filtered indices for mapping back to original arrays
    filtered_indices = np.where(filter_mask)[0]
    
    # Filter all input arrays
    filtered_arrays = []
    for arr in arrays:
        if arr is None:
            filtered_arrays.append(None)
        elif isinstance(arr, np.ndarray):
            filtered_arrays.append(arr[filter_mask])
        elif isinstance(arr, torch.Tensor):
            filtered_arrays.append(arr[filter_mask])
        else:
            # Assume indexable sequence
            filtered_arrays.append(np.asarray(arr)[filter_mask])
    
    return filtered_indices, pred_digits[filter_mask], *filtered_arrays

def get_pred_filter_suffix():
    """
    Build filename suffix from UMAP_PRED_FILTER.
    
    Returns:
        str: suffix like '_all', '_5', '_0_1_2'
    """
    if UMAP_PRED_FILTER == 'all':
        return '_all'
    elif isinstance(UMAP_PRED_FILTER, int):
        return f'_{UMAP_PRED_FILTER}'
    elif isinstance(UMAP_PRED_FILTER, (list, tuple)):
        # Join list digits with underscores
        nums_str = '_'.join(str(n) for n in sorted(UMAP_PRED_FILTER))
        return f'_{nums_str}'
    else:
        return '_unknown'


def get_color_mode_suffix():
    """Color mode suffix: consistent adds none; others appended to avoid overwrite."""
    if UMAP_COLOR_MODE == "consistent":
        return ""
    return f"_color_{UMAP_COLOR_MODE}"


def get_color_legend_labels(with_prediction_state=False):
    """Return blue/red legend labels for UMAP_COLOR_MODE."""
    if UMAP_COLOR_MODE == "prefix_is_correct":
        blue_label = "Prefix Correct"
        red_label = "Prefix Incorrect"
    else:
        blue_label = "Positive"
        red_label = "Negative"

    if with_prediction_state:
        blue_label += " (correct)"
        red_label += " (correct)"
    return blue_label, red_label


def compute_prefix_is_correct_labels(data_path, position, sample_indices):
    """
    Compute prefix correctness labels (same as classifier TRAIN_TARGET=prefix_is_correct).
    Returns:
        np.ndarray[int64], 1=all prefix correct, 0=any prefix error
    """
    sample_indices = np.asarray(sample_indices, dtype=np.int64)

    if not isinstance(position, (int, np.integer)):
        raise ValueError(
            f"UMAP_COLOR_MODE='prefix_is_correct' requires numeric position; current position={position}"
        )

    current_pos = int(position)
    if current_pos <= 0:
        # pos=0 has no prefix; treat as all-prefix-correct for color.
        print("  Warning: position=0 has no prefix; prefix_is_correct colors all blue.")
        return np.ones(len(sample_indices), dtype=np.int64)

    prefix_positions = list(range(current_pos))
    prefix_results = load_all_token_results(
        data_path, prefix_positions, feature_type=FEATURE_TYPE, verbose=False
    )

    prefix_maps = {}
    for p in prefix_positions:
        pos_data = prefix_results.get(p, {})
        labels = pos_data.get("labels", [])
        s_indices = pos_data.get("sample_indices")
        if s_indices is None:
            s_indices = list(range(len(labels)))
        prefix_maps[p] = {int(s_idx): bool(lbl) for s_idx, lbl in zip(s_indices, labels)}

    out = np.zeros(len(sample_indices), dtype=np.int64)
    for i, s_idx in enumerate(sample_indices):
        prefix_all_correct = True
        for p in prefix_positions:
            if not prefix_maps.get(p, {}).get(int(s_idx), False):
                prefix_all_correct = False
                break
        out[i] = 1 if prefix_all_correct else 0

    return out


def compute_umap_color_labels(data_path, position, sample_indices, consistent_labels):
    """
    Binary labels for coloring (1=blue, 0=red).
    Color only; marker text/shape should still use original labels.
    """
    consistent_labels = np.asarray(consistent_labels, dtype=np.int64)

    if UMAP_COLOR_MODE == "consistent":
        return consistent_labels

    if UMAP_COLOR_MODE == "prefix_is_correct":
        color_labels = compute_prefix_is_correct_labels(data_path, position, sample_indices)
        print(
            f"  UMAP_COLOR_MODE=prefix_is_correct: "
            f"prefix all correct {np.sum(color_labels == 1)}, prefix has errors {np.sum(color_labels == 0)}"
        )
        return color_labels

    raise ValueError(f"Unsupported UMAP_COLOR_MODE: {UMAP_COLOR_MODE}")

def compute_marker_mode_values(data_path, position, sample_indices, pred_digits):
    """
    Compute marker-mode values: raw_sum, next_raw_sum, next_pred, next_incoming_carry.
    
    Args:
        data_path: data file path
        position: current position
        sample_indices: sample index list
        pred_digits: predicted digits
    
    Returns:
        dict: marker values
    """
    result = {
        'raw_sum': None,
        'next_raw_sum': None,
        'next_pred': None,
        'next_incoming_carry': None,
        'c_potential': None
    }
    
    # Return early if marker mode does not need these
    if UMAP_MARKER_MODE not in ['raw_sum', 'next_raw_sum', 'next_pred', 'next_incoming_carry', 'gt_pred_carry_potential', 'gt_carry']:
        return result
    
    # Check addition dataset
    if not is_addition_dataset_path(data_path):
        print(f"  Warning: UMAP_MARKER_MODE='{UMAP_MARKER_MODE}' only supports addition datasets")
        return result
    
    # Load sample metadata
    samples_meta = load_samples_meta(data_path)
    if not samples_meta:
        print(f"  Warning: cannot load samples_meta; {UMAP_MARKER_MODE} mode unavailable")
        return result
    
    n_samples = len(sample_indices)
    
    # Compute raw_sum (digit sum at current position)
    if UMAP_MARKER_MODE == 'raw_sum':
        raw_sum_values = np.full(n_samples, -1, dtype=np.int64)
        for i, s_idx in enumerate(sample_indices):
            sample_info = samples_meta.get(int(s_idx))
            if sample_info is not None:
                question = sample_info.get("question", "")
                val = compute_raw_sum(question, position)
                if val != -1:
                    raw_sum_values[i] = val
        result['raw_sum'] = raw_sum_values
    
    # Compute c_potential
    elif UMAP_MARKER_MODE == 'gt_pred_carry_potential':
        if not isinstance(position, int):
            print(f"  Warning: gt_pred_carry_potential requires numeric position ({position})")
            return result
        c_potential_values = np.full(n_samples, -1.0, dtype=np.float64)
        for i, s_idx in enumerate(sample_indices):
            sample_info = samples_meta.get(int(s_idx))
            if sample_info is not None:
                question = sample_info.get("question", "")
                t_val = compute_c_potential(question, position)
                c_potential_values[i] = t_val
        result['c_potential'] = c_potential_values
    
    # Compute next_raw_sum
    elif UMAP_MARKER_MODE == 'next_raw_sum':
        if not isinstance(position, int):
            print(f"  Warning: next_raw_sum requires numeric position ({position})")
            return result
        next_pos = position + 1
        next_raw_sum_values = np.full(n_samples, -1, dtype=np.int64)
        for i, s_idx in enumerate(sample_indices):
            sample_info = samples_meta.get(int(s_idx))
            if sample_info is not None:
                question = sample_info.get("question", "")
                val = compute_raw_sum(question, next_pos)
                if val != -1:
                    next_raw_sum_values[i] = val
        result['next_raw_sum'] = next_raw_sum_values
    
    # Compute next_pred
    elif UMAP_MARKER_MODE == 'next_pred':
        if not isinstance(position, int):
            print(f"  Warning: next_pred requires numeric position ({position})")
            return result
        next_pos = position + 1
        
        # Load next-position data
        try:
            all_token_results = load_all_token_results(
                data_path, next_pos, feature_type=FEATURE_TYPE, verbose=False
            )
            if next_pos not in all_token_results:
                print(f"  Warning: next position {next_pos} data not found")
                return result
            
            next_pos_data = all_token_results[next_pos]
            next_preds = next_pos_data.get("preds", [])
            next_indices = next_pos_data.get("sample_indices")
            if next_indices is None:
                next_indices = list(range(len(next_preds)))
            
            # Build sample_idx -> pred map
            next_pred_map = {int(idx): str(pred) for idx, pred in zip(next_indices, next_preds)}
            
            next_pred_values = np.full(n_samples, -1, dtype=np.int64)
            for i, s_idx in enumerate(sample_indices):
                pred_char = next_pred_map.get(int(s_idx))
                if pred_char is not None and str(pred_char).isdigit():
                    next_pred_values[i] = int(pred_char)
            result['next_pred'] = next_pred_values
        except Exception as e:
            print(f"  Warning: failed to load next-position data: {e}")
    
    # Compute next_incoming_carry
    elif UMAP_MARKER_MODE == 'next_incoming_carry':
        if not isinstance(position, int):
            print(f"  Warning: next_incoming_carry requires numeric position ({position})")
            return result
        next_pos = position + 1
        
        # Load next-position data
        try:
            all_token_results = load_all_token_results(
                data_path, next_pos, feature_type=FEATURE_TYPE, verbose=False
            )
            if next_pos not in all_token_results:
                print(f"  Warning: next position {next_pos} data not found")
                return result
            
            next_pos_data = all_token_results[next_pos]
            next_carries = next_pos_data.get("incoming_carries", [])
            next_indices = next_pos_data.get("sample_indices")
            if next_indices is None:
                next_indices = list(range(len(next_carries)))
            
            # Build sample_idx -> incoming_carry map
            next_carry_map = {int(idx): int(carry) for idx, carry in zip(next_indices, next_carries)}
            
            next_carry_values = np.full(n_samples, -1, dtype=np.int64)
            for i, s_idx in enumerate(sample_indices):
                carry_val = next_carry_map.get(int(s_idx))
                if carry_val is not None and carry_val != -1:
                    next_carry_values[i] = carry_val
            result['next_incoming_carry'] = next_carry_values
        except Exception as e:
            print(f"  Warning: failed to load next-position data: {e}")
    
    return result

def get_sampled_indices(labels_np, seed=42, in_carry=None):
    """
    Unified sampling so all layers use the same samples.
    
    Args:
        labels_np: label array
        seed: random seed
        in_carry: optional incoming-carry array; if BALANCE_IN_CARRY_SAMPLING=True,
                  balance across in_carry values (ignores pos/neg split).
    
    Returns:
        selected_indices: sampled indices
    """
    np.random.seed(seed)
    
    # If in_carry balanced sampling enabled
    if in_carry is not None and BALANCE_IN_CARRY_SAMPLING and not COMPUTE_ALL_COMBINATIONS:
        # Get unique in_carry values
        unique_carries = np.unique(in_carry)
        num_carries = len(unique_carries)
        print(f"  BALANCE_IN_CARRY_SAMPLING enabled: {num_carries} distinct in_carry values: {sorted(unique_carries.tolist())}")
        
        # Count raw samples per carry
        carry_counts = {}
        for carry_val in unique_carries:
            carry_counts[carry_val] = np.sum(in_carry == carry_val)
        
        # Target count: min of per-carry quota and smallest bucket (forced balance)
        total_max_points = 2 * UMAP_MAX_POINTS
        max_per_carry = total_max_points // num_carries
        min_available = min(carry_counts.values())
        samples_per_carry = min(max_per_carry, min_available)
        
        print(f"  Forced balance: {samples_per_carry} samples per carry (quota cap={max_per_carry}, min available={min_available})")
        
        selected_all = []
        carry_stats = []  # per-carry stats
        
        for carry_val in unique_carries:
            # All indices for this carry (pos/neg combined)
            carry_indices = np.where(in_carry == carry_val)[0]
            total_for_carry = len(carry_indices)
            pos_for_carry = np.sum(labels_np[carry_indices] == 1)
            neg_for_carry = np.sum(labels_np[carry_indices] == 0)
            
            # Force sample samples_per_carry
            if len(carry_indices) > samples_per_carry:
                carry_indices = np.random.choice(carry_indices, size=samples_per_carry, replace=False)
            
            # Pos/neg counts after sampling
            sampled_pos = np.sum(labels_np[carry_indices] == 1)
            sampled_neg = np.sum(labels_np[carry_indices] == 0)
            carry_stats.append({
                'carry': carry_val,
                'total': total_for_carry,
                'pos': pos_for_carry,
                'neg': neg_for_carry,
                'sampled': len(carry_indices),
                'sampled_pos': sampled_pos,
                'sampled_neg': sampled_neg
            })
            selected_all.append(carry_indices)
        
        selected_indices = np.concatenate(selected_all) if selected_all else np.array([], dtype=np.int64)
        
        # Print detailed stats
        print(f"  Per-carry sampling details:")
        for stat in carry_stats:
            print(f"    carry={stat['carry']}: raw {stat['total']} (pos{stat['pos']}/neg{stat['neg']}) -> sampled {stat['sampled']} (pos{stat['sampled_pos']}/neg{stat['sampled_neg']})")
        
        # Total pos/neg after sampling
        pos_count = np.sum(labels_np[selected_indices] == 1)
        neg_count = np.sum(labels_np[selected_indices] == 0)
        print(f"  Balanced sampling: {len(selected_indices)} samples (pos {pos_count}, neg {neg_count})")
    else:
        # Original logic: sample pos and neg separately
        pos_indices = np.where(labels_np == 1)[0]
        neg_indices = np.where(labels_np == 0)[0]
        
        # Max sample count per class
        max_pos = UMAP_MAX_POS_POINTS if UMAP_MAX_POS_POINTS is not None else UMAP_MAX_POINTS
        max_neg = UMAP_MAX_NEG_POINTS if UMAP_MAX_NEG_POINTS is not None else UMAP_MAX_POINTS
        
        if len(pos_indices) > max_pos:
            pos_indices = np.random.choice(pos_indices, size=max_pos, replace=False)
        if len(neg_indices) > max_neg:
            neg_indices = np.random.choice(neg_indices, size=max_neg, replace=False)
        
        selected_indices = np.concatenate([pos_indices, neg_indices])

    np.random.shuffle(selected_indices)
    
    return selected_indices


def decode_umap_grid(umap_model, model_obj, tokenizer, hiddens_mean, 
                     centered_hiddens=None, grid_size=20, n_workers=4):
    """
    Grid-sample UMAP embedding, inverse-transform and decode to tokens.
    Supports 2D and 3D UMAP.
    
    Args:
        umap_model: fitted UMAP model
        model_obj: language model (for lm_head)
        tokenizer: tokenizer
        hiddens_mean: hidden-state mean (undo centering)
        centered_hiddens: centered hiddens (for KNN training)
        grid_size: grid resolution
        n_workers: parallel threads
        
    Returns:
        list of tuples: 2D (x,y,token), 3D (x,y,z,token)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # UMAP embedding bounds and dimension
    embedding = umap_model.embedding_
    n_dim = embedding.shape[1]
    is_3d = (n_dim >= 3)
    
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    
    # Slightly expand bounds for edges
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    xs = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
    ys = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
    
    # Build grid points
    if is_3d:
        z_min, z_max = embedding[:, 2].min(), embedding[:, 2].max()
        z_margin = (z_max - z_min) * 0.05
        # Smaller 3D resolution to limit point count
        grid_size_3d = min(grid_size, 20)  # cap 3D grid at 20x20x20 = 8000 points
        xs = np.linspace(x_min - x_margin, x_max + x_margin, grid_size_3d)
        ys = np.linspace(y_min - y_margin, y_max + y_margin, grid_size_3d)
        zs = np.linspace(z_min - z_margin, z_max + z_margin, grid_size_3d)
        
        grid_points = []
        for x in xs:
            for y in ys:
                for z in zs:
                    grid_points.append([x, y, z])
        grid_points = np.array(grid_points)
        print(f"    Grid decode (3D): {len(grid_points)} points ({grid_size_3d}x{grid_size_3d}x{grid_size_3d})")
    else:
        grid_points = []
        for x in xs:
            for y in ys:
                grid_points.append([x, y])
        grid_points = np.array(grid_points)
        print(f"    Grid decode (2D): {len(grid_points)} points ({grid_size}x{grid_size})")
    
    # Choose inverse-transform method
    if UMAP_GRID_USE_KNN and centered_hiddens is not None:
        # KNN regressor for high-dim vectors
        from sklearn.neighbors import KNeighborsRegressor
        print(f"    Using KNN regressor (K={UMAP_GRID_KNN_K}) to predict high-dim vectors...")
        
        # Training data: embedding -> centered_hiddens
        # Note: embedding may include anchors; take samples portion only
        n_samples = len(centered_hiddens)
        sample_embedding = embedding[:n_samples]  # samples embedding only
        
        knn = KNeighborsRegressor(n_neighbors=UMAP_GRID_KNN_K, weights='distance', n_jobs=n_workers)
        knn.fit(sample_embedding, centered_hiddens)
        
        # Predict high-dim vectors for grid points
        centered_vectors = knn.predict(grid_points)
        print(f"    KNN prediction done")
    else:
        # UMAP inverse transform (slow)
        print(f"    Running UMAP inverse transform...")
        try:
            centered_vectors = umap_model.inverse_transform(grid_points)
        except Exception as e:
            print(f"    [Error] UMAP inverse transform failed: {e}")
            return []
    
    # Restore mean and decode
    print(f"    Decoding tokens...")
    device = model_obj.device
    hiddens_mean_device = hiddens_mean.to(device)
    
    def decode_single(idx):
        centered_vec = torch.tensor(centered_vectors[idx], dtype=torch.float32).unsqueeze(0).to(device)
        restored_vec = centered_vec + hiddens_mean_device
        
        with torch.no_grad():
            logits = model_obj.lm_head(restored_vec.to(model_obj.dtype))
            token_id = logits.argmax(dim=-1).item()
            token = tokenizer.decode([token_id])
        
        if is_3d:
            x, y, z = grid_points[idx]
            return (x, y, z, token.strip())
        else:
            x, y = grid_points[idx]
            return (x, y, token.strip())
    
    # Parallel decode
    grid_tokens = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(decode_single, range(len(grid_points))))
    grid_tokens = results
    
    print(f"    Grid decode done: {len(grid_tokens)} tokens")
    return grid_tokens

def plot_embedding(plot_data, labels_plot, digits_plot, gt_plot, 
                   title_suffix, position_label, layer_idx, 
                   correctness_plot=None, pred_plot=None, in_carry_plot=None,
                   c_potential_plot=None,
                   color_labels_plot=None,
                   anchor_embedding=None, anchor_labels=None,
                   grid_tokens=None):
    """
    Plotting logic; picks 2D or 3D from UMAP_DIM.
    
    Args:
        pred_plot: optional preds for carry(pred) markers in in_carry/out_carry.
        in_carry_plot: optional in_carry for raw_sum(in_carry) markers.
        c_potential_plot: optional C_potential for gt_pred_carry_potential.
    """
    plt.rcParams['figure.dpi'] = 300
    is_3d = UMAP_DIM == 3
    dim_suffix = "3D" if is_3d else "2D"
    
    # Create figure/axes by dimension
    if is_3d:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
    
    legend_added = False
    colors = {1: 'blue', 0: 'red'}
    marker_labels_plot = np.asarray(labels_plot)
    color_labels_plot = marker_labels_plot if color_labels_plot is None else np.asarray(color_labels_plot)
    if len(color_labels_plot) != len(marker_labels_plot):
        raise ValueError(
            f"color_labels_plot length ({len(color_labels_plot)}) != labels_plot length ({len(marker_labels_plot)})"
        )
    
    for lbl in [1, 0]:
        mask = marker_labels_plot == lbl
        color_mask = color_labels_plot[mask]
        if UMAP_MARKER_MODE == "point":
            if LOAD_PRETRAINED_MODEL and correctness_plot is not None:
                base_colors = np.where(color_mask == 1, colors[1], colors[0])
                sample_colors = np.where(correctness_plot[mask], base_colors, 'green')
                if is_3d:
                    for i, (x, y, z, c) in enumerate(zip(plot_data[mask, 0], plot_data[mask, 1], plot_data[mask, 2], sample_colors)):
                        ax.scatter(x, y, z, s=8, alpha=0.3, color=c)
                else:
                    for i, (x, y, c) in enumerate(zip(plot_data[mask, 0], plot_data[mask, 1], sample_colors)):
                        ax.scatter(x, y, s=8, alpha=0.3, color=c)
                if not legend_added:
                    blue_label, red_label = get_color_legend_labels(with_prediction_state=True)
                    if is_3d:
                        ax.scatter([], [], [], s=8, alpha=0.3, color=colors[1], label=blue_label)
                        ax.scatter([], [], [], s=8, alpha=0.3, color=colors[0], label=red_label)
                        ax.scatter([], [], [], s=8, alpha=0.3, color='green', label='Wrong prediction')
                    else:
                        ax.scatter([], [], s=8, alpha=0.3, color=colors[1], label=blue_label)
                        ax.scatter([], [], s=8, alpha=0.3, color=colors[0], label=red_label)
                        ax.scatter([], [], s=8, alpha=0.3, color='green', label='Wrong prediction')
                    legend_added = True
            else:
                base_colors = np.where(color_mask == 1, colors[1], colors[0])
                if is_3d:
                    for i, (x, y, z, c) in enumerate(zip(plot_data[mask, 0], plot_data[mask, 1], plot_data[mask, 2], base_colors)):
                        ax.scatter(x, y, z, s=8, alpha=0.3, color=c)
                else:
                    for i, (x, y, c) in enumerate(zip(plot_data[mask, 0], plot_data[mask, 1], base_colors)):
                        ax.scatter(x, y, s=8, alpha=0.3, color=c)
                if not legend_added:
                    blue_label, red_label = get_color_legend_labels(with_prediction_state=False)
                    if is_3d:
                        ax.scatter([], [], [], s=8, alpha=0.3, color=colors[1], label=blue_label)
                        ax.scatter([], [], [], s=8, alpha=0.3, color=colors[0], label=red_label)
                    else:
                        ax.scatter([], [], s=8, alpha=0.3, color=colors[1], label=blue_label)
                        ax.scatter([], [], s=8, alpha=0.3, color=colors[0], label=red_label)
                    legend_added = True
        else:  # UMAP_MARKER_MODE == "pred" or "input"
            if digits_plot is None:
                # Fallback to point mode if digits_plot is None
                base_colors = np.where(color_mask == 1, colors[1], colors[0])
                if is_3d:
                    for i, (x, y, z, c) in enumerate(zip(plot_data[mask, 0], plot_data[mask, 1], plot_data[mask, 2], base_colors)):
                        ax.scatter(x, y, z, s=8, alpha=0.3, color=c)
                else:
                    for i, (x, y, c) in enumerate(zip(plot_data[mask, 0], plot_data[mask, 1], base_colors)):
                        ax.scatter(x, y, s=8, alpha=0.3, color=c)
                if not legend_added:
                    blue_label, red_label = get_color_legend_labels(with_prediction_state=False)
                    if is_3d:
                        ax.scatter([], [], [], s=8, alpha=0.3, color=colors[1], label=blue_label)
                        ax.scatter([], [], [], s=8, alpha=0.3, color=colors[0], label=red_label)
                    else:
                        ax.scatter([], [], s=8, alpha=0.3, color=colors[1], label=blue_label)
                        ax.scatter([], [], s=8, alpha=0.3, color=colors[0], label=red_label)
                    legend_added = True
                continue
            
            if is_3d:
                ax.scatter(
                    plot_data[mask, 0], plot_data[mask, 1], plot_data[mask, 2],
                    s=6, alpha=0, color=colors[lbl]
                )
                xs, ys, zs = plot_data[mask, 0], plot_data[mask, 1], plot_data[mask, 2]
            else:
                ax.scatter(
                    plot_data[mask, 0], plot_data[mask, 1],
                    s=6, alpha=0, color=colors[lbl]
                )
                xs, ys = plot_data[mask, 0], plot_data[mask, 1]
                zs = None
            
            ds, gs = digits_plot[mask], gt_plot[mask]
            
            if LOAD_PRETRAINED_MODEL and correctness_plot is not None:
                correctness_mask = correctness_plot[mask]
                if is_3d:
                    for idx_in_mask, (x, y, z, d, g, is_correct) in enumerate(zip(xs, ys, zs, ds, gs, correctness_mask)):
                        if d >= 0 or d == -2:  # -2 means "pl" (input mode)
                            if lbl == 0 and g >= 0:
                                if UMAP_MARKER_MODE == "input":
                                    txt = "PL" if d == -2 else str(int(d))
                                elif UMAP_MARKER_MODE in ["in_carry", "out_carry"]:
                                    # in_carry/out_carry: show carry(pred)
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else g
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum":
                                    # raw_sum: show raw_sum(in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else 0
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    # gt_pred_carry: show (gt, pred, in_carry)
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    # gt_pred_carry_potential: show (gt, pred, in_carry, c_potential)
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    if UMAP_MARKER_MODE in ["next_raw_sum", "next_pred", "next_incoming_carry"]:
                                         txt = "PL" if d == -2 else str(int(d))
                                    else:
                                         txt = (f"PL({int(g)})" if d == -2 else f"{int(d)}({int(g)})")
                            else:
                                if UMAP_MARKER_MODE in ["in_carry", "out_carry"] and pred_plot is not None:
                                    p = pred_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum" and in_carry_plot is not None:
                                    ic = in_carry_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    txt = "PL" if d == -2 else str(int(d))
                            text_color = 'green' if not is_correct else colors[int(color_mask[idx_in_mask])]
                            ax.text(x, y, z, txt, color=text_color, fontsize=6, alpha=0.4)
                else:
                    # 2D version with index tracking
                    for idx_in_mask, (x, y, d, g, is_correct) in enumerate(zip(xs, ys, ds, gs, correctness_mask)):
                        if d >= 0 or d == -2:  # -2 means "pl" (input mode)
                            if lbl == 0 and g >= 0:
                                if UMAP_MARKER_MODE == "input":
                                    txt = "PL" if d == -2 else str(int(d))
                                elif UMAP_MARKER_MODE in ["in_carry", "out_carry"]:
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else g
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum":
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else 0
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    if UMAP_MARKER_MODE in ["next_raw_sum", "next_pred", "next_incoming_carry"]:
                                         txt = "PL" if d == -2 else str(int(d))
                                    else:
                                         txt = (f"PL({int(g)})" if d == -2 else f"{int(d)}({int(g)})")
                            else:
                                if UMAP_MARKER_MODE in ["in_carry", "out_carry"] and pred_plot is not None:
                                    p = pred_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum" and in_carry_plot is not None:
                                    ic = in_carry_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    txt = "PL" if d == -2 else str(int(d))
                            text_color = 'green' if not is_correct else colors[int(color_mask[idx_in_mask])]
                            ax.text(x, y, txt, color=text_color, fontsize=6, alpha=0.4, ha='center', va='center')
                if not legend_added:
                    blue_label, red_label = get_color_legend_labels(with_prediction_state=True)
                    if is_3d:
                        ax.scatter([], [], [], s=30, alpha=0.5, color=colors[1], label=blue_label)
                        ax.scatter([], [], [], s=30, alpha=0.5, color=colors[0], label=red_label)
                        ax.scatter([], [], [], s=30, alpha=0.5, color='green', label='Wrong prediction')
                    else:
                        ax.scatter([], [], s=30, alpha=0.5, color=colors[1], label=blue_label)
                        ax.scatter([], [], s=30, alpha=0.5, color=colors[0], label=red_label)
                        ax.scatter([], [], s=30, alpha=0.5, color='green', label='Wrong prediction')
                    legend_added = True
            else:
                if is_3d:
                    for idx_in_mask, (x, y, z, d, g) in enumerate(zip(xs, ys, zs, ds, gs)):
                        if d >= 0 or d == -2:  # -2 means "pl" (input mode)
                            if lbl == 0 and g >= 0:
                                if UMAP_MARKER_MODE == "input":
                                    txt = "PL" if d == -2 else str(int(d))
                                elif UMAP_MARKER_MODE in ["in_carry", "out_carry"]:
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else g
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum":
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else 0
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    txt = (f"PL({int(g)})" if d == -2 else f"{int(d)}({int(g)})")
                            else:
                                if UMAP_MARKER_MODE in ["in_carry", "out_carry"] and pred_plot is not None:
                                    p = pred_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum" and in_carry_plot is not None:
                                    ic = in_carry_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    txt = "PL" if d == -2 else str(int(d))
                            ax.text(x, y, z, txt, color=colors[int(color_mask[idx_in_mask])], fontsize=6, alpha=0.4)
                else:
                    for idx_in_mask, (x, y, d, g) in enumerate(zip(xs, ys, ds, gs)):
                        if d >= 0 or d == -2:  # -2 means "pl" (input mode)
                            if lbl == 0 and g >= 0:
                                if UMAP_MARKER_MODE == "input":
                                    txt = "PL" if d == -2 else str(int(d))
                                elif UMAP_MARKER_MODE in ["in_carry", "out_carry"]:
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else g
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum":
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else 0
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    txt = (f"PL({int(g)})" if d == -2 else f"{int(d)}({int(g)})")
                            else:
                                if UMAP_MARKER_MODE in ["in_carry", "out_carry"] and pred_plot is not None:
                                    p = pred_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(p)})"
                                elif UMAP_MARKER_MODE == "raw_sum" and in_carry_plot is not None:
                                    ic = in_carry_plot[mask][idx_in_mask]
                                    txt = f"{int(d)}({int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_carry":
                                    # gt_carry: show (gt, in_carry)
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    txt = f"({int(g)},{int(p)},{int(ic)})"
                                elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
                                    p = pred_plot[mask][idx_in_mask] if pred_plot is not None else -1
                                    ic = in_carry_plot[mask][idx_in_mask] if in_carry_plot is not None else -1
                                    ti = c_potential_plot[mask][idx_in_mask] if c_potential_plot is not None else 0.0
                                    txt = f"({int(g)},{int(p)},{int(ic)},{ti:.2f})"
                                else:
                                    txt = "PL" if d == -2 else str(int(d))
                            ax.text(x, y, txt, color=colors[int(color_mask[idx_in_mask])], fontsize=6, alpha=0.4, ha='center', va='center')
                        else:
                            raise ValueError(f"Invalid input digit: {d}")
                if not legend_added:
                    blue_label, red_label = get_color_legend_labels(with_prediction_state=False)
                    if is_3d:
                        ax.scatter([], [], [], s=30, alpha=0.5, color=colors[1], label=blue_label)
                        ax.scatter([], [], [], s=30, alpha=0.5, color=colors[0], label=red_label)
                    else:
                        ax.scatter([], [], s=30, alpha=0.5, color=colors[1], label=blue_label)
                        ax.scatter([], [], s=30, alpha=0.5, color=colors[0], label=red_label)
                    legend_added = True
    
    # Plot anchors if provided
    if anchor_embedding is not None and anchor_labels is not None:
        if is_3d:
            for i, label in enumerate(anchor_labels):
                 ax.text(anchor_embedding[i, 0], anchor_embedding[i, 1], anchor_embedding[i, 2],
                         str(label), color='black', fontsize=14, fontweight='bold')
        else:
            for i, label in enumerate(anchor_labels):
                 ax.text(anchor_embedding[i, 0], anchor_embedding[i, 1],
                         str(label), color='black', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Plot grid decoded regions with colors
    if grid_tokens is not None:
        # Unique tokens and colors
        all_tokens = [item[-1] for item in grid_tokens if item[-1]]
        unique_tokens = sorted(list(set(all_tokens)))
        # tab20 or other colormap
        cmap = plt.get_cmap('tab20')
        token_to_color = {t: cmap(i % 20) for i, t in enumerate(unique_tokens)}
        
        if is_3d:
            # 3D: scatter
            xs_g, ys_g, zs_g, colors_g = [], [], [], []
            for item in grid_tokens:
                if len(item) == 4 and item[3]:
                    xs_g.append(item[0])
                    ys_g.append(item[1])
                    zs_g.append(item[2])
                    colors_g.append(token_to_color[item[3]])
            
            if xs_g:
                ax.scatter(xs_g, ys_g, zs_g, c=colors_g, alpha=0.1, s=5, marker='.')
        else:
            # 2D: pcolormesh or scatter
            # Use scatter for simplicity (works on irregular grids)
            # pcolormesh needs a regular grid
            # grid_tokens is a point list; scatter is general
            xs_g, ys_g, colors_g = [], [], []
            for item in grid_tokens:
                if len(item) == 3 and item[2]:
                    xs_g.append(item[0])
                    ys_g.append(item[1])
                    colors_g.append(token_to_color[item[2]])
            
            if xs_g:
                ax.scatter(xs_g, ys_g, c=colors_g, alpha=0.1, s=15, marker='s', edgecolors='none', zorder=-1) # behind points

        # Legend for tokens
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='s', color='w', label=token,
                                  markerfacecolor=token_to_color[token], markersize=8, alpha=0.5)
                           for token in unique_tokens]
        # Second legend (tokens)
        if legend_elements:
            # Outside or corner
            leg2 = ax.legend(handles=legend_elements, title="Decoded Tokens", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
            ax.add_artist(leg2) # Re-add main legend later; avoid overwriting


    # Axis labels and title
    if is_3d:
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.set_zlabel('UMAP-3')
        ax.set_title(f'UMAP (3D) - {title_suffix} [marker: {UMAP_MARKER_MODE}, color: {UMAP_COLOR_MODE}]')
    else:
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.set_title(f'UMAP (2D) - {title_suffix} [marker: {UMAP_MARKER_MODE}, color: {UMAP_COLOR_MODE}]')
        ax.grid(True, alpha=0.3)
    ax.legend()
    
    if SAVE_PLOT:
        sub_dir = Path(DATA_PATH).stem
        # Continuous mode: subfolder per alignment mode
        mode_suffix = f"_{UMAP_ALIGNMENT_MODE}" if UMAP_ALIGNMENT_MODE != 'independent' else ""
        base_save_dir = os.path.join(SAVE_DIR, sub_dir + mode_suffix)
        # Per-pos subfolder with 2D/3D, model, and marker suffixes
        model_suffix = "_with_model" if LOAD_PRETRAINED_MODEL else ""
        marker_suffix = f"_{UMAP_MARKER_MODE}"  # marker mode suffix
        color_suffix = get_color_mode_suffix()
        filter_suffix = get_pred_filter_suffix()  # pred filter suffix
        pos_subdir = f"pos{position_label}_{dim_suffix}{model_suffix}{marker_suffix}{color_suffix}{filter_suffix}"
        final_save_dir = os.path.join(base_save_dir, pos_subdir)
        os.makedirs(final_save_dir, exist_ok=True)
        
        filename = f"umap{dim_suffix}_pos{position_label}_layer{layer_idx}_{UMAP_MARKER_MODE}{color_suffix}{filter_suffix}.png"
        save_path = os.path.join(final_save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        print("\n" + "-"*60)
        print(f"Plot saved to: {save_path}")

    if SAVE_PLOT_DATA:
        sub_dir = Path(DATA_PATH).stem
        # Continuous mode: subfolder per alignment mode
        mode_suffix = f"_{UMAP_ALIGNMENT_MODE}" if UMAP_ALIGNMENT_MODE != 'independent' else ""
        base_save_dir = os.path.join(SAVE_DIR, sub_dir + mode_suffix)
        # Per-pos subfolder with 2D/3D, model, and marker suffixes
        model_suffix = "_with_model" if LOAD_PRETRAINED_MODEL else ""
        marker_suffix = f"_{UMAP_MARKER_MODE}"  # marker mode suffix
        color_suffix = get_color_mode_suffix()
        filter_suffix = get_pred_filter_suffix()  # pred filter suffix
        pos_subdir = f"pos{position_label}_{dim_suffix}{model_suffix}{marker_suffix}{color_suffix}{filter_suffix}"
        final_save_dir = os.path.join(base_save_dir, pos_subdir)
        os.makedirs(final_save_dir, exist_ok=True)
        
        filename = f"umap{dim_suffix}_pos{position_label}_layer{layer_idx}_{UMAP_MARKER_MODE}{color_suffix}{filter_suffix}.pkl"
        save_path = os.path.join(final_save_dir, filename)
        
        # Data to save
        save_data = {
            'embedding': plot_data,
            'labels': labels_plot,
            'color_labels': color_labels_plot,
            'digits': digits_plot,
            'gt': gt_plot,
            'title_suffix': title_suffix,
            'position_label': position_label,
            'layer_idx': layer_idx,
            'correctness': correctness_plot,
            'anchor_embedding': anchor_embedding,
            'anchor_labels': anchor_labels,
            'c_potential_plot': c_potential_plot,
            'in_carry': in_carry_plot,
            'grid_tokens': grid_tokens,
            # Add other relevant meta info
            'UMAP_MARKER_MODE': UMAP_MARKER_MODE,
            'UMAP_COLOR_MODE': UMAP_COLOR_MODE,
        }
        
        # NOTE: plot_embedding signature does not include samples_plot, so I cannot save it unless I pass it or it is global (likely not global).
        # However, labels, digits, gt are passed.
        # Let's check what plot_embedding argument list has.
        # def plot_embedding(embedding, labels, digits, gt, title_suffix, position_label, layer_idx, 
        #           correctness=None, anchor_embedding=None, anchor_labels=None, 
        #           c_potential_plot=None, grid_tokens=None, output_path=None):
        
        try:
             with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
             print(f"Data saved to: {save_path}")
        except Exception as e:
             print(f"Failed to save data: {e}")

    plt.close()

def save_aligned_umap_data(embeddings_list, marker_labels_sampled, color_labels_sampled, digits_sampled, gt_sampled,
                            aligned_layers, position_label, correctness_sampled, 
                            layers_to_plot, selected_indices, data_path):
    """
    Save Aligned UMAP plot data to .pkl.
    
    Args:
        embeddings_list: embeddings per layer
        marker_labels_sampled: labels for marker logic
        color_labels_sampled: labels for color
        digits_sampled: sampled digits
        gt_sampled: sampled ground truth
        aligned_layers: layer indices
        position_label: position label
        correctness_sampled: optional correctness array
        layers_to_plot: layers to plot
        selected_indices: sampled indices
        data_path: data path (for save filename)
    """
    # PKL in base_save_dir; UMAP compute is independent of marker mode
    # Different marker modes can share one PKL
    is_3d = UMAP_DIM == 3
    sub_dir = Path(data_path).stem
    mode_suffix = f"_{UMAP_ALIGNMENT_MODE}" if UMAP_ALIGNMENT_MODE != 'independent' else ""
    base_save_dir = os.path.join(SAVE_DIR, sub_dir + mode_suffix)
    model_suffix = "_with_model" if LOAD_PRETRAINED_MODEL else ""
    dim_suffix = "3D" if is_3d else "2D"
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Save filename in base_save_dir
    color_suffix = get_color_mode_suffix()
    filename = f"aligned_umap{dim_suffix}_data_pos{position_label}{model_suffix}{color_suffix}.pkl"
    save_path = os.path.join(base_save_dir, filename)
    
    # Data to save; ensure all values are serializable
    # Embeddings as plain numpy (avoid Numba serialization)
    embeddings_serializable = []
    for emb in embeddings_list:
        if isinstance(emb, np.ndarray):
            embeddings_serializable.append(emb.copy())
        else:
            # Convert other types (e.g. torch tensor) to numpy
            embeddings_serializable.append(np.asarray(emb))
    
    # Ensure other arrays are numpy
    marker_labels_serializable = np.asarray(marker_labels_sampled) if marker_labels_sampled is not None else None
    color_labels_serializable = np.asarray(color_labels_sampled) if color_labels_sampled is not None else None
    digits_serializable = np.asarray(digits_sampled) if digits_sampled is not None else None
    gt_serializable = np.asarray(gt_sampled) if gt_sampled is not None else None
    correctness_serializable = np.asarray(correctness_sampled) if correctness_sampled is not None else None
    selected_indices_serializable = np.asarray(selected_indices) if selected_indices is not None else None
    
    save_data = {
        'embeddings': embeddings_serializable,
        'labels_sampled': marker_labels_serializable,  # backward compatibility
        'marker_labels_sampled': marker_labels_serializable,
        'color_labels_sampled': color_labels_serializable,
        'digits_sampled': digits_serializable,
        'gt_sampled': gt_serializable,
        'aligned_layers': list(aligned_layers),  # ensure list
        'position_label': position_label,
        'correctness_sampled': correctness_serializable,
        'layers_to_plot': list(layers_to_plot) if layers_to_plot is not None else None,  # ensure list
        'selected_indices': selected_indices_serializable,
        # Saved config params
        'UMAP_DIM': UMAP_DIM,
        'UMAP_MARKER_MODE': UMAP_MARKER_MODE,
        'UMAP_COLOR_MODE': UMAP_COLOR_MODE,
        'UMAP_ALIGNMENT_MODE': UMAP_ALIGNMENT_MODE,
        'LOAD_PRETRAINED_MODEL': LOAD_PRETRAINED_MODEL,
        'SAVE_PLOT': SAVE_PLOT,
        'SAVE_DIR': SAVE_DIR,
        'EXPORT_GIF': EXPORT_GIF,
        'DATA_PATH': data_path,
    }
    
    # Atomic write via temp file
    temp_save_path = save_path + '.tmp'
    try:
        with open(temp_save_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())  # flush to disk
        
        # Verify file integrity
        if os.path.getsize(temp_save_path) == 0:
            raise IOError("Saved file is empty")
        
        # Verify loadable
        with open(temp_save_path, 'rb') as f:
            pickle.load(f)  # verify loadable
        
        # Rename on success
        if os.path.exists(save_path):
            os.remove(save_path)
        os.rename(temp_save_path, save_path)
        
        print(f"  Saved Aligned UMAP data to: {save_path} (size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        # Clean temp file on error
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)
        print(f"  Error saving data: {e}")
        raise

def load_aligned_umap_data(position_label, data_path):
    """
    Load Aligned UMAP plot data from .pkl.
    
    Args:
        position_label: position label
        data_path: data path (for load filename)
        
    Returns:
        saved dict, or None if missing
    """
    # Load PKL from base_save_dir (same as save)
    is_3d = UMAP_DIM == 3
    sub_dir = Path(data_path).stem
    mode_suffix = f"_{UMAP_ALIGNMENT_MODE}" if UMAP_ALIGNMENT_MODE != 'independent' else ""
    base_save_dir = os.path.join(SAVE_DIR, sub_dir + mode_suffix)
    model_suffix = "_with_model" if LOAD_PRETRAINED_MODEL else ""
    dim_suffix = "3D" if is_3d else "2D"
    
    # Load from base_save_dir
    color_suffix = get_color_mode_suffix()
    filename = f"aligned_umap{dim_suffix}_data_pos{position_label}{model_suffix}{color_suffix}.pkl"
    load_path = os.path.join(base_save_dir, filename)
    
    if not os.path.exists(load_path):
        print(f"  Warning: saved data file not found: {load_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(load_path)
    if file_size == 0:
        print(f"  Warning: data file is empty: {load_path}")
        return None
    
    print(f"  Loading saved data: {load_path} (size: {file_size / 1024 / 1024:.2f} MB)")
    
    try:
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verify required keys
        required_keys = ['embeddings', 'labels_sampled', 'aligned_layers', 'position_label']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Loaded data missing required key: {key}")
        
        print(f"  Data loaded ({len(data['embeddings'])} layer embeddings)")
        return data
    except EOFError as e:
        print(f"  Error: incomplete/corrupt data file (EOFError): {load_path}")
        print(f"  Suggestion: delete file and re-run to regenerate")
        return None
    except Exception as e:
        print(f"  Error loading data: {e}")
        print(f"  File path: {load_path}")
        return None

def create_aligned_umap_gif(embeddings_list, marker_labels_sampled, color_labels_sampled, digits_sampled, gt_sampled,
                            aligned_layers, position_label, correctness_sampled=None, layers_to_plot=None):
    """
    Create Aligned UMAP GIF showing embedding evolution per layer.
    In aligned mode, also saves each frame as PNG.
    
    Args:
        embeddings_list: per-layer (N, UMAP_DIM) arrays
        marker_labels_sampled: marker labels
        color_labels_sampled: color labels
        digits_sampled: optional sampled digits
        gt_sampled: optional sampled GT
        aligned_layers: layer indices
        position_label: position label
        correctness_sampled: optional correctness
        layers_to_plot: optional layers for PNG; None saves all
    """
    # In aligned mode, save each frame as PNG even if EXPORT_GIF=False
    # Do not return early
    
    print(f"  Starting GIF generation...")
    
    is_3d = UMAP_DIM == 3
    num_frames = len(embeddings_list)
    
    if num_frames == 0:
        print(f"  Warning: no embeddings; skipping GIF")
        return
    
    # Coordinate range across frames
    all_coords = np.concatenate(embeddings_list, axis=0)
    if is_3d:
        # 3D: use first two dims for simpler 2D animation
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        # Add margin
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax_bound = [x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin]
    else:
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax_bound = [x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin]
    
    # Per-frame coords (3D: first two dims only)
    if is_3d:
        offsets_list = [emb[:, :2] for emb in embeddings_list]  # first two dims only
    else:
        offsets_list = embeddings_list
    
    # Create figure
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    
    # Init scatter
    colors = {1: 'blue', 0: 'red'}
    marker_labels_sampled = np.asarray(marker_labels_sampled)
    color_labels_sampled = marker_labels_sampled if color_labels_sampled is None else np.asarray(color_labels_sampled)
    if len(marker_labels_sampled) != len(color_labels_sampled):
        raise ValueError(
            f"marker_labels_sampled length ({len(marker_labels_sampled)}) != color_labels_sampled length ({len(color_labels_sampled)})"
        )
    
    if UMAP_MARKER_MODE == "point":
        # Point mode: scatter
        # Colors from labels
        color_array = np.array([colors[int(lbl)] for lbl in color_labels_sampled])
        if LOAD_PRETRAINED_MODEL and correctness_sampled is not None:
            # Wrong predictions in green
            color_array = np.where(correctness_sampled, color_array, 'green')
        # Init scatter with frame 0 colors (avoid set_array colormap)
        initial_offsets = offsets_list[0] if len(offsets_list) > 0 else np.array([]).reshape(0, 2)
        scat = ax.scatter(initial_offsets[:, 0], initial_offsets[:, 1], 
                         s=8, alpha=0.3, c=color_array)
    else:
        # Digit markers: transparent scatter placeholder
        scat = ax.scatter([], [], s=6, alpha=0)
    
    # Text for current layer
    text = ax.text(ax_bound[0] + (ax_bound[1] - ax_bound[0]) * 0.02, 
                   ax_bound[3] - (ax_bound[3] - ax_bound[2]) * 0.05, 
                   '', fontsize=12, fontweight='bold')
    
    ax.set_xlim(ax_bound[0], ax_bound[1])
    ax.set_ylim(ax_bound[2], ax_bound[3])
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title(f'UMAP Aligned Animation - Pos {position_label} [marker: {UMAP_MARKER_MODE}, color: {UMAP_COLOR_MODE}]')
    ax.set(xticks=[], yticks=[])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Digit mode: text objects updated each frame
    text_objects = []
    if UMAP_MARKER_MODE != "point" and digits_sampled is not None:
        # One text object per sample
        for i in range(len(marker_labels_sampled)):
            txt_obj = ax.text(0, 0, '', fontsize=5, alpha=0.4, ha='center', va='center', visible=False)
            text_objects.append(txt_obj)
        
        # Set text from digits and gt
        for i, (lbl, d, g) in enumerate(zip(marker_labels_sampled, digits_sampled, gt_sampled)):
            if d >= 0 or d == -2:
                if lbl == 0 and g >= 0:
                    if UMAP_MARKER_MODE == "input":
                        txt = "PL" if d == -2 else str(int(d))
                    else:
                        txt = (f"PL({int(g)})" if d == -2 else f"{int(d)}({int(g)})")
                else:
                    txt = "PL" if d == -2 else str(int(d))
                text_objects[i].set_text(txt)
                # Set color
                if LOAD_PRETRAINED_MODEL and correctness_sampled is not None:
                    text_color = 'green' if not correctness_sampled[i] else colors[int(color_labels_sampled[i])]
                else:
                    text_color = colors[int(color_labels_sampled[i])]
                text_objects[i].set_color(text_color)
    
    # Prepare save path
    if SAVE_PLOT:
        sub_dir = Path(DATA_PATH).stem
        mode_suffix = f"_{UMAP_ALIGNMENT_MODE}" if UMAP_ALIGNMENT_MODE != 'independent' else ""
        base_save_dir = os.path.join(SAVE_DIR, sub_dir + mode_suffix)
        model_suffix = "_with_model" if LOAD_PRETRAINED_MODEL else ""
        dim_suffix = "3D" if is_3d else "2D"
        marker_suffix = f"_{UMAP_MARKER_MODE}"  # marker mode suffix
        color_suffix = get_color_mode_suffix()
        pos_subdir = f"pos{position_label}_{dim_suffix}{model_suffix}{marker_suffix}{color_suffix}"
        final_save_dir = os.path.join(base_save_dir, pos_subdir)
        os.makedirs(final_save_dir, exist_ok=True)
    else:
        final_save_dir = None
    
    # Save each frame as PNG
    if SAVE_PLOT:
        print(f"  Saving each frame as PNG...")
        for frame_idx in range(num_frames):
            layer_idx = aligned_layers[frame_idx]
            # If layers_to_plot set, save only those layers
            if layers_to_plot is not None and layer_idx not in layers_to_plot:
                continue
            offsets = offsets_list[frame_idx]
            
            # Update scatter positions
            scat.set_offsets(offsets)
            
            # Update layer text
            text.set_text(f'Layer {layer_idx}')
            
            # Update digit text positions
            if UMAP_MARKER_MODE != "point" and digits_sampled is not None:
                for i, txt_obj in enumerate(text_objects):
                    if digits_sampled[i] >= 0 or digits_sampled[i] == -2:
                        txt_obj.set_position((offsets[i, 0], offsets[i, 1]))
                        txt_obj.set_visible(True)
                    else:
                        txt_obj.set_visible(False)
            
            # Force draw
            fig.canvas.draw()
            
            # Save frame PNG
            filter_suffix = get_pred_filter_suffix()
            color_suffix = get_color_mode_suffix()
            png_filename = f"umap{dim_suffix}_pos{position_label}_layer{layer_idx}_{UMAP_MARKER_MODE}{color_suffix}{filter_suffix}.png"
            png_path = os.path.join(final_save_dir, png_filename)
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {png_filename}")
    
    # Animate (GIF only)
    def animate(frame_idx):
        # Update scatter positions
        offsets = offsets_list[frame_idx]
        scat.set_offsets(offsets)
        
        # Update layer text
        layer_idx = aligned_layers[frame_idx]
        text.set_text(f'Layer {layer_idx}')
        
        # Update digit text positions
        if UMAP_MARKER_MODE != "point" and digits_sampled is not None:
            for i, txt_obj in enumerate(text_objects):
                if digits_sampled[i] >= 0 or digits_sampled[i] == -2:
                    txt_obj.set_position((offsets[i, 0], offsets[i, 1]))
                    txt_obj.set_visible(True)
                else:
                    txt_obj.set_visible(False)
            # Return artists to update
            return [scat, text] + text_objects
        else:
            return scat, text
    
    # Create/save GIF if enabled
    if EXPORT_GIF and SAVE_PLOT:
        print(f"  Starting GIF generation...")
        anim = animation.FuncAnimation(
            fig,
            init_func=None,
            func=animate,
            frames=num_frames,
            interval=GIF_INTERVAL,
            blit=False  # digit mode needs blit=False
        )
        
        filter_suffix = get_pred_filter_suffix()
        color_suffix = get_color_mode_suffix()
        gif_filename = f"umap{dim_suffix}_pos{position_label}_aligned_animation_{UMAP_MARKER_MODE}{color_suffix}{filter_suffix}.gif"
        gif_path = os.path.join(final_save_dir, gif_filename)
        
        try:
            anim.save(gif_path, writer="pillow", fps=GIF_FPS)
            print(f"  GIF saved to: {gif_path}")
        except Exception as e:
            print(f"  Failed to save GIF: {e}")
            print(f"  Hint: install pillow: pip install pillow")
    
    plt.close(fig)

def run_umap_and_plot_v2(X_all_raw, y_all, selected_indices, 
                         pred_digits, gt_digits, position_label, 
                         seq_len, feature_dim,
                         correctness=None,
                         color_labels=None,
                         ignore_layer_selection=False,
                         input_digits=None,
                         data_path=None,
                         sample_indices=None,
                         incoming_carry=None,
                         outgoing_carry=None,
                         raw_sum=None,
                         next_raw_sum=None,
                         next_pred=None,
                         next_incoming_carry=None,
                         c_potential=None,
                         anchor_data=None,
                         anchor_labels=None,
                         model_obj=None,
                         tokenizer=None,
                         hiddens_mean=None):
    X_np = X_all_raw.numpy() if isinstance(X_all_raw, torch.Tensor) else np.asarray(X_all_raw)
    labels_np = y_all.cpu().numpy() if isinstance(y_all, torch.Tensor) else np.asarray(y_all)
    color_labels_np = labels_np if color_labels is None else np.asarray(color_labels)
    if len(color_labels_np) != len(labels_np):
        raise ValueError(
            f"color_labels length ({len(color_labels_np)}) != y_all length ({len(labels_np)})"
        )
    
    X_sampled = X_np[selected_indices]
    labels_sampled = labels_np[selected_indices]
    color_labels_sampled = color_labels_np[selected_indices]
    
    # Pick digit sequence for UMAP_MARKER_MODE
    if UMAP_MARKER_MODE == "pred":
        digits_sampled = pred_digits[selected_indices]
    elif UMAP_MARKER_MODE == "input":
        if input_digits is None:
            raise ValueError("UMAP_MARKER_MODE='input' requires input_digits")
        digits_sampled = input_digits[selected_indices]
    elif UMAP_MARKER_MODE == "in_carry":
        if incoming_carry is None:
            raise ValueError("UMAP_MARKER_MODE='in_carry' requires incoming_carry")
        digits_sampled = incoming_carry[selected_indices]
    elif UMAP_MARKER_MODE == "out_carry":
        if outgoing_carry is None:
            raise ValueError("UMAP_MARKER_MODE='out_carry' requires outgoing_carry")
        digits_sampled = outgoing_carry[selected_indices]
    elif UMAP_MARKER_MODE == "raw_sum":
        if raw_sum is None:
            raise ValueError("UMAP_MARKER_MODE='raw_sum' requires raw_sum")
        digits_sampled = raw_sum[selected_indices]
    elif UMAP_MARKER_MODE == "next_raw_sum":
        if next_raw_sum is None:
            raise ValueError("UMAP_MARKER_MODE='next_raw_sum' requires next_raw_sum")
        digits_sampled = next_raw_sum[selected_indices]
    elif UMAP_MARKER_MODE == "next_pred":
        if next_pred is None:
            raise ValueError("UMAP_MARKER_MODE='next_pred' requires next_pred")
        digits_sampled = next_pred[selected_indices]
    elif UMAP_MARKER_MODE == "next_incoming_carry":
        if next_incoming_carry is None:
            raise ValueError("UMAP_MARKER_MODE='next_incoming_carry' requires next_incoming_carry")
        digits_sampled = next_incoming_carry[selected_indices]
    elif UMAP_MARKER_MODE == "gt_carry":
        # gt_carry: use gt_digits with in_carry in label
        digits_sampled = gt_digits[selected_indices]
    elif UMAP_MARKER_MODE == "gt_pred_carry":
        # digits_sampled mainly checks validity (>=0)
        # Label combines gt, pred, in_carry
        digits_sampled = pred_digits[selected_indices]
    elif UMAP_MARKER_MODE == "gt_pred_carry_potential":
        # Like gt_pred_carry plus c_potential
        if c_potential is None:
            raise ValueError("UMAP_MARKER_MODE='gt_pred_carry_potential' requires c_potential")
        digits_sampled = pred_digits[selected_indices]
    else:  # "point"
        digits_sampled = None
    
    gt_sampled = gt_digits[selected_indices]
    
    # pred_sampled for carry(pred) in in_carry/out_carry
    pred_sampled = pred_digits[selected_indices]
    
    # in_carry_sampled for raw_sum(in_carry)
    in_carry_sampled = incoming_carry[selected_indices] if incoming_carry is not None else None
    
    # c_potential_sampled for gt_pred_carry_potential
    c_potential_sampled = c_potential[selected_indices] if c_potential is not None else None
    
    # Sample sample_indices for interactive HTML
    sample_indices_sampled = None
    if sample_indices is not None:
        sample_indices_sampled = np.asarray(sample_indices)[selected_indices]
    
    # Reshape to (N, L, D)
    X_layers = X_sampled.reshape(len(X_sampled), seq_len, feature_dim)
    num_layers = seq_len

    umap_random_state = 42 if USE_SEED else None

    # Layers to process/plot
    if UMAP_SELECT_SPECIFIC_LAYER and not ignore_layer_selection:
        layers_to_plot = [UMAP_LAYER_INDEX]
    else:
        layers_to_plot = list(range(num_layers))
    
    # SKIP_LAYER_0 when computing all layers: skip layer 0
    if SKIP_LAYER_0 and 0 in layers_to_plot and (not UMAP_SELECT_SPECIFIC_LAYER or ignore_layer_selection):
        layers_to_plot.remove(0)
        # Print once to avoid repeat per position
        if not hasattr(run_umap_and_plot_v2, '_skip_layer0_printed'):
            print(f"  Skipping Layer 0 (SKIP_LAYER_0 = True)")
            run_umap_and_plot_v2._skip_layer0_printed = True
    
    if UMAP_ALIGNMENT_MODE == 'aligned':
        
        # Load saved data and plot if enabled
        if LOAD_SAVED_DATA and data_path is not None:
            print(f"Loading saved Aligned UMAP data from file...")
            saved_data = load_aligned_umap_data(position_label, data_path)
            if saved_data is not None:
                # Plot from loaded data
                create_aligned_umap_gif(
                    saved_data['embeddings'], 
                    saved_data.get('marker_labels_sampled', saved_data['labels_sampled']),
                    saved_data.get('color_labels_sampled', saved_data.get('marker_labels_sampled', saved_data['labels_sampled'])),
                    saved_data['digits_sampled'], 
                    saved_data['gt_sampled'],
                    saved_data['aligned_layers'], 
                    saved_data['position_label'], 
                    saved_data['correctness_sampled'], 
                    layers_to_plot=saved_data['layers_to_plot']
                )
                return
            else:
                print(f"  Warning: could not load saved data; recomputing UMAP...")
        
        print("\n" + "="*60)
        print(f"Running Aligned UMAP on {num_layers} layers...")
        
        # SKIP_LAYER_0 in all-layers mode: skip layer 0
        if SKIP_LAYER_0:
            aligned_layers = layers_to_plot  # layers_to_plot already excludes 0
            print(f"  Skipping Layer 0; layers: {aligned_layers}")
        else:
            # Default: all layers
            aligned_layers = list(range(num_layers))
        
        data_list = [X_layers[:, i, :] for i in aligned_layers]
        # One-to-one sample alignment
        constant_relations = {j: j for j in range(len(X_sampled))}
        # relations count = num_layers - 1
        relations = [constant_relations for _ in range(len(aligned_layers) - 1)]
        
        # Single thread if random_state set (reproducibility)
        if umap_random_state is not None:
            original_n_threads = numba.get_num_threads()
            numba.set_num_threads(1)
            print(f"  Single-threaded (random_state set for reproducibility)")
        else:
            effective_n_jobs = UMAP_N_JOBS if UMAP_N_JOBS > 0 else numba.get_num_threads()
            print(f"  Numba threads: {numba.get_num_threads()}, OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        
        try:
            aligned_model = AlignedUMAP(
                n_components=UMAP_DIM, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                random_state=umap_random_state, metric=UMAP_METRIC
            )
            embeddings = aligned_model.fit_transform(data_list, relations=relations)
        finally:
            # Restore thread count when random_state was set
            if umap_random_state is not None:
                numba.set_num_threads(original_n_threads)
        
        # aligned mode: frames via create_aligned_umap_gif, not plot_embedding
        correctness_sampled = correctness[selected_indices] if correctness is not None else None
        
        # Save .pkl before plotting
        if data_path is not None:
            save_aligned_umap_data(
                embeddings, labels_sampled, color_labels_sampled, digits_sampled, gt_sampled,
                aligned_layers, position_label, correctness_sampled,
                layers_to_plot, selected_indices, data_path
            )
        
        create_aligned_umap_gif(
            embeddings, labels_sampled, color_labels_sampled, digits_sampled, gt_sampled,
            aligned_layers, position_label, correctness_sampled, layers_to_plot=layers_to_plot
        )
            
    elif UMAP_ALIGNMENT_MODE == 'sequential':
        print("\n" + "="*60)
        print(f"Sequential init mode for {num_layers} layers...")
        previous_embedding = 'spectral'
        
        # Sequential: compute through max needed layer
        max_layer = max(layers_to_plot) if layers_to_plot else 0
        effective_n_jobs = UMAP_N_JOBS if umap_random_state is None else 1
        print(f"  Threads: {effective_n_jobs}")
        
        # Start layer (SKIP_LAYER_0 all-layers mode: start at 1)
        start_layer = 1 if SKIP_LAYER_0 else 0
        
        for i in range(start_layer, max_layer + 1):
            print(f"  Processing Layer {i}...")
            umap_model = umap.UMAP(
                n_components=UMAP_DIM, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                random_state=umap_random_state, metric=UMAP_METRIC,
                init=previous_embedding,
                n_jobs=effective_n_jobs
            )
            current_embedding = umap_model.fit_transform(X_layers[:, i, :])
            
            if i in layers_to_plot:
                title = f"Layer {i} (Pos {position_label}, Sequential)"
                correctness_sampled = correctness[selected_indices] if correctness is not None else None
                plot_embedding(
                    current_embedding, labels_sampled, digits_sampled, gt_sampled, title, position_label, i,
                    correctness_sampled, pred_plot=pred_sampled, in_carry_plot=in_carry_sampled,
                    c_potential_plot=c_potential_sampled, color_labels_plot=color_labels_sampled
                )
            
            # Update init embedding
            previous_embedding = current_embedding
            
    else: # independent
        print("\n" + "="*60)
        print(f"Independent mode for {len(layers_to_plot)} layers...")
        # Threads: 1 if random_state else UMAP_N_JOBS
        effective_n_jobs = 1 if umap_random_state is not None else UMAP_N_JOBS
        print(f"  Threads: {effective_n_jobs}")
        
        for i in layers_to_plot:
            if i >= num_layers:
                print(f"  Skipping Layer {i} (out of range, max={num_layers-1})")
                continue
            print(f"  Processing Layer {i}...")
        
            current_layer_data = X_layers[:, i, :]
            
            # Concat anchors for joint UMAP
            if anchor_data is not None:
                # anchor_data: (num_anchors, feature_dim)
                # current_layer_data: (num_samples, feature_dim)
                
                tensor_hiddens = torch.tensor(current_layer_data, dtype=torch.float32)
                tensor_anchors = torch.tensor(anchor_data, dtype=torch.float32)
                
                # --- Independent mean centering ---
                # Center hiddens and anchors separately (remove shared offset)
                # Best empirically: anchors and hiddens mix naturally in UMAP
                hiddens_mean = torch.mean(tensor_hiddens, dim=0, keepdim=True)
                centered_hiddens = tensor_hiddens - hiddens_mean
                
                anchors_mean = torch.mean(tensor_anchors, dim=0, keepdim=True)
                centered_anchors = tensor_anchors - anchors_mean
                
                # When UMAP_METRIC=cosine, UMAP L2-normalizes internally
                # No extra F.normalize needed here
                centered_hiddens = F.normalize(centered_hiddens, dim=1)
                centered_anchors = F.normalize(centered_anchors, dim=1)
                
                combined_data = np.vstack([centered_hiddens.numpy(), centered_anchors.numpy()])
                print(f"    Merged anchors (independent centering): {len(current_layer_data)} samples + {len(anchor_data)} anchors")
            else:
                combined_data = current_layer_data
                # hiddens_mean for grid decode without anchors
                # hiddens_mean for grid decode without anchors
                # When UMAP_DIM=2/3 and grid decode enabled
                enable_grid_decode = (UMAP_DIM == 2 and UMAP_GRID_DECODE_2D) or (UMAP_DIM == 3 and UMAP_GRID_DECODE_3D)
                if enable_grid_decode:
                    tensor_hiddens = torch.tensor(current_layer_data, dtype=torch.float32)
                    hiddens_mean = torch.mean(tensor_hiddens, dim=0, keepdim=True)
                else:
                    hiddens_mean = None
            
            umap_model = umap.UMAP(
                n_components=UMAP_DIM, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                random_state=umap_random_state, metric=UMAP_METRIC,
                n_jobs=effective_n_jobs
            )
            combined_embedding = umap_model.fit_transform(combined_data)
            
            # Trustworthiness: neighborhood preservation
            tw_score = trustworthiness(combined_data, combined_embedding, n_neighbors=min(TRUST_N_NEIGHBORS, len(combined_data) - 1))
            print(f"    Trustworthiness (n_neighbors={min(TRUST_N_NEIGHBORS, len(combined_data) - 1)}): {tw_score:.4f}")
            
            # Split samples and anchors
            if anchor_data is not None:
                n_samples = len(current_layer_data)
                current_embedding = combined_embedding[:n_samples]
                anchor_embedding = combined_embedding[n_samples:]
            else:
                current_embedding = combined_embedding
                anchor_embedding = None
            
            title = f"Layer {i} (Pos {position_label})"
            correctness_sampled = correctness[selected_indices] if correctness is not None else None
            
            # Grid decode
            grid_tokens = None
            enable_grid_decode = (UMAP_DIM == 2 and UMAP_GRID_DECODE_2D) or (UMAP_DIM == 3 and UMAP_GRID_DECODE_3D)
            if enable_grid_decode and model_obj is not None and tokenizer is not None and hiddens_mean is not None:
                # centered_hiddens for KNN (sampled points only)
                knn_centered_hiddens = centered_hiddens.numpy() if isinstance(centered_hiddens, torch.Tensor) else centered_hiddens
                grid_tokens = decode_umap_grid(
                    umap_model, model_obj, tokenizer, hiddens_mean,
                    centered_hiddens=knn_centered_hiddens,
                    grid_size=UMAP_GRID_SIZE, n_workers=UMAP_GRID_WORKERS
                )
            
            plot_embedding(current_embedding, labels_sampled, digits_sampled, gt_sampled, title, position_label, i, 
                           correctness_sampled, pred_plot=pred_sampled, in_carry_plot=in_carry_sampled,
                           c_potential_plot=c_potential_sampled,
                           color_labels_plot=color_labels_sampled,
                           anchor_embedding=anchor_embedding, anchor_labels=anchor_labels,
                           grid_tokens=grid_tokens)
            
            # Interactive HTML when enabled
            if EXPORT_INTERACTIVE_HTML and sample_indices_sampled is not None and data_path is not None:
                from interactive_umap_html import create_interactive_html
                from utils.flow_utils import load_samples_meta
                samples_meta = load_samples_meta(data_path)
                
                # Output directory
                from pathlib import Path
                sub_dir = Path(data_path).stem
                mode_suffix = f"_{UMAP_ALIGNMENT_MODE}" if UMAP_ALIGNMENT_MODE != 'independent' else ""
                base_save_dir = os.path.join(SAVE_DIR, sub_dir + mode_suffix)
                model_suffix = "_with_model" if LOAD_PRETRAINED_MODEL else ""
                dim_suffix = "3D" if UMAP_DIM == 3 else "2D"
                marker_suffix = f"_{UMAP_MARKER_MODE}"  # marker mode suffix
                color_suffix = get_color_mode_suffix()
                filter_suffix = get_pred_filter_suffix()  # pred filter suffix
                pos_subdir = f"pos{position_label}_{dim_suffix}{model_suffix}{marker_suffix}{color_suffix}{filter_suffix}"
                final_save_dir = os.path.join(base_save_dir, pos_subdir)
                
                create_interactive_html(
                    embedding=current_embedding,
                    labels=labels_sampled,
                    sample_indices=sample_indices_sampled,
                    samples_meta=samples_meta,
                    position=position_label,
                    layer_idx=i,
                    save_dir=final_save_dir,
                    umap_dim=UMAP_DIM,
                    correctness=correctness_sampled,
                    marker_mode=UMAP_MARKER_MODE,
                    color_labels=color_labels_sampled,
                    color_mode=UMAP_COLOR_MODE,
                    digits=digits_sampled,
                    gt_digits=gt_sampled,
                    pred_digits=pred_sampled,
                    in_carry_digits=in_carry_sampled,
                    t_inertia_digits=c_potential_sampled,
                    pred_filter_suffix=f"{color_suffix}{filter_suffix}",
                    anchor_embedding=anchor_embedding,
                    anchor_labels=anchor_labels,
                    grid_tokens=grid_tokens
                )
            print("-" * 60 + "\n")

def build_position_runtime(pos, data_path=None, loaded_model_obj=None):
    """Build runtime context for UMAP at one position."""
    if data_path is None:
        data_path = DATA_PATH

    saved_config = None
    saved_val_indices = None
    saved_model_state = None
    if LOAD_PRETRAINED_MODEL:
        saved_config, saved_val_indices, saved_model_state = load_pretrained_and_get_val_indices(SAVED_MODEL_PATH)
        validate_position_consistency(saved_config, pos)
        effective_train_target = saved_config['train_target']
        effective_sample_filter = saved_config['sample_filter_mode']
    else:
        effective_train_target = 'is_correct'
        effective_sample_filter = 'all'

    load_kwargs = dict(
        pooling_type=POOLING,
        use_pca=USE_PCA,
        pca_dim=PCA_DIM,
        pca_mode=PCA_MODE,
        apply_carry_filter=False,
        train_target=effective_train_target,
        sample_filter_mode=effective_sample_filter,
        apply_model_norm=APPLY_MODEL_NORM,
        model_path=MODEL_PATH,
    )
    load_kwargs['loaded_model_obj'] = loaded_model_obj

    if saved_config:
        load_kwargs.update(dict(
            raw_sum_mod_10=saved_config['raw_sum_mod_10'],
            filter_by_result_len=saved_config['filter_by_result_len'],
            allowed_input_digits=saved_config['allowed_input_digits'],
            allowed_output_digits=saved_config['allowed_output_digits'],
            allowed_incoming_carries=saved_config['allowed_incoming_carries'],
            allowed_target_input_digits=saved_config['allowed_target_input_digits'],
            allowed_target_output_digits=saved_config['allowed_target_output_digits'],
            allowed_target_incoming_carries=saved_config['allowed_target_incoming_carries'],
            balance_mode=saved_config['balance_mode'],
            balance_target_classes=saved_config['balance_target_classes'],
            seed=saved_config['seed'],
        ))
    else:
        load_kwargs.update(dict(
            raw_sum_mod_10=RAW_SUM_MOD_10,
            filter_by_result_len=FILTER_BY_RESULT_LEN,
            allowed_input_digits=ALLOWED_INPUT_DIGITS,
            allowed_output_digits=ALLOWED_OUTPUT_DIGITS,
            allowed_incoming_carries=ALLOWED_INCOMING_CARRIES,
            allowed_target_input_digits=ALLOWED_TARGET_INPUT_DIGITS,
            allowed_target_output_digits=ALLOWED_TARGET_OUTPUT_DIGITS,
            allowed_target_incoming_carries=ALLOWED_TARGET_INCOMING_CARRIES,
            balance_mode=BALANCE_MODE,
            balance_target_classes=BALANCE_TARGET_CLASSES,
            seed=SEED,
        ))

    X_all_raw, y_all, y_binary, pos_idx, _, seq_len, feature_dim, _, sample_idx_all, meta = load_and_process_data(
        data_path,
        position_select=pos,
        feature_type=FEATURE_TYPE,
        **load_kwargs
    )
    print()

    pred_digits = np.asarray([int(p) if str(p).isdigit() else -1 for p in meta['preds']], dtype=np.int64)
    gt_digits = np.asarray([int(g) if str(g).isdigit() else -1 for g in meta['gts']], dtype=np.int64)
    input_digits = np.asarray([
        int(i) if str(i).isdigit() else (-2 if i == "pl" else -1) for i in meta['input_tokens']
    ], dtype=np.int64)
    sample_indices = np.asarray(
        sample_idx_all.cpu().numpy() if isinstance(sample_idx_all, torch.Tensor) else sample_idx_all,
        dtype=np.int64
    )
    incoming_carry = np.asarray(meta['incoming_carries'], dtype=np.int64)
    outgoing_carry = np.asarray(meta['outgoing_carries'], dtype=np.int64)

    if UMAP_PRED_FILTER != 'all':
        (_, pred_digits, X_all_raw, y_all, y_binary, pos_idx,
         gt_digits, input_digits, sample_indices, incoming_carry, outgoing_carry) = filter_by_pred(
            pred_digits, X_all_raw, y_all, y_binary, pos_idx,
            gt_digits, input_digits, sample_indices, incoming_carry, outgoing_carry
        )

    labels_np = y_all.cpu().numpy() if isinstance(y_all, torch.Tensor) else np.asarray(y_all)
    consistent_labels_np = y_binary.cpu().numpy() if isinstance(y_binary, torch.Tensor) else np.asarray(y_binary)
    color_labels = compute_umap_color_labels(data_path, pos, sample_indices, consistent_labels_np)

    return {
        'pos': pos,
        'data_path': data_path,
        'load_kwargs': load_kwargs,
        'saved_config': saved_config,
        'saved_val_indices': saved_val_indices,
        'saved_model_state': saved_model_state,
        'X_all_raw': X_all_raw,
        'y_all': y_all,
        'y_binary': y_binary,
        'seq_len': seq_len,
        'feature_dim': feature_dim,
        'meta': meta,
        'sample_indices': sample_indices,
        'incoming_carry': incoming_carry,
        'outgoing_carry': outgoing_carry,
        'pred_digits': pred_digits,
        'gt_digits': gt_digits,
        'input_digits': input_digits,
        'color_labels': color_labels,
        'labels_np': labels_np,
    }


def prepare_plot_selection(runtime, mode):
    """Compute plot sample indices and pretrained inference."""
    del mode  # logic unified; param kept for call-site clarity.

    selected_indices = None
    correctness = None
    labels_np = runtime['labels_np']
    incoming_carry = runtime['incoming_carry']

    if LOAD_PRETRAINED_MODEL:
        print("\n" + "="*60)
        print(f"Evaluating pretrained model (Pos: {runtime['pos']})")
        if USE_VAL_ONLY and runtime['saved_val_indices'] is not None:
            selected_indices = runtime['saved_val_indices']
            print(f"  Using val_indices from .pt as plot samples: {len(selected_indices)}")
        else:
            selected_indices = get_sampled_indices(labels_np, in_carry=incoming_carry)

        if runtime['saved_model_state'] is not None and runtime['saved_config'] is not None:
            correctness = evaluate_pretrained_model(
                runtime['X_all_raw'],
                runtime['y_all'],
                runtime['saved_config'],
                runtime['saved_model_state'],
                runtime['saved_val_indices'],
                runtime['seq_len'],
                runtime['feature_dim'],
                current_umap_position=runtime['pos'],
                data_path=runtime['data_path'],
                load_kwargs=runtime['load_kwargs'],
            )
    else:
        selected_indices = get_sampled_indices(labels_np, in_carry=incoming_carry)

    return {
        'selected_indices': selected_indices,
        'correctness': correctness,
    }


def prepare_anchor_and_model_bundle(mode):
    """Extract anchors; keep model for grid decode when needed."""
    bundle = {
        'anchor_data': None,
        'anchor_labels': None,
        'model_obj': None,
        'tokenizer': None,
        'hiddens_mean': None,
    }
    if not APPLY_MODEL_NORM:
        return bundle

    if UMAP_PRED_FILTER == 'all':
        target_digits = list(range(10))
    elif isinstance(UMAP_PRED_FILTER, int):
        target_digits = [int(UMAP_PRED_FILTER)]
    elif isinstance(UMAP_PRED_FILTER, (list, tuple)):
        target_digits = [int(d) for d in UMAP_PRED_FILTER]
    else:
        target_digits = list(range(10))

    print("\n" + "=" * 60)
    print("Loading model to extract anchors...")
    tokenizer = None
    model_obj = None
    if MODEL_BACKEND == "quanta":
        from quanta_loader import get_digit_anchors, load_quanta_model

        try:
            print(f"  Loading quanta model: {MODEL_PATH}")
            model_obj, _meta = load_quanta_model(MODEL_PATH)
            anchor_weights = get_digit_anchors(model_obj, digits=target_digits)
            bundle['anchor_data'] = anchor_weights.cpu().numpy()
            bundle['anchor_labels'] = np.array(target_digits, dtype=np.int64)
            print(f"  Extracted {len(bundle['anchor_data'])} quanta digit anchors.")
        except Exception as e:
            print(f"  [Error] quanta model load or anchor extraction failed: {e}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            print(f"  Loading model weights: {MODEL_PATH}")
            model_obj = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                device_map="auto",
                dtype="auto",
            )
            model_obj.eval()
        except Exception as e:
            print(f"  [Error] model load failed: {e}")

        if model_obj is not None:
            try:
                embeddings = get_model_embeddings(model_obj)
                embeddings.weight.data_ptr()
                model_obj.lm_head.weight.data_ptr()
                print("  Extracting digit 0-9 token anchors from LM head...")

                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
                target_ids = []
                valid_digits = []
                for d in target_digits:
                    ids = tokenizer.encode(str(d), add_special_tokens=False)
                    if len(ids) == 1:
                        target_ids.append(ids[0])
                        valid_digits.append(int(d))
                    elif len(ids) > 0:
                        target_ids.append(ids[0])
                        valid_digits.append(int(d))

                if target_ids:
                    anchor_weights = model_obj.lm_head.weight[target_ids].detach().float()
                    bundle['anchor_data'] = anchor_weights.cpu().numpy()
                    bundle['anchor_labels'] = np.array(valid_digits)
                    print(f"  Extracted {len(bundle['anchor_data'])} anchors.")
                else:
                    print("  No valid digit tokens; skipping anchor extraction.")
            except Exception as e:
                print(f"  [Warning] token anchor extraction failed: {e}")

    enable_grid_decode = (
        mode == "single"
        and ((UMAP_DIM == 2 and UMAP_GRID_DECODE_2D) or (UMAP_DIM == 3 and UMAP_GRID_DECODE_3D))
    )
    if model_obj is not None:
        bundle['model_obj'] = model_obj
        if tokenizer is not None:
            bundle['tokenizer'] = tokenizer
        
        if enable_grid_decode and MODEL_BACKEND != "quanta":
            bundle['keep_for_grid_decode'] = True
            print("  Keeping model and tokenizer for grid decode")
        else:
            bundle['keep_for_grid_decode'] = False

    return bundle


def cleanup_model_bundle(bundle):
    """Release model resources held across functions."""
    model_obj = bundle.get('model_obj')
    if model_obj is not None:
        del model_obj
        torch.cuda.empty_cache()


def run_position_pipeline(pos, data_path=None, mode="single"):
    """Run UMAP and plotting for one position."""
    import traceback

    if data_path is None:
        data_path = DATA_PATH

    if mode == "single":
        print(f"====== Processing single position {pos} ======")
    else:
        print(f"\n>>> Processing position: {pos}")

    bundle = None
    try:
        bundle = prepare_anchor_and_model_bundle(mode)
        runtime = build_position_runtime(pos, data_path, loaded_model_obj=bundle.get('model_obj'))
        
        # After load, drop language model if grid decode not needed (free VRAM)
        if bundle.get('model_obj') is not None and not bundle.get('keep_for_grid_decode', False):
            bundle['model_obj'] = None
            bundle['tokenizer'] = None
            import torch as torch_cleanup
            torch_cleanup.cuda.empty_cache()
            
        selection = prepare_plot_selection(runtime, mode)
        marker_values = compute_marker_mode_values(data_path, pos, runtime['sample_indices'], runtime['pred_digits'])

        run_umap_and_plot_v2(
            runtime['X_all_raw'],
            runtime['y_binary'],
            selection['selected_indices'],
            runtime['pred_digits'],
            runtime['gt_digits'],
            pos,
            runtime['seq_len'],
            runtime['feature_dim'],
            correctness=selection['correctness'],
            color_labels=runtime['color_labels'],
            ignore_layer_selection=(mode == "batch"),
            input_digits=runtime['input_digits'],
            data_path=data_path,
            sample_indices=runtime['sample_indices'],
            incoming_carry=runtime['incoming_carry'],
            outgoing_carry=runtime['outgoing_carry'],
            raw_sum=marker_values['raw_sum'],
            next_raw_sum=marker_values['next_raw_sum'],
            next_pred=marker_values['next_pred'],
            next_incoming_carry=marker_values['next_incoming_carry'],
            c_potential=marker_values['c_potential'],
            anchor_data=bundle.get('anchor_data'),
            anchor_labels=bundle.get('anchor_labels'),
            model_obj=bundle.get('model_obj'),
            tokenizer=bundle.get('tokenizer'),
            hiddens_mean=bundle.get('hiddens_mean'),
        )
        return f"Position {pos}: done"
    except Exception as e:
        traceback.print_exc()
        return f"Position {pos}: error - {e}"
    finally:
        if bundle is not None:
            cleanup_model_bundle(bundle)


def process_single_position(pos, data_path=None):
    """Subprocess entry: batch mode for one position."""
    return run_position_pipeline(pos, data_path=data_path, mode="batch")



parse_position = parse_position_arg


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Reproduce residual-stream UMAP visualizations.")
    parser.add_argument("--h5", "--h5", dest="data_path", default=DATA_PATH)
    parser.add_argument("--position", "--position-select", dest="position_select", type=parse_position, default=POSITION_SELECT)
    parser.add_argument("--layer", dest="umap_layer_index", type=parse_position, default=UMAP_LAYER_INDEX)
    parser.add_argument("--feature-type", default=FEATURE_TYPE)
    parser.add_argument("--pooling", default=POOLING, choices=["none", "avg", "max"])
    parser.add_argument("--model-backend", choices=["hf", "quanta"], default=MODEL_BACKEND)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--apply-model-norm", type=str_to_bool, default=APPLY_MODEL_NORM)
    parser.add_argument("--marker-mode", dest="umap_marker_mode", default=UMAP_MARKER_MODE)
    parser.add_argument("--color-mode", dest="umap_color_mode", choices=["consistent", "prefix_is_correct"], default=UMAP_COLOR_MODE)
    parser.add_argument("--max-points", dest="umap_max_points", type=int, default=UMAP_MAX_POINTS)
    parser.add_argument("--n-neighbors", dest="umap_n_neighbors", type=int, default=UMAP_N_NEIGHBORS)
    parser.add_argument("--min-dist", dest="umap_min_dist", type=float, default=UMAP_MIN_DIST)
    parser.add_argument("--metric", dest="umap_metric", default=UMAP_METRIC)
    parser.add_argument("--dim", dest="umap_dim", type=int, choices=[2, 3], default=UMAP_DIM)
    parser.add_argument("--use-seed", type=str_to_bool, default=USE_SEED)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", dest="umap_n_jobs", type=int, default=UMAP_N_JOBS)
    parser.add_argument("--script-workers", type=int, default=SCRIPT_PARALLEL_WORKERS)
    parser.add_argument("--compute-all-combinations", type=str_to_bool, default=COMPUTE_ALL_COMBINATIONS)
    parser.add_argument("--save-plot", type=str_to_bool, default=SAVE_PLOT)
    parser.add_argument("--save-plot-data", type=str_to_bool, default=SAVE_PLOT_DATA)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    return parser.parse_args()



def apply_cli_args(args):
    """Apply CLI values to legacy module-level settings used by plotting helpers."""
    updates = {
        "DATA_PATH": args.h5,
        "POSITION_SELECT": args.position_select,
        "UMAP_LAYER_INDEX": args.umap_layer_index,
        "FEATURE_TYPE": args.feature_type,
        "POOLING": args.pooling,
        "MODEL_BACKEND": args.model_backend,
        "MODEL_PATH": args.model,
        "APPLY_MODEL_NORM": args.apply_model_norm,
        "UMAP_MARKER_MODE": args.umap_marker_mode,
        "UMAP_COLOR_MODE": args.umap_color_mode,
        "UMAP_MAX_POINTS": args.umap_max_points,
        "UMAP_N_NEIGHBORS": args.umap_n_neighbors,
        "UMAP_MIN_DIST": args.umap_min_dist,
        "UMAP_METRIC": args.umap_metric,
        "UMAP_DIM": args.umap_dim,
        "USE_SEED": args.use_seed,
        "SEED": args.seed,
        "UMAP_N_JOBS": args.umap_n_jobs,
        "SCRIPT_PARALLEL_WORKERS": args.script_workers,
        "COMPUTE_ALL_COMBINATIONS": args.compute_all_combinations,
        "SAVE_PLOT": args.save_plot,
        "SAVE_PLOT_DATA": args.save_plot_data,
        "SAVE_DIR": args.save_dir,
    }
    globals().update(updates)
    os.environ['OMP_NUM_THREADS'] = str(UMAP_N_JOBS) if UMAP_N_JOBS > 0 else str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    os.environ['NUMEXPR_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    try:
        if numba is not None:
            numba.set_num_threads(UMAP_N_JOBS if UMAP_N_JOBS > 0 else numba.get_num_threads())
    except Exception as exc:
        print(f"Warning: failed to set numba thread count: {exc}")
    return args

def main():
    args = parse_cli_args()
    apply_cli_args(args)

    # Validate parameters
    validate_parameters()
    
    if COMPUTE_ALL_COMBINATIONS:
        validate_batch_mode_capabilities()
        print("====== Auto-discover all positions and run ======")
        temp_pos_meta, _, _, _, _, _ = load_token_meta_aligned(
            DATA_PATH, 'all', feature_type=FEATURE_TYPE, verbose=False
        )
        all_positions = temp_pos_meta
        # Filter out 'extra' positions
        all_positions = [pos for pos in all_positions if pos != 'extra']
        print(f"All positions: {all_positions} (skipped 'extra')")
        
        # Parallel if SCRIPT_PARALLEL_WORKERS > 1
        if SCRIPT_PARALLEL_WORKERS > 1:
            print(f"\n>>> Script-level parallelism: {SCRIPT_PARALLEL_WORKERS} workers, {UMAP_N_JOBS} threads per UMAP")
            print(f">>> Approx total threads: {SCRIPT_PARALLEL_WORKERS * UMAP_N_JOBS}")
            
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            with ProcessPoolExecutor(max_workers=SCRIPT_PARALLEL_WORKERS) as executor:
                # Submit all tasks
                future_to_pos = {executor.submit(process_single_position, pos, DATA_PATH): pos for pos in all_positions}
                
                # Collect results
                for future in as_completed(future_to_pos):
                    pos = future_to_pos[future]
                    try:
                        result = future.result()
                        print(f"[Done] {result}")
                    except Exception as e:
                        print(f"[Error] position {pos} failed: {e}")
        else:
            # Serial processing
            print("\n>>> Serial processing mode")
            for pos in all_positions:
                result = process_single_position(pos, DATA_PATH)
                print(f"[Done] {result}")
    else:
        # Skip if POSITION_SELECT is 'extra'
        if POSITION_SELECT == 'extra':
            print(f"Skipping position 'extra'; no plot")
            return
        
        result = run_position_pipeline(POSITION_SELECT, DATA_PATH, mode="single")
        print(result)

if __name__ == "__main__":
    main()
