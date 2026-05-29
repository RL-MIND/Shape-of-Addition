"""
PCA Plot Script - 2D PCA Visualization for residual-stream activation data

Based on umap_plot_script.py, simplified for PCA visualization.
"""

from pathlib import Path as _Path
import sys as _sys

_SRC_DIR = _Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import argparse
from utils.cli import parse_position_arg, str_to_bool
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from utils.flow_utils import load_and_process_data
from utils.flow_utils import (
    load_token_meta_aligned,
    load_all_token_results,
    select_positions_extended,
)


# ========== Configuration ==========
# Data path and loading settings
DATA_PATH = "results/activations/plus_num3len10_Qwen3-4B_nocheckall_balance_both"
# DATA_PATH = "results/activations/plus_num4len10_Qwen3-4B"

BALANCE_MODE = 'none'  # 'none', 'normal', 'strong'
POSITION_SELECT = 4    # Position select: 'all'/'all_no_extra'/'extra' or int/list
FEATURE_TYPE = "post_ffn"  # 'post_attn' | 'post_ffn' | 'flows' | 'velocities' | 'curvatures'
POOLING = 'none'       # 'none' | 'avg' | 'max'

# PCA parameters
PCA_LAYER_INDEX = 36   # Layer index for PCA (None = flatten all layers)
PCA_AXIS_X = 0         # Principal component index for X axis (0-based)
PCA_AXIS_Y = 1         # Principal component index for Y axis (0-based)
PCA_MAX_POINTS = 2000  # Max points per class (0 = no limit)

# Marker mode settings
# "point": plain dots
# "pred": use predicted digits (pred_digits)
# "input": use input digits (input_digits)
# "in_carry": use incoming carry (incoming_carry)
# "out_carry": use outgoing carry (outgoing_carry)
PCA_MARKER_MODE = "pred"  # "point", "pred", "input", "in_carry", "out_carry"

# Save settings
SAVE_PLOT = True
SAVE_DIR = "plots_pca"

# ========== Sampling ==========
def get_sampled_indices(labels_np, max_points, seed=42):
    """
    Sample positive and negative classes separately; return selected indices.

    Args:
        labels_np: Label array
        max_points: Max points per class (0 = no limit)
        seed: Random seed

    Returns:
        selected_indices: Sampled index array
    """
    np.random.seed(seed)

    if max_points <= 0:
        # No limit: return all indices
        return np.arange(len(labels_np))

    pos_indices = np.where(labels_np == 1)[0]
    neg_indices = np.where(labels_np == 0)[0]

    if len(pos_indices) > max_points:
        pos_indices = np.random.choice(pos_indices, size=max_points, replace=False)
    if len(neg_indices) > max_points:
        neg_indices = np.random.choice(neg_indices, size=max_points, replace=False)

    selected_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(selected_indices)

    return selected_indices

# ========== Plotting ==========
def plot_pca_2d(pca_data, labels, markers, marker_mode, position, layer_idx,
                axis_x, axis_y, gt_digits=None, pred_digits=None, save_path=None):
    """
    Plot 2D PCA with text markers.

    Args:
        pca_data: PCA-reduced data (N, n_components)
        labels: Label array (for colors)
        markers: Primary marker values (depends on marker_mode)
        marker_mode: Marker mode
        position: Position index
        layer_idx: Layer index
        axis_x: Principal component index for X axis
        axis_y: Principal component index for Y axis
        gt_digits: Ground-truth digits (shown for wrong samples)
        pred_digits: Predicted digits (auxiliary display)
        save_path: Save path (None = do not save)
    """
    fig, ax = plt.subplots(figsize=(15, 12))

    colors = {1: 'blue', 0: 'red'}

    # Legend (dummy markers)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='$1$', color='w', markerfacecolor='blue',
               markeredgecolor='blue', markersize=10, label='Positive (Correct)'),
        Line2D([0], [0], marker='$1$', color='w', markerfacecolor='red',
               markeredgecolor='red', markersize=10, label='Negative (Wrong)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    if marker_mode == "point":
        # Plain scatter plot
        color_array = np.where(labels == 1, 'blue', 'red')
        ax.scatter(pca_data[:, axis_x], pca_data[:, axis_y],
                   c=color_array, alpha=0.6, s=20)
    else:
        # Text marker mode
        # Transparent scatter first to set axis limits
        ax.scatter(pca_data[:, axis_x], pca_data[:, axis_y], s=0, alpha=0)

        for i in range(len(pca_data)):
            x, y = pca_data[i, axis_x], pca_data[i, axis_y]
            lbl = labels[i]
            d = markers[i]

            # Text content and color
            txt = "?"
            color = colors.get(lbl, 'black')

            # label=1 is correct/positive
            is_correct = (lbl == 1)

            # Build display text
            if d >= 0 or d == -2:  # -2 is PL
                if not is_correct and gt_digits is not None:
                    # Wrong sample (negative, red): show d(gt)
                    g = gt_digits[i]
                    if g >= 0:
                        if marker_mode == "input":
                             txt = "PL" if d == -2 else str(int(d))
                        elif marker_mode in ["in_carry", "out_carry"]:
                             p = pred_digits[i] if pred_digits is not None else g
                             txt = f"{int(d)}({int(p)})"
                        else:
                             txt = (f"PL({int(g)})" if d == -2 else f"{int(d)}({int(g)})")
                    else:
                        txt = str(int(d))
                else:
                    # Correct sample (positive, blue): show d only
                    # in_carry/out_carry may also show pred
                    if marker_mode in ["in_carry", "out_carry"] and pred_digits is not None:
                        p = pred_digits[i]
                        txt = f"{int(d)}({int(p)})"
                    else:
                        txt = "PL" if d == -2 else str(int(d))

            ax.text(x, y, txt, color=color, fontsize=8, ha='center', va='center', alpha=0.6)

    ax.set_xlabel(f'PC{axis_x + 1}', fontsize=12)
    ax.set_ylabel(f'PC{axis_y + 1}', fontsize=12)

    layer_str = f"Layer {layer_idx}" if layer_idx is not None else "All Layers"
    ax.set_title(f'PCA - Position {position}, {layer_str}\n'
                 f'Marker: {marker_mode}, Axes: PC{axis_x+1} vs PC{axis_y+1}',
                 fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.close()

# ========== Main ==========

parse_position = parse_position_arg


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Reproduce residual-stream PCA visualizations.")
    parser.add_argument("--h5", "--h5", dest="data_path", default=DATA_PATH)
    parser.add_argument("--balance-mode", choices=["none", "normal", "strong"], default=BALANCE_MODE)
    parser.add_argument("--position", "--position-select", dest="position_select", type=parse_position, default=POSITION_SELECT)
    parser.add_argument("--feature-type", default=FEATURE_TYPE)
    parser.add_argument("--pooling", default=POOLING, choices=["none", "avg", "max"])
    parser.add_argument("--layer", dest="pca_layer_index", type=parse_position, default=PCA_LAYER_INDEX)
    parser.add_argument("--axis-x", type=int, default=PCA_AXIS_X)
    parser.add_argument("--axis-y", type=int, default=PCA_AXIS_Y)
    parser.add_argument("--max-points", type=int, default=PCA_MAX_POINTS)
    parser.add_argument("--marker-mode", choices=["point", "pred", "input", "in_carry", "out_carry"], default=PCA_MARKER_MODE)
    parser.add_argument("--save-plot", type=str_to_bool, default=SAVE_PLOT)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    return parser.parse_args()



def apply_cli_args(args):
    """Return parsed CLI args; kept as a compatibility hook without mutating globals."""
    return args

def main():
    args = apply_cli_args(parse_cli_args())
    data_path = args.h5
    balance_mode = args.balance_mode
    position_select = args.position_select
    feature_type = args.feature_type
    pooling = args.pooling
    pca_layer_index = args.pca_layer_index
    pca_axis_x = args.axis_x
    pca_axis_y = args.axis_y
    pca_max_points = args.max_points
    pca_marker_mode = args.marker_mode
    save_plot = args.save_plot
    save_dir_arg = args.save_dir
    print("=" * 60)
    print("PCA Plot Script")
    print("=" * 60)

    # Load data
    print(f"\nLoading data: {data_path}")
    print(f"Position select: {position_select}")

    X_all_raw, y_all, y_binary, pos_idx, selected_positions, seq_len, feature_dim, _, _, _ = load_and_process_data(
        data_path, position_select=position_select, feature_type=feature_type,
        pooling_type=pooling, use_pca=False, pca_dim=100, pca_mode='per_layer',
        apply_carry_filter=False,
        train_target='is_correct',
        sample_filter_mode='all',
    )

    # Load token meta
    _, _, _, preds_list, gt_chars_list, input_tokens_list = load_token_meta_aligned(
        data_path, position_select, feature_type=feature_type, verbose=False
    )

    pred_digits = np.asarray([int(p) if str(p).isdigit() else -1 for p in preds_list], dtype=np.int64)
    gt_digits = np.asarray([int(g) if str(g).isdigit() else -1 for g in gt_chars_list], dtype=np.int64)
    input_digits = np.asarray([
        int(i) if str(i).isdigit() else (-2 if i == "pl" else -1) for i in input_tokens_list
    ], dtype=np.int64)

    # Carry information
    all_token_results = load_all_token_results(
        data_path, position_select, feature_type=feature_type, verbose=False
    )
    _, _, _, _, _, _, sample_idx_list, incoming_carries_list, outgoing_carries_list = select_positions_extended(
        all_token_results, position_select, verbose=False
    )
    incoming_carry = np.asarray(incoming_carries_list, dtype=np.int64)
    outgoing_carry = np.asarray(outgoing_carries_list, dtype=np.int64)

    # Convert tensors to numpy
    labels_np = y_binary.cpu().numpy() if isinstance(y_binary, torch.Tensor) else np.asarray(y_binary)
    X_np = X_all_raw.numpy() if isinstance(X_all_raw, torch.Tensor) else np.asarray(X_all_raw)

    print(f"Total samples: {len(X_np)}")
    print(f"Sequence length (layers): {seq_len}, feature dim: {feature_dim}")

    # Sampling
    selected_indices = get_sampled_indices(labels_np, pca_max_points)
    print(f"Samples after sampling: {len(selected_indices)}")

    # Subset
    X_selected = X_np[selected_indices]
    labels_selected = labels_np[selected_indices]
    pred_selected = pred_digits[selected_indices]
    gt_selected = gt_digits[selected_indices]
    input_selected = input_digits[selected_indices]
    in_carry_selected = incoming_carry[selected_indices]
    out_carry_selected = outgoing_carry[selected_indices]

    # Markers by mode
    if pca_marker_mode == "pred":
        markers = pred_selected
    elif pca_marker_mode == "input":
        markers = input_selected
    elif pca_marker_mode == "in_carry":
        markers = in_carry_selected
    elif pca_marker_mode == "out_carry":
        markers = out_carry_selected
    else:
        markers = np.zeros(len(labels_selected), dtype=np.int64)

    # Extract layer
    X_reshaped = X_selected.reshape(len(X_selected), seq_len, feature_dim)

    if pca_layer_index is not None:
        layer_idx = pca_layer_index
        if layer_idx >= seq_len:
            print(f"Warning: pca_layer_index ({layer_idx}) out of range (max={seq_len-1}), using layer 0")
            layer_idx = 0
        X_layer = X_reshaped[:, layer_idx, :]
        print(f"Using data from layer {layer_idx}")
    else:
        X_layer = X_selected.reshape(len(X_selected), -1)
        layer_idx = None
        print("Using all layers (flattened)")

    # PCA
    n_components = max(pca_axis_x, pca_axis_y) + 1
    print(f"\nRunning PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X_layer)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

    # Plot
    if save_plot:
        save_dir = Path(save_dir_arg)
        save_dir.mkdir(parents=True, exist_ok=True)

        layer_str = f"layer{layer_idx}" if layer_idx is not None else "all_layers"
        filename = f"pca_pos{position_select}_{layer_str}_pc{pca_axis_x+1}_pc{pca_axis_y+1}_{pca_marker_mode}.png"
        save_path = save_dir / filename
    else:
        save_path = None

    plot_pca_2d(pca_data, labels_selected, markers, pca_marker_mode,
                position_select, layer_idx, pca_axis_x, pca_axis_y,
                gt_digits=gt_selected, pred_digits=pred_selected, save_path=save_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
