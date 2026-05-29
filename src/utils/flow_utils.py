"""
Shared utilities for residual-stream activation processing and result loading.

Extracted common logic from generate.py, classifier.py, and flow_diff.py:
- Position selection parsing
- HDF5/pickle result loading for all_token_results
- Position key extraction from HDF5 groups
- Flattening token results into sample lists
- Class balancing helpers (numpy-based)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import h5py  # type: ignore

    HAS_H5PY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_H5PY = False


# =========================
# Position helpers
# =========================
def parse_position_select(raw: str) -> Union[str, int, List[int]]:
    """Parse CLI/string position_select into supported types."""
    lowered = raw.lower()
    if lowered in {"all", "all_no_extra", "extra"}:
        return lowered
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if raw.strip().lstrip("-").isdigit():
        return int(raw)
    return raw


def extract_position_key(pos_name: str, pos_group: Any) -> Union[int, str]:
    """Recover original position key from HDF5 node (aligned with generate.py)."""
    orig = pos_group.attrs.get("original_key")
    if isinstance(orig, bytes):
        orig = orig.decode("utf-8")
    if isinstance(orig, str):
        if orig.startswith("int:"):
            try:
                return int(orig.split(":", 1)[1])
            except Exception:
                return orig
        if orig.isdigit():
            return int(orig)
        return orig
    if isinstance(pos_name, str) and pos_name.startswith("pos_"):
        try:
            return int(pos_name.split("_", 1)[1])
        except Exception:
            return pos_name
    return pos_name


def _collect_position_names(positions_group: Any) -> Tuple[List[int], List[str]]:
    """Infer numeric/string positions from HDF5 group attributes or keys."""
    numeric_positions = list(positions_group.attrs.get("numeric_positions", []))
    string_positions = list(positions_group.attrs.get("string_positions", []))
    numeric_positions = [int(p) for p in numeric_positions]
    string_positions = [str(p) for p in string_positions]

    if not numeric_positions and not string_positions:
        for key in positions_group.keys():
            if key.startswith("pos_"):
                suffix = key[4:]
                if suffix.lstrip("-").isdigit():
                    numeric_positions.append(int(suffix))
                else:
                    string_positions.append(str(suffix))
            else:
                string_positions.append(str(key))
    return numeric_positions, string_positions


def _resolve_positions_to_load(
    position_select: Union[str, int, List[int]],
    numeric_positions: List[int],
    string_positions: List[str],
) -> List[Union[int, str]]:
    if position_select == "all":
        return numeric_positions + string_positions
    if position_select == "all_no_extra":
        return numeric_positions
    if position_select == "extra":
        return ["extra"] if "extra" in string_positions else []
    if isinstance(position_select, int):
        return [position_select] if position_select in numeric_positions else []
    if isinstance(position_select, list):
        return [p for p in position_select if p in numeric_positions + string_positions]
    return numeric_positions + string_positions


def decode_array(arr: List[Any]) -> List[str]:
    """Decode bytes array to string list."""
    return [a.decode("utf-8") if isinstance(a, (bytes, bytearray)) else str(a) for a in arr]


def pos_sort_value(pos: Union[int, str]) -> Tuple[int, Union[int, str]]:
    """Helper for sorting position keys (ints first, then strings)."""
    return (0, pos) if isinstance(pos, (int, np.integer)) else (1, str(pos))


_BASE_FLOW_FEATURE_TYPES = {"flows", "velocities", "curvatures"}
_FLOW_DATASET_KEYS = ("flows", "flows_post_attn", "flows_post_ffn")


def parse_feature_type_config(feature_type: str) -> Dict[str, str]:
    """
    Parse FEATURE_TYPE into:
    - dataset_key: which raw flow dataset to load from file
    - feature_kind: how to transform the loaded raw flow

    Supported examples:
    - 'flows', 'velocities', 'curvatures'
    - 'post_attn', 'post_ffn'
    - 'post_attn_flows', 'post_attn_velocities', 'post_attn_curvatures'
    - 'post_ffn_flows', 'post_ffn_velocities', 'post_ffn_curvatures'
    - 'flows_post_attn', 'flows_post_ffn'
    """
    feature_type = str(feature_type).strip()

    if feature_type in _BASE_FLOW_FEATURE_TYPES:
        return {"dataset_key": "flows", "feature_kind": feature_type}

    dataset_aliases = {
        "post_attn": "flows_post_attn",
        "post_ffn": "flows_post_ffn",
        "flows_post_attn": "flows_post_attn",
        "flows_post_ffn": "flows_post_ffn",
    }
    if feature_type in dataset_aliases:
        return {"dataset_key": dataset_aliases[feature_type], "feature_kind": "flows"}

    for prefix, dataset_key in dataset_aliases.items():
        for feature_kind in _BASE_FLOW_FEATURE_TYPES:
            if feature_type == f"{prefix}_{feature_kind}":
                return {"dataset_key": dataset_key, "feature_kind": feature_kind}

    raise ValueError(
        "Unsupported FEATURE_TYPE: "
        f"{feature_type}. Supported: 'flows'/'velocities'/'curvatures', "
        "'post_attn'/'post_ffn', and "
        "'post_attn_flows'/'post_attn_velocities'/'post_attn_curvatures', "
        "'post_ffn_flows'/'post_ffn_velocities'/'post_ffn_curvatures'."
    )


def resolve_flow_dataset_key(
    available_keys: List[str], feature_type: str, context: str = ""
) -> str:
    """
    Resolve which dataset key should be used for the requested FEATURE_TYPE.

    For legacy single-flow files, FEATURE_TYPE='flows' still maps to 'flows'.
    For *_both.h5 files, callers must specify post_attn/post_ffn explicitly.
    """
    feature_cfg = parse_feature_type_config(feature_type)
    requested_key = feature_cfg["dataset_key"]
    available_flow_keys = [key for key in _FLOW_DATASET_KEYS if key in available_keys]

    if requested_key in available_keys:
        return requested_key

    if requested_key == "flows" and len(available_flow_keys) == 1:
        return available_flow_keys[0]

    context_prefix = f"{context} " if context else ""
    if requested_key == "flows" and len(available_flow_keys) > 1:
        raise ValueError(
            f"{context_prefix}Multiple flow datasets detected {available_flow_keys}; "
            "set FEATURE_TYPE='post_attn' or FEATURE_TYPE='post_ffn' explicitly."
        )

    raise ValueError(
        f"{context_prefix}Missing required feature dataset '{requested_key}'; "
        f"available: {available_flow_keys or available_keys}"
    )


# =========================
# Loading helpers
# =========================
def load_all_token_results_from_pickle(
    file_path: Path, feature_type: str = "flows", verbose: bool = True
) -> Dict[Any, Dict[str, Any]]:
    import pickle

    if verbose:
        print(f"Loading results from pickle: {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    raw_results = data["all_token_results"]
    results: Dict[Any, Dict[str, Any]] = {}
    for pos, pos_data_raw in raw_results.items():
        pos_data = dict(pos_data_raw)
        flow_key = resolve_flow_dataset_key(
            list(pos_data.keys()), feature_type, context=f"pickle position {pos}"
        )
        if flow_key in pos_data:
            pos_data["flows"] = pos_data[flow_key]
            pos_data["_loaded_flow_key"] = flow_key
        results[pos] = pos_data
    return results


def load_all_token_results_from_hdf5(
    file_path: Path,
    position_select: Union[str, int, List[int]],
    feature_type: str = "flows",
    verbose: bool = True,
) -> Dict[Any, Dict[str, Any]]:
    if not HAS_H5PY:
        raise ImportError("h5py is not installed; cannot read HDF5 files")

    if verbose:
        print(f"Loading results from HDF5: {file_path}")
    results: Dict[Any, Dict[str, Any]] = {}

    with h5py.File(file_path, "r") as hf:  # type: ignore
        positions_group = hf["all_token_results"]
        numeric_positions, string_positions = _collect_position_names(positions_group)
        positions_to_load = _resolve_positions_to_load(position_select, numeric_positions, string_positions)

        if verbose:
            print(f"  HDF5 available positions: numeric={numeric_positions}, string={string_positions}")
            print(f"  Loading positions: {positions_to_load}")

        for pos in positions_to_load:
            pos_name = f"pos_{pos}" if isinstance(pos, int) else str(pos)
            if pos_name not in positions_group:
                continue
            pos_group = positions_group[pos_name]
            key = extract_position_key(pos_name, pos_group)

            pos_data: Dict[str, Any] = {}
            available_keys = list(pos_group.keys())
            if not available_keys:
                if verbose: print(f"    Position {key}: empty group, skipping")
                continue
                
            flow_key = resolve_flow_dataset_key(
                available_keys, feature_type, context=f"HDF5 position {key}"
            )
            if flow_key in pos_group:
                flows_array = pos_group[flow_key][:]
                pos_data["flows"] = [flows_array[i] for i in range(len(flows_array))]
                pos_data["_loaded_flow_key"] = flow_key
            if "labels" in pos_group:
                pos_data["labels"] = list(pos_group["labels"][:])
            if "preds" in pos_group:
                pos_data["preds"] = decode_array(pos_group["preds"][:])
            if "gt_chars" in pos_group:
                pos_data["gt_chars"] = decode_array(pos_group["gt_chars"][:])
            
            # Read sample_indices if present
            if "sample_indices" in pos_group:
                pos_data["sample_indices"] = list(pos_group["sample_indices"][:])
            
            # Read carry information if present
            if "incoming_carries" in pos_group:
                pos_data["incoming_carries"] = list(pos_group["incoming_carries"][:])
            if "outgoing_carries" in pos_group:
                pos_data["outgoing_carries"] = list(pos_group["outgoing_carries"][:])

            results[key] = pos_data
            if verbose:
                print(f"    Position {key}: loaded {len(pos_data.get('labels', []))} samples")

    return results


def load_all_token_results(
    file_path: Union[str, Path],
    position_select: Union[str, int, List[int]],
    feature_type: str = "flows",
    verbose: bool = True,
) -> Dict[Any, Dict[str, Any]]:
    """
    Load all_token_results from pickle or HDF5 (preferring explicit suffix).
    Accepts path with or without extension; will try .h5 then .pkl when no suffix.
    """
    file_path = Path(file_path)
    if file_path.suffix in {".pkl", ".h5"}:
        base_path = file_path
    else:
        h5_path = file_path.with_suffix(".h5")
        pkl_path = file_path.with_suffix(".pkl")
        if h5_path.exists():
            base_path = h5_path
        elif pkl_path.exists():
            base_path = pkl_path
        else:
            raise FileNotFoundError(f"Result file not found: {file_path} / {h5_path} / {pkl_path}")

    if base_path.suffix == ".h5":
        return load_all_token_results_from_hdf5(
            base_path, position_select, feature_type=feature_type, verbose=verbose
        )
    return load_all_token_results_from_pickle(base_path, feature_type=feature_type, verbose=verbose)


def load_samples_meta(file_path: Union[str, Path]) -> Dict[int, Dict[str, str]]:
    """
    Load per-sample metadata (question/pred/gt) from the HDF5 samples group.
    Returns an empty dict if the file is not .h5 or lacks a samples group.
    """
    file_path = Path(file_path)
    if file_path.suffix not in {".h5", ""}:
        return {}

    base_path = file_path
    if base_path.suffix == "":
        candidate = base_path.with_suffix(".h5")
        if candidate.exists():
            base_path = candidate
        else:
            return {}

    if base_path.suffix != ".h5" or not base_path.exists():
        return {}

    if not HAS_H5PY:
        raise ImportError("h5py is not installed; cannot read the samples group from HDF5")

    meta: Dict[int, Dict[str, str]] = {}
    with h5py.File(base_path, "r") as hf:  # type: ignore
        if "samples" not in hf:
            return {}
        grp = hf["samples"]
        sample_idx = grp.get("sample_idx")
        questions = grp.get("question")
        preds = grp.get("pred")
        gts = grp.get("gt")
        if sample_idx is None or questions is None or preds is None or gts is None:
            return {}
        idx_arr = np.asarray(sample_idx[:], dtype=np.int64)
        q_arr = decode_array(questions[:])
        p_arr = decode_array(preds[:])
        g_arr = decode_array(gts[:])
        for idx, q, p, g in zip(idx_arr, q_arr, p_arr, g_arr):
            meta[int(idx)] = {"question": q, "pred": p, "gt": g}
    return meta


def select_positions(
    all_token_results: Dict[Any, Dict[str, Any]], position_select: Union[str, int, List[int]]
) -> Tuple[List[Any], List[int], List[int], List[np.ndarray]]:
    """
    Filter positions and flatten to sample lists (flows + labels).
    Returns: selected_positions, labels_list, position_idx_list, flows_list
    """
    # Reuse extended version but just return the original 4 outputs
    selected, labels, pos_idx, flows, _, _, _, _, _ = select_positions_extended(all_token_results, position_select)
    return selected, labels, pos_idx, flows


def select_positions_extended(
    all_token_results: Dict[Any, Dict[str, Any]], position_select: Union[str, int, List[int]], verbose: bool = True
) -> Tuple[
    List[Any],
    List[int],
    List[int],
    List[np.ndarray],
    List[str],
    List[str],
    List[int],
    List[int],
    List[int],
]:
    """
    Filter positions and flatten to sample lists (flows + labels + preds + gt_chars + carries).
    Returns: selected_positions, labels_list, position_idx_list, flows_list, preds_list, gt_chars_list, 
             sample_idx_list, incoming_carries_list, outgoing_carries_list
    """
    available_positions = list(all_token_results.keys())
    numeric_positions = sorted([p for p in available_positions if isinstance(p, int)])
    # print("!!!!!!!!!!!!!", available_positions)

    if position_select == "all":
        selected_positions = available_positions
    elif position_select == "all_no_extra":
        selected_positions = numeric_positions
    elif position_select == "extra":
        selected_positions = ["extra"] if "extra" in available_positions else []
    elif isinstance(position_select, int):
        selected_positions = [position_select] if position_select in available_positions else []
    elif isinstance(position_select, list):
        selected_positions = [p for p in position_select if p in available_positions]
    else:
        raise ValueError(f"Unsupported position_select value: {position_select}")

    if not selected_positions:
        raise ValueError(f"No positions matched the selection: {position_select}")

    if verbose:
        print(f"Selected positions: {selected_positions}")

    flows_list: List[np.ndarray] = []
    labels_list: List[int] = []
    position_idx_list: List[int] = []
    preds_list: List[str] = []
    gt_chars_list: List[str] = []
    pos_to_idx = {pos: i for i, pos in enumerate(selected_positions)}
    sample_idx_list: List[int] = []
    incoming_carries_list: List[int] = []
    outgoing_carries_list: List[int] = []

    for pos in selected_positions:
        pos_data = all_token_results[pos]
        flows = pos_data.get("flows", [])
        labels = pos_data.get("labels", [])
        preds = pos_data.get("preds", [""] * len(labels))
        gt_chars = pos_data.get("gt_chars", [""] * len(labels))
        # Load carry info; default to -1 when missing (invalid marker)
        incoming_carries = pos_data.get("incoming_carries", [-1] * len(labels))
        outgoing_carries = pos_data.get("outgoing_carries", [-1] * len(labels))
        # Prefer saved sample_indices; fall back to local indices
        saved_indices = pos_data.get("sample_indices")
        if saved_indices is None or len(saved_indices) != len(labels):
            saved_indices = list(range(len(labels)))

        if len(flows) != len(labels):
            raise ValueError(f"Position {pos}: flows/labels count mismatch: {len(flows)} vs {len(labels)}")
        
        if verbose:
            print(f"  Position {pos}: {len(labels)} samples, positive {sum(labels)}, negative {len(labels) - sum(labels)}")
        for i, (flow, label, pred, gt, s_idx, inc_carry, out_carry) in enumerate(
            zip(flows, labels, preds, gt_chars, saved_indices, incoming_carries, outgoing_carries)
        ):
            flows_list.append(np.asarray(flow))
            labels_list.append(1 if bool(label) else 0)
            position_idx_list.append(pos_to_idx[pos])
            preds_list.append(str(pred))
            gt_chars_list.append(str(gt))
            sample_idx_list.append(int(s_idx))  # Use the true sample_idx
            incoming_carries_list.append(int(inc_carry) if inc_carry is not None else -1)
            outgoing_carries_list.append(int(out_carry) if out_carry is not None else -1)

    return (
        selected_positions,
        labels_list,
        position_idx_list,
        flows_list,
        preds_list,
        gt_chars_list,
        sample_idx_list,
        incoming_carries_list,
        outgoing_carries_list,
    )


def load_token_meta_aligned(
    file_path: Union[str, Path],
    position_select: Union[str, int, List[int]],
    feature_type: str = "flows",
    verbose: bool = True,
) -> Tuple[List[Any], np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Load meta aligned with flattened sample order (labels/preds/gt_chars/position_idx/input_tokens).

    Order matches classifier.load_and_process_data select_positions flattening:
    iterate selected_positions, then samples within each position in original order.

    input_tokens: input token at the current position (prediction from the previous position).
                  For pos=0, set to 'pl'.
    """
    # 1. Preload to determine selected positions (verbosity controlled externally)
    all_token_results = load_all_token_results(
        file_path, position_select, feature_type=feature_type, verbose=verbose
    )
    selected_positions, labels_list, position_idx_list, _flows_list, preds_list, gt_chars_list, sample_idx_list, _, _ = (
        select_positions_extended(all_token_results, position_select, verbose=verbose)
    )

    # 2. Determine prerequisite positions to load
    needed_prev_positions = []
    for pos in selected_positions:
        if isinstance(pos, int) and pos > 0:
            prev_pos = pos - 1
            if prev_pos not in all_token_results:
                needed_prev_positions.append(prev_pos)
    
    # Reload if prerequisite positions are missing (second load is usually quiet)
    if needed_prev_positions:
        combined_select = list(set(selected_positions) | set(needed_prev_positions))
        all_token_results = load_all_token_results(
            file_path, combined_select, feature_type=feature_type, verbose=False
        )

    # 3. Build (pos, sample_idx) -> pred mapping
    pred_map = {} # pos -> {sample_idx: pred}
    for pos, data in all_token_results.items():
        preds = data.get("preds", [])
        indices = data.get("sample_indices")
        if indices is None:
            indices = list(range(len(preds)))
        
        pos_map = {}
        for s_idx, p in zip(indices, preds):
            pos_map[int(s_idx)] = str(p)
        pred_map[pos] = pos_map

    # 4. Build input_tokens_list
    input_tokens_list = []
    
    # Align over flattened samples (select_positions_extended output is already flat)
    # Recover original pos keys from flattened position_idx_list
    idx_to_pos = {i: pos for i, pos in enumerate(selected_positions)}

    # preds_list, gt_chars_list, etc. are already flat; build input_tokens_list in sync
    # using original pos and sample_idx from select_positions_extended
    
    for p_idx, s_idx in zip(position_idx_list, sample_idx_list):
        pos = idx_to_pos[p_idx]
        if isinstance(pos, int):
            if pos == 0:
                input_tokens_list.append("pl")
            else:
                prev_pos = pos - 1
                input_token = pred_map.get(prev_pos, {}).get(s_idx, "")
                input_tokens_list.append(input_token)
        else:
            # For string positions (e.g. 'extra'), use empty string for now
            input_tokens_list.append("")

    return (
        selected_positions,
        np.asarray(labels_list, dtype=np.int64),
        np.asarray(position_idx_list, dtype=np.int64),
        preds_list,
        gt_chars_list,
        input_tokens_list,
    )


# =========================
# Balancing (numpy-based)
# =========================
def balance_indices_numpy(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Globally balance classes and return kept indices (shuffled)."""
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"\nClass distribution before global balancing: {class_counts}")

    min_count = counts.min()
    kept_indices: List[int] = []
    for cls in unique:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) > min_count:
            selected = rng.choice(cls_indices, size=min_count, replace=False)
        else:
            selected = cls_indices
        kept_indices.extend(selected.tolist())
        print(f"  Class {cls}: kept {len(selected)} / {len(cls_indices)}")

    rng.shuffle(kept_indices)
    return np.asarray(kept_indices, dtype=np.int64)


def balance_dataset_numpy(
    flows: np.ndarray, labels: np.ndarray, pos_indices: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Globally balance classes; returns balanced arrays (shuffled)."""
    kept = balance_indices_numpy(labels, seed=seed)
    return flows[kept], labels[kept], pos_indices[kept]


def balance_indices_per_position_numpy(
    labels: np.ndarray, pos_indices: np.ndarray, position_names: List[Any], seed: int = 42
) -> np.ndarray:
    """Balance within each position; return kept indices (shuffled)."""
    rng = np.random.default_rng(seed)
    unique_positions = np.unique(pos_indices)
    kept_indices: List[int] = []

    print("\nPer-position strong balancing")
    for pos_idx in unique_positions:
        mask = pos_indices == pos_idx
        pos_labels = labels[mask]
        pos_indices_arr = np.where(mask)[0]
        pos_name = position_names[pos_idx] if pos_idx < len(position_names) else pos_idx
        if len(pos_labels) == 0:
            print(f"  Position {pos_name}: no samples, skipping")
            continue
        unique, counts = np.unique(pos_labels, return_counts=True)
        if len(unique) < 2:
            print(f"  Position {pos_name}: single class, dropping all {len(pos_labels)}")
            continue
        min_count = counts.min()
        print(f"  Position {pos_name}: class distribution {dict(zip(unique.tolist(), counts.tolist()))}, target {min_count} each")
        for cls in unique:
            cls_indices = pos_indices_arr[pos_labels == cls]
            if len(cls_indices) > min_count:
                selected = rng.choice(cls_indices, size=min_count, replace=False)
            else:
                selected = cls_indices
            kept_indices.extend(selected.tolist())
            print(f"    Class {cls}: kept {len(selected)} / {len(cls_indices)}")

    rng.shuffle(kept_indices)
    return np.asarray(kept_indices, dtype=np.int64)


def balance_dataset_per_position_numpy(
    flows: np.ndarray, labels: np.ndarray, pos_indices: np.ndarray, position_names: List[Any], seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance within each position, then combine; returns balanced arrays (shuffled)."""
    kept = balance_indices_per_position_numpy(labels, pos_indices, position_names, seed=seed)
    return flows[kept], labels[kept], pos_indices[kept]


import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def train_classifier_on_umap(plot_data, labels_plot, test_size=0.2, random_state=42, classifier_type='rf', use_grid_search=False):
    """
    Train a nonlinear classifier on 2D UMAP-reduced data.
    
    Args:
        plot_data: 2D UMAP-reduced data, shape (N, 2)
        labels_plot: Binary labels, shape (N,), values 0 or 1
        test_size: Test set fraction, default 0.2
        random_state: Random seed
        classifier_type: Classifier type
            - 'rf': Random Forest (default)
            - 'svm': Support Vector Machine with RBF kernel
            - 'mlp': Multi-layer Perceptron
        use_grid_search: Use grid search for hyperparameters ('rf' and 'svm' only), default False
    
    Returns:
        clf: Trained classifier
        X_train, X_test, y_train, y_test: Split datasets
        metrics: Dict with accuracy, AUC, etc.
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        plot_data, labels_plot, test_size=test_size, random_state=random_state, stratify=labels_plot
    )
    
    print(f"Train size: {len(X_train)}, test size: {len(X_test)}")
    print(f"Train class distribution: positive {np.sum(y_train==1)}, negative {np.sum(y_train==0)}")
    print(f"Test class distribution: positive {np.sum(y_test==1)}, negative {np.sum(y_test==0)}")
    
    # Choose classifier
    if classifier_type == 'rf':
        if use_grid_search:
            print("\nClassifier: Random Forest (grid search)")
            # Random Forest parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            clf = GridSearchCV(
                base_clf, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
            print("\nClassifier: Random Forest")
    elif classifier_type == 'svm':
        if use_grid_search:
            print("\nClassifier: SVM (RBF kernel, grid search)")
            # SVM parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
            }
            base_clf = SVC(kernel='rbf', probability=True, random_state=random_state)
            clf = GridSearchCV(
                base_clf, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
        else:
            clf = SVC(kernel='rbf', C=10, gamma=10, probability=True, random_state=random_state)
            print("\nClassifier: SVM (RBF kernel)")
    elif classifier_type == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='tanh', max_iter=1000, random_state=random_state)
        print("\nClassifier: Multi-layer Perceptron")
        if use_grid_search:
            print("  Note: MLP does not support grid search; using default parameters")
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Train
    print("Training...")
    clf.fit(X_train, y_train)
    
    # Print best params after grid search
    if use_grid_search and classifier_type in ['rf', 'svm']:
        print(f"\nBest parameters: {clf.best_params_}")
        print(f"Best cross-validation score (AUC): {clf.best_score_:.4f}")
        # Use best estimator
        clf = clf.best_estimator_
    
    # Predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]  # Positive-class probability
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print("\n" + "="*60)
    print("Training results:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print("="*60)
    
    print("\nTest classification report:")
    print(classification_report(y_test, y_test_pred, target_names=['negative', 'positive']))
    
    print("\nTest confusion matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    metrics = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'confusion_matrix': cm
    }
    
    return clf, X_train, X_test, y_train, y_test, metrics

# ==========================================
# ========== Moved from classifier.py ======
# ==========================================

import torch
from sklearn.decomposition import PCA

def get_model_norm(model):
    """
    Automatically locate the model's final norm layer.
    Supports:
    - Qwen/Llama (model.model.norm)
    - Gemma 3 (model.model.language_model.norm)
    - HookedTransformer / Quanta (model.ln_final)
    - Others (model.norm)
    """
    # 1. Gemma 3: model -> model(Gemma3Model) -> language_model(Gemma3TextModel) -> norm
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "norm"):
        return model.model.language_model.norm
    
    # 2. Qwen / Llama / Standard: model -> model -> norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
        
    # 3. Flat or wrapped differently: model -> norm
    if hasattr(model, "norm"):
        return model.norm

    # 4. transformer_lens HookedTransformer
    if hasattr(model, "ln_final"):
        return model.ln_final
    
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f
        
    raise AttributeError(
        "Could not locate final norm layer automatically "
        "(checked: model.model.language_model.norm, model.model.norm, model.norm, model.ln_final)"
    )


def is_addition_dataset_path(path_like) -> bool:
    lowered = str(path_like).lower()
    return "plus" in lowered or "_add_" in lowered or lowered.startswith("add_") or "num2len" in lowered or "quanta" in lowered


# POSITION_SELECT validation
def _validate_position_select(pos_select):
    """Validate POSITION_SELECT parameter."""
    valid_strings = {'all', 'all_no_extra', 'extra'}
    if isinstance(pos_select, str):
        if pos_select not in valid_strings:
            raise ValueError(
                f"Invalid POSITION_SELECT value '{pos_select}'. "
                f"String values must be one of: {valid_strings}"
            )
    elif isinstance(pos_select, int):
        pass  # Single integer is valid
    elif isinstance(pos_select, list):
        if not all(isinstance(x, int) for x in pos_select):
            raise ValueError(
                f"When POSITION_SELECT is a list, all elements must be integers; got: {pos_select}"
            )
    else:
        raise ValueError(
            f"Invalid POSITION_SELECT type '{type(pos_select).__name__}'. "
            f"Must be 'all', 'all_no_extra', 'extra', a single int, or a list of ints"
        )

def compute_feature_from_flow(flow, feature_type):
    """
    Compute the requested feature from a flow array.
    flow: numpy array, shape (seq_len, feature_dim)
    """
    if feature_type == 'flows':
        return flow
    if feature_type == 'velocities':
        return np.diff(flow, axis=0)
    if feature_type == 'curvatures':
        return np.diff(np.diff(flow, axis=0), axis=0)
    raise ValueError(f"Unsupported feature type: {feature_type}")


def compute_raw_sum(question: str, sum_pos: int) -> int:
    """
    Compute the raw digit sum at a given sum position (sum of operand digits, no mod).
    
    Args:
        question: Expression string, e.g. "123 + 456 + 789"
        sum_pos: Sum position index (from most significant, 0 = MSB)
    
    Returns:
        int: Raw sum (>=0), or -1 if it cannot be computed
    
    Note:
        Sum position and operand position may differ. If the MSB has carry, the sum
        may have more digits than operands.
        operand_pos = sum_pos - (sum_digit_count - operand_digit_count)
    """
    try:
        # Parse operands
        operands = []
        for part in question.split('+'):
            part = part.replace('=', '').strip()
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
        
        # Digit offset (sum may be one digit longer)
        extra_digits = result_len - max_operand_len
        
        # Corresponding operand position
        operand_pos = sum_pos - extra_digits
        
        # operand_pos < 0: no operand digit at this sum position (carry-only MSB)
        if operand_pos < 0:
            # Carry-only digit; raw sum is 0 here
            return 0
        
        # Sum operand digits at this position
        digit_sum = 0
        for op in operands:
            op_str = str(op)
            op_len = len(op_str)
            right_idx = result_len - 1 - sum_pos  # index from the right
            op_digit_idx = op_len - 1 - right_idx  # index from the left in operand
            
            if 0 <= op_digit_idx < op_len:
                digit_sum += int(op_str[op_digit_idx])
            # Shorter operand contributes 0 at this position
        
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
        is_padded = False
        max_op_len = 0
        
        for part in question.split('+'):
            part = part.replace('=', '').strip()
            if part.isdigit():
                operands.append(int(part))
                if len(part) > len(str(int(part))):
                    is_padded = True
                max_op_len = max(max_op_len, len(part))
                
        if len(operands) < 2:
            return 0.0
            
        # Expected answer length and distance from current pos to ones place
        if is_padded:
            # Fixed-width (Quanta): answer length is max_op_len + 1
            ans_len = max_op_len + 1
        else:
            # Unpadded (qwen/llama): natural sum length
            ans_len = len(str(sum(operands)))
            
        right_digits = (ans_len - 1) - current_pos
        
        if right_digits <= 0:
            return 0.0
            
        # C_potential = sum(op_i % 10^R) / 10^R
        divisor = 10 ** right_digits
        lower_sum = sum(op % divisor for op in operands)
        c_potential = float(lower_sum) / divisor
        
        return c_potential
    except Exception:
        return 0.0

def compute_all_position_features(question: str):
    """
    Compute raw_sum and in_carry for all positions in an expression.
    Output order: most significant (left) to least significant (right).
    """
    try:
        # Parse operands
        operands = []
        for part in question.split('+'):
            part = part.replace('=', '').strip()
            if part.isdigit():
                operands.append(int(part))
        
        if not operands:
            return None, None
        
        # Sum and its length
        result = sum(operands)
        result_str = str(result)
        result_len = len(result_str)
        
        # Reversed operand strings for indexing from ones place (e.g. 123 -> "321")
        op_strs = [str(op)[::-1] for op in operands]
        
        raw_sums = []
        in_carries = []
        
        carry = 0  # Initial carry is 0
        
        # Traverse from ones place (index 0) to MSB
        for i in range(result_len):
            # Sum operand digits at this place
            digit_sum = 0
            for s in op_strs:
                if i < len(s):
                    digit_sum += int(s[i])
            
            # Input carry at this place
            in_carries.append(carry)
            
            # raw_sum (digit sum without carry)
            raw_sums.append(digit_sum % 10)
            
            # Total with carry; carry to next (more significant) place
            total = digit_sum + carry
            carry = total // 10
            
        # Lists are [ones, tens, hundreds...]; reverse to MSB -> LSB
        raw_sums.reverse()
        in_carries.reverse()
        
        features = []
        # Interleave: [raw_sum_0, in_carry_0, raw_sum_1, in_carry_1, ...]; index 0 = MSB
        for i in range(result_len):
            features.append(raw_sums[i])
            features.append(in_carries[i])
        
        info = {
            'result_len': result_len,
            'operands': operands,
            'result': result,
            'raw_sums': raw_sums,
            'in_carries': in_carries
        }
        
        return features, info
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def detect_max_result_len(file_path: str, filter_by_result_len=None):
    """
    Detect maximum sum digit count in the dataset (for feature dimension).
    
    Args:
        file_path: Data file path
    
    Returns:
        tuple: (target_result_len, sample_idx_to_result_len)
            - target_result_len: Target digit count (mode if FILTER_BY_RESULT_LEN else max)
            - sample_idx_to_result_len: sample_idx -> result_len
    """
    from utils.flow_utils import load_samples_meta
    from collections import Counter
    
    samples_meta = load_samples_meta(file_path)
    
    if not samples_meta:
        raise ValueError("Could not load samples_meta; ensure HDF5 file with samples group")
    
    sample_idx_to_result_len = {}
    
    # Detect result_len for all samples
    for sample_idx, meta in samples_meta.items():
        question = meta.get("question", "")
        _, info = compute_all_position_features(question)
        if info is not None:
            sample_idx_to_result_len[sample_idx] = info['result_len']
    
    if not sample_idx_to_result_len:
        raise ValueError("Could not detect result digit count from samples")
    
    result_len_list = list(sample_idx_to_result_len.values())
    result_len_counter = Counter(result_len_list)
    
    # Print result_len distribution
    print(f"  Result length distribution: {dict(result_len_counter)}")
    
    import sys
    this_mod = sys.modules[__name__]
    eff_filter_by_result_len = filter_by_result_len if filter_by_result_len is not None else getattr(this_mod, 'FILTER_BY_RESULT_LEN', True)
    
    if eff_filter_by_result_len:
        # Most frequent result_len
        target_result_len = result_len_counter.most_common(1)[0][0]
        target_count = result_len_counter[target_result_len]
        print(f"  Selected result_len={target_result_len} (most common: {target_count})")
    else:
        # Maximum result_len
        target_result_len = max(result_len_list)
        print(f"  Using max_result_len={target_result_len}")
    
    return target_result_len, sample_idx_to_result_len


def map_label_to_id(label_char: str) -> int:
    """
    Map label character ('0'-'9' or 'extra') to class ID (0-10).
    """
    if label_char == 'extra' or not label_char:
        return 10
    try:
        val = int(label_char)
        if 0 <= val <= 9:
            return val
    except (ValueError, TypeError):
        pass
    return 10


def is_hard_sample(pred_char: str, gt_char: str) -> bool:
    """
    Whether this is a hard sample: pred and gt digits differ by 1 (including 0/9 wrap).
    
    Args:
        pred_char: predicted digit character '0'-'9'
        gt_char: ground-truth digit character '0'-'9'
    
    Returns:
        bool: True if digits differ by 1 (including 0 adjacent to 9)
    """
    try:
        pred_num = int(pred_char)
        gt_num = int(gt_char)
        # Difference with 0/9 wrap-around
        diff = abs(pred_num - gt_num)
        return diff == 1 or (pred_num == 0 and gt_num == 9) or (pred_num == 9 and gt_num == 0)
    except (ValueError, TypeError):
        return False  # Not numeric; not a hard sample


def get_pred_offset_direction(pred_char: str, gt_char: str, allow_equal: bool = False) -> int:
    """
    Offset direction of pred vs gt (for pred_offset_direction training target).
    
    Args:
        pred_char: Predicted digit '0'-'9'
        gt_char: Ground-truth digit '0'-'9'
        allow_equal: Allow pred = gt (three-class mode)
    
    Returns:
        int:
        - 0: pred = gt - 1 (pred one below gt)
        - 1: pred = gt (only when allow_equal=True)
        - 2: pred = gt + 1 (pred one above gt)
        Special: 0 vs 9 -> -1 (return 0); 9 vs 0 -> +1 (return 2)
    
    Raises:
        ValueError: If diff is not in {-1, 0, 1} (with wrap) and allow_equal=False
    """
    try:
        pred_num = int(pred_char)
        gt_num = int(gt_char)
        
        # 0/9 wrap
        if pred_num == 0 and gt_num == 9:
            return 0  # 0-9 treated as -1
        elif pred_num == 9 and gt_num == 0:
            return 2  # 9-0 treated as +1 (three-class returns 2)
        
        # Normal case
        diff = pred_num - gt_num
        if diff == 0:
            if allow_equal:
                return 1  # pred = gt (equal class in three-class)
            else:
                raise ValueError(f"pred equals gt but allow_equal=False: pred={pred_num}, gt={gt_num}")
        elif diff == 1:
            return 2  # pred = gt + 1
        elif diff == -1:
            return 0  # pred = gt - 1
        else:
            if allow_equal:
                raise ValueError(f"pred-gt diff not in {{-1, 0, 1}}: pred={pred_num}, gt={gt_num}, diff={diff}")
            else:
                raise ValueError(f"pred-gt diff not 1: pred={pred_num}, gt={gt_num}, diff={diff}")
    except (ValueError, TypeError) as e:
        if isinstance(e, ValueError) and ("diff not" in str(e) or "equals gt" in str(e)):
            raise
        raise ValueError(f"Cannot compute offset direction: pred='{pred_char}', gt='{gt_char}'")


def load_and_process_data(file_path, position_select='all', feature_type='flows', 
                          pooling_type='none', use_pca=False, pca_dim=512, pca_mode='per_layer',
                          apply_carry_filter=None, train_target=None, sample_filter_mode=None,
                          apply_model_norm=False, model_path=None, raw_sum_mod_10=False,
                          filter_by_result_len=True, allowed_input_digits=None,
                          allowed_output_digits=None, allowed_incoming_carries=None,
                          allowed_target_input_digits=None, allowed_target_output_digits=None,
                          allowed_target_incoming_carries=None,
                          balance_mode='none', balance_target_classes=False,
                          seed=42,
                          model_type=None, has_h5py=True, target_pos='consistent',
                          input_pos='consistent',
                          filter_prefix_correct=None,
                          loaded_model_obj=None,
                          sign=None):
    """
    Load and preprocess data produced by generate.py.
    
    Auto-detects .pkl or .h5; prefers HDF5 when available.
    
    Args:
        file_path: data path (.pkl or .h5)
        position_select: TARGET_POS — drives main loop; labels/preds from this position
            (int, list, 'all', 'all_no_extra', 'extra', etc.)
        input_pos: INPUT_POS — source of activation features
            'consistent': same position as TARGET_POS (default)
            int: fixed position for activations (decoupled from TARGET_POS)
        feature_type: 'flows', 'velocities', 'curvatures'; *_both.h5 also supports post_attn/post_ffn
        pooling_type: 'none', 'avg', 'max'
        use_pca: whether to apply PCA
        pca_dim: PCA output dimension
        pca_mode: PCA ordering mode
        train_target: training target name
        sample_filter_mode: TARGET_POS filter ('correct', 'incorrect', 'all')
        allowed_input_digits / allowed_output_digits / allowed_incoming_carries:
            INPUT_POS-side filters; None disables
        allowed_target_*: TARGET_POS-side filters; None disables
        filter_prefix_correct:
            'prefix_correct': all positions before TARGET_POS correct
            'prefix_incorrect': at least one prefix position wrong
            None: no prefix filter
        sign: Optional arithmetic sign override. When set to 'plus', raw-sum and
            carry-potential targets are treated as addition targets without
            relying on the file name.
    
    Returns:
        X_all, y_all, y_binary, position_indices, selected_positions,
        seq_len, feature_dim, is_regression_task, sample_idx_all, meta
    """
    # Use passed-in parameters
    effective_train_target = train_target
    effective_sample_filter_mode = sample_filter_mode
    effective_apply_model_norm = apply_model_norm
    effective_model_path = model_path
    effective_raw_sum_mod_10 = raw_sum_mod_10
    
    FILTER_BY_RESULT_LEN = filter_by_result_len
    ALLOWED_INPUT_DIGITS = allowed_input_digits
    ALLOWED_OUTPUT_DIGITS = allowed_output_digits
    ALLOWED_INCOMING_CARRIES = allowed_incoming_carries
    ALLOWED_TARGET_INPUT_DIGITS = allowed_target_input_digits
    ALLOWED_TARGET_OUTPUT_DIGITS = allowed_target_output_digits
    ALLOWED_TARGET_INCOMING_CARRIES = allowed_target_incoming_carries
    SEED = seed
    MODEL_TYPE = model_type
    HAS_H5PY = has_h5py

    file_path = Path(file_path)
    feature_cfg = parse_feature_type_config(feature_type)
    raw_flow_feature_type = feature_cfg["feature_kind"]
    requested_flow_dataset_key = feature_cfg["dataset_key"]

    # Load results (.h5/.pkl); position_select is TARGET_POS
    all_token_results = load_all_token_results(
        file_path, position_select, feature_type=feature_type
    )
    
    # Extended select keeps sample_idx_list aligned; labels/preds/carries from TARGET_POS
    (
        selected_positions,
        labels_list,
        position_idx_list,
        flows_list,          # Used directly when input_pos=='consistent'; else overridden
        preds_list,
        gt_chars_list,
        sample_idx_list,  # For question metadata lookup
        incoming_carries_list,
        outgoing_carries_list,
    ) = select_positions_extended(
        all_token_results, position_select
    )
    
    # ========== INPUT_POS handling ==========
    # When input_pos != 'consistent', activations come from fixed input_pos, not TARGET_POS.
    # Pre-build sample_idx -> flow map.
    effective_input_pos = input_pos
    input_pos_flow_map = {}  # sample_idx -> flow (ndarray)
    
    if effective_input_pos != 'consistent':
        input_pos_val = effective_input_pos  # must be int
        if not isinstance(input_pos_val, (int,)):
            raise ValueError(
                f"When INPUT_POS != 'consistent', INPUT_POS must be a single int; got '{input_pos_val}'"
            )
        # Load input_pos data (may already be in all_token_results)
        if input_pos_val not in all_token_results:
            positions_to_load_inp = list(set(list(all_token_results.keys()) + [input_pos_val]))
            input_pos_results = load_all_token_results(
                file_path, positions_to_load_inp, feature_type=feature_type, verbose=False
            )
        else:
            input_pos_results = all_token_results
        
        if input_pos_val in input_pos_results:
            inp_data = input_pos_results[input_pos_val]
            inp_flows_raw = inp_data.get("flows", [])
            inp_indices = inp_data.get("sample_indices")
            if inp_indices is None:
                inp_indices = list(range(len(inp_flows_raw)))
            for s_idx, fl in zip(inp_indices, inp_flows_raw):
                input_pos_flow_map[int(s_idx)] = fl
            print(f"  INPUT_POS={input_pos_val}: loaded activations for {len(input_pos_flow_map)} samples")
        else:
            raise ValueError(f"INPUT_POS={input_pos_val} not found in data file")
    
    # -------------------------------------------------------------------------
    # Optional APPLY_MODEL_NORM: load model and norm hidden states (INPUT_POS side only)
    # Most meaningful for raw flows; velocities/curvatures can be normed too
    # -------------------------------------------------------------------------
    if effective_apply_model_norm and raw_flow_feature_type == 'flows':
        from transformers import AutoModelForCausalLM
        
        model_obj = loaded_model_obj
        if model_obj is not None:
            print(f"APPLY_MODEL_NORM=True: using attached preloaded model for final norm...")
        else:
            print(f"APPLY_MODEL_NORM=True: loading model and applying final norm...")
            # 1. Try loading model
            try:
                print(f"  Loading model weights: {effective_model_path}")
                model_obj = AutoModelForCausalLM.from_pretrained(
                    effective_model_path, 
                    trust_remote_code=True, 
                    device_map="auto", 
                    dtype="auto",
                )
                model_obj.eval()
            except Exception as e:
                print(f"  [Error] Model load failed: {e}")
                print("  Skipping norm; using raw hidden states.")
        
        # 2. Apply norm (INPUT_POS side only)
        if model_obj is not None:
            try:
                # Get final norm layer (see logit_lens_analysis.py)
                try:
                    norm_layer = get_model_norm(model_obj)
                except AttributeError as e:
                     raise ValueError(f"Could not find model final norm layer: {e}")
                # norm_layer should be on device
                device = getattr(model_obj, "device", None)
                if device is None:
                    try:
                        device = next(model_obj.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")
                device = torch.device(device)
                
                if hasattr(norm_layer, "weight") and norm_layer.weight is not None:
                    target_dtype = norm_layer.weight.dtype
                elif hasattr(norm_layer, "w") and norm_layer.w is not None:
                    target_dtype = norm_layer.w.dtype
                else:
                    try:
                        target_dtype = next(norm_layer.parameters()).dtype
                    except StopIteration:
                        target_dtype = torch.float32

                def _norm_flow(flow_val):
                    if isinstance(flow_val, np.ndarray):
                        flow_tensor = torch.from_numpy(flow_val)
                    else:
                        flow_tensor = flow_val
                    flow_tensor = flow_tensor.to(device=device, dtype=target_dtype)
                    with torch.no_grad():
                        normed_flow = norm_layer(flow_tensor)
                    return normed_flow.float().cpu().numpy()

                if effective_input_pos == 'consistent':
                    print(f"  Applying final norm to {len(flows_list)} samples (INPUT_POS=consistent)...")
                    flows_list = [_norm_flow(flow) for flow in flows_list]
                    print(f"  Final norm done. INPUT_POS=consistent: {len(flows_list)} samples")
                else:
                    print(f"  Applying final norm to {len(input_pos_flow_map)} samples (INPUT_POS={effective_input_pos})...")
                    for s_idx in list(input_pos_flow_map.keys()):
                        input_pos_flow_map[s_idx] = _norm_flow(input_pos_flow_map[s_idx])
                    print(f"  Final norm done. INPUT_POS={effective_input_pos}: {len(input_pos_flow_map)} samples")
                
                # Release GPU memory if we loaded the model here
                if loaded_model_obj is None:
                    del model_obj
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  [Error] Norm application failed: {e}")
                print("  Continuing with raw hidden states.")
    
    # Dataset type (add/mul) and task type (classification/regression)
    is_addition_dataset = str(sign).lower() in {"plus", "add", "addition", "+"} if sign is not None else is_addition_dataset_path(file_path)
    # Carry filter from apply_carry_filter or train_target default
    if apply_carry_filter is None:
        is_carry_target = effective_train_target in ['incoming_carry', 'outgoing_carry']
    else:
        is_carry_target = apply_carry_filter
    # Regression task flag
    if effective_train_target in ['C_potential', 'raw_sum_regress']:
        is_regression_task = True
    else:
        is_regression_task = is_carry_target and not is_addition_dataset  # Mul carry -> regression
    
    # raw_sum targets only for addition datasets
    if effective_train_target in ['raw_sum_classify', 'raw_sum_regress', 'C_potential'] and not is_addition_dataset:
        raise ValueError(f"TRAIN_TARGET = '{effective_train_target}' only supports addition datasets; pass sign='plus' or use an addition HDF5 file.")
    
    # Load samples_meta when raw_sum or text-based filters need question strings
    samples_meta = None
    if (effective_train_target in ['raw_sum_classify', 'raw_sum_regress', 'C_potential'] or 
        ALLOWED_INPUT_DIGITS is not None or ALLOWED_OUTPUT_DIGITS is not None or FILTER_BY_RESULT_LEN or
        ALLOWED_TARGET_INPUT_DIGITS is not None):
        samples_meta = load_samples_meta(file_path)
        if not samples_meta:
            raise ValueError("samples_meta required for filters or targets; use HDF5 with samples group")
    
    # Prepare result-length filtering
    target_result_len = -1
    sample_idx_to_result_len = {}
    if FILTER_BY_RESULT_LEN:
        from collections import Counter
        sample_idx_to_result_len = {}
        for sample_idx, meta in samples_meta.items():
            question = meta.get("question", "")
            _, info = compute_all_position_features(question)
            if info is not None:
                sample_idx_to_result_len[sample_idx] = info['result_len']
        
        if sample_idx_to_result_len:
            result_len_list = list(sample_idx_to_result_len.values())
            result_len_counter = Counter(result_len_list)
            target_result_len = result_len_counter.most_common(1)[0][0]
            target_result_len = result_len_counter.most_common(1)[0][0]
            # Logging deferred to end of function
    
    # Discover all numeric positions from file
    file_path_obj = Path(file_path)
    if file_path_obj.suffix == "":
        h5_path = file_path_obj.with_suffix(".h5")
        pkl_path = file_path_obj.with_suffix(".pkl")
        if h5_path.exists():
            file_path_obj = h5_path
        elif pkl_path.exists():
            file_path_obj = pkl_path
    
    all_numeric_positions = []
    if file_path_obj.suffix == ".h5" and HAS_H5PY:
        # Read all positions from HDF5
        try:
            import h5py
            with h5py.File(file_path_obj, "r") as hf:
                positions_group = hf.get("all_token_results")
                if positions_group:
                    # From attrs or keys
                    numeric_positions = list(positions_group.attrs.get("numeric_positions", []))
                    if not numeric_positions:
                        # Infer from group keys
                        for key in positions_group.keys():
                            if key.startswith("pos_"):
                                suffix = key[4:]
                                if suffix.lstrip("-").isdigit():
                                    numeric_positions.append(int(suffix))
                    numeric_positions = [int(p) for p in numeric_positions]
                    all_numeric_positions = sorted(numeric_positions)
        except Exception:
            # Fallback to loaded positions
            all_numeric_positions = sorted([p for p in all_token_results.keys() if isinstance(p, int)])
    else:
        # Pickle or HDF5 read failure: use loaded positions
        all_numeric_positions = sorted([p for p in all_token_results.keys() if isinstance(p, int)])
    
    # ========== TARGET_POS / INPUT_POS semantics ==========
    # Main loop driven by TARGET_POS (position_select); labels/preds/carries from there.
    # Activations from INPUT_POS:
    #   - 'consistent': same as TARGET_POS (flows_list)
    #   - int: fixed position (input_pos_flow_map)
    # effective_target_pos='consistent' for backward-compatible prefix_correct_map
    effective_target_pos = 'consistent'


    # ========== Prefix correctness map ==========
    # When filter_prefix_correct or train_target=='prefix_is_correct', load prior positions
    # prefix_correct_map: {pos -> {sample_idx -> bool}}
    prefix_correct_map = {}  # pos -> {sample_idx -> is_correct}
    if filter_prefix_correct is not None or effective_train_target == 'prefix_is_correct':
        # Prefix positions relative to TARGET_POS
        base_positions_for_prefix = position_select if effective_target_pos == 'consistent' else effective_target_pos
        
        # Positions to load for prefix check
        prefix_positions_needed = set()
        if isinstance(base_positions_for_prefix, int):
            for p in range(0, base_positions_for_prefix):
                prefix_positions_needed.add(p)
        elif isinstance(base_positions_for_prefix, list):
            for pos_val in base_positions_for_prefix:
                if isinstance(pos_val, int):
                    for p in range(0, pos_val):
                        prefix_positions_needed.add(p)
        # For 'all', 'all_no_extra', 'extra', etc., prefix checked per tgt_entry["pos"] in main loop
        elif isinstance(base_positions_for_prefix, str) and base_positions_for_prefix not in ('extra',):
            # All numeric positions as candidate prefix positions
            for p in all_numeric_positions:
                prefix_positions_needed.add(p)

        if prefix_positions_needed:
            # Load prefix position data (may already be in all_token_results)
            missing_prefix_positions = [p for p in prefix_positions_needed if p not in all_token_results]
            if missing_prefix_positions:
                prefix_load_positions = list(set(list(all_token_results.keys()) + missing_prefix_positions))
                prefix_token_results = load_all_token_results(
                    file_path, prefix_load_positions, feature_type=feature_type, verbose=False
                )
            else:
                prefix_token_results = all_token_results

            # Build map
            for p in prefix_positions_needed:
                if p in prefix_token_results:
                    p_data = prefix_token_results[p]
                    p_labels = p_data.get("labels", [])
                    p_indices = p_data.get("sample_indices")
                    if p_indices is None:
                        p_indices = list(range(len(p_labels)))
                    p_map = {}
                    for s_idx, lbl in zip(p_indices, p_labels):
                        p_map[int(s_idx)] = bool(lbl)
                    prefix_correct_map[p] = p_map

            if filter_prefix_correct is not None:
                print(f"  Prefix filter: mode='{filter_prefix_correct}', loaded labels for {len(prefix_correct_map)} prefix positions")
            else:
                print(f"  Prefix labels (prefix_is_correct): loaded {len(prefix_correct_map)} prefix positions")
    
    # Unified source/target entries to avoid cross-coupled filter and label logic.
    def get_filter_input_digit(question: str, pos: Any) -> int:
        """
        Input digit for position-based filtering.

        Legacy rule: for numeric pos, return result_str[pos - 1]; else -1.
        """
        if not isinstance(pos, (int, np.integer)) or int(pos) < 1:
            return -1
        operands = [int(p.strip()) for p in question.split('+') if p.strip().isdigit()]
        if not operands:
            return -1
        result_str = str(sum(operands))
        idx = int(pos) - 1
        if 0 <= idx < len(result_str):
            return int(result_str[idx])
        return -1

    def map_output_char_to_filter_value(output_char: Any) -> int:
        """Map output char to filter value: digit -> 0-9, extra -> 10, else -1."""
        if str(output_char).isdigit():
            return int(output_char)
        if output_char == "extra":
            return 10
        return -1

    def build_target_entry(idx: int, original_pos: Any, label: Any) -> Dict[str, Any]:
        """Build TARGET_POS-side sample view; labels/preds/carries from current position."""
        return {
            "pos": original_pos,
            "pred": preds_list[idx] if preds_list is not None else None,
            "gt": gt_chars_list[idx] if gt_chars_list is not None else None,
            "is_correct": bool(label),
            "incoming_carry": incoming_carries_list[idx] if incoming_carries_list is not None else -1,
            "outgoing_carry": outgoing_carries_list[idx] if outgoing_carries_list is not None else -1,
        }

    def get_input_flow(idx: int, sample_idx_cur: int) -> Optional[Any]:
        """Return activation flow; None if missing at INPUT_POS (skip sample)."""
        if effective_input_pos == 'consistent':
            return flows_list[idx]
        return input_pos_flow_map.get(sample_idx_cur)
    
    
    # Positions to exclude
    positions_to_exclude = set()
    if is_carry_target:
        positions_to_exclude.add('extra')  # Always exclude extra
        if effective_train_target == 'incoming_carry':
            # Exclude last digit pos (incoming carry always 0)
            if all_numeric_positions:
                positions_to_exclude.add(all_numeric_positions[-1])
        elif effective_train_target == 'outgoing_carry':
            # Exclude pos=0 (outgoing carry always 0)
            if 0 in all_numeric_positions:
                positions_to_exclude.add(0)
    
    # Collect tensors
    X_list = []
    y_list = []
    y_binary_list = []  # Always store correct/incorrect
    position_idx_list_out = []
    sample_idx_list_out = []  # Sample indices
    
    # --- META ---
    preds_list_out = []
    gts_list_out = []
    incoming_carries_out = []
    outgoing_carries_out = []
    input_tokens_out = []
    
    # Position index -> original position key
    pos_to_original_pos = {i: pos for i, pos in enumerate(selected_positions)}
    
    # Filter drop counters
    drop_counts = {
        'result_len': 0,
        'prefix_correct': 0,
        'src_input_digits': 0,
        'src_output_digits': 0,
        'src_incoming_carry': 0,
        'tgt_input_digits': 0,
        'tgt_output_digits': 0,
        'tgt_incoming_carry': 0,
        'prefix_is_correct_pos0': 0,  # Dropped pos=0 samples with no prefix (prefix_is_correct)
    }
    total_reached_additional_filters = 0

    for idx, (flow, label, pos_idx) in enumerate(zip(flows_list, labels_list, position_idx_list)):
        original_pos = pos_to_original_pos[pos_idx]
        
        # Skip excluded positions
        if original_pos in positions_to_exclude:
            continue
        
        # Count before additional filters
        total_reached_additional_filters += 1

        sample_idx_cur = sample_idx_list[idx] if sample_idx_list is not None else -1
        tgt_entry = build_target_entry(idx, original_pos, label)
        # Activation (possibly from fixed INPUT_POS)
        actual_flow = get_input_flow(idx, sample_idx_cur)
        if actual_flow is None:
            continue  # No activation at INPUT_POS for this sample

        # Load question only when needed
        question = ""
        if samples_meta and sample_idx_cur in samples_meta:
            question = samples_meta[sample_idx_cur].get("question", "")

        # ========== Global / sample-level filters ==========
        if FILTER_BY_RESULT_LEN:
            if sample_idx_to_result_len.get(sample_idx_cur, -1) != target_result_len:
                drop_counts['result_len'] += 1
                continue

        # ========== Prefix correctness filter ==========
        if filter_prefix_correct is not None and isinstance(tgt_entry["pos"], (int, np.integer)):
            prefix_all_correct = True
            for p in range(0, int(tgt_entry["pos"])):
                p_map = prefix_correct_map.get(p)
                if p_map is None:
                    # Missing prefix position data; treat as not satisfied
                    prefix_all_correct = False
                    break
                if not p_map.get(sample_idx_cur, False):
                    prefix_all_correct = False
                    break
            
            if filter_prefix_correct == 'prefix_correct' and not prefix_all_correct:
                drop_counts['prefix_correct'] += 1
                continue
            elif filter_prefix_correct == 'prefix_incorrect' and prefix_all_correct:
                drop_counts['prefix_correct'] += 1
                continue

        # ========== Source / INPUT_POS filters ==========
        # When INPUT_POS='consistent', equivalent to target filters.
        # Fixed INPUT_POS still filters on TARGET-side fields for consistency.
        if ALLOWED_INPUT_DIGITS is not None:
            src_input_digit = get_filter_input_digit(question, tgt_entry["pos"])
            if src_input_digit not in ALLOWED_INPUT_DIGITS:
                drop_counts['src_input_digits'] += 1
                continue

        if ALLOWED_OUTPUT_DIGITS is not None:
            src_output_digit = map_output_char_to_filter_value(tgt_entry.get("pred"))
            if src_output_digit not in ALLOWED_OUTPUT_DIGITS:
                drop_counts['src_output_digits'] += 1
                continue

        if ALLOWED_INCOMING_CARRIES is not None:
            if tgt_entry.get("incoming_carry", -1) not in ALLOWED_INCOMING_CARRIES:
                drop_counts['src_incoming_carry'] += 1
                continue

        # ========== Target / TARGET_POS filters ==========
        if ALLOWED_TARGET_INPUT_DIGITS is not None:
            tgt_input_digit = get_filter_input_digit(question, tgt_entry["pos"])
            if tgt_input_digit not in ALLOWED_TARGET_INPUT_DIGITS:
                drop_counts['tgt_input_digits'] += 1
                continue

        if ALLOWED_TARGET_OUTPUT_DIGITS is not None:
            tgt_output_digit = map_output_char_to_filter_value(tgt_entry.get("pred"))
            if tgt_output_digit not in ALLOWED_TARGET_OUTPUT_DIGITS:
                drop_counts['tgt_output_digits'] += 1
                continue

        if ALLOWED_TARGET_INCOMING_CARRIES is not None:
            if tgt_entry.get("incoming_carry", -1) not in ALLOWED_TARGET_INCOMING_CARRIES:
                drop_counts['tgt_incoming_carry'] += 1
                continue

        if effective_train_target in ['pred', 'gt', 'incoming_carry', 'outgoing_carry', 'raw_sum_classify', 'raw_sum_regress', 'C_potential', 'pos']:
            # SAMPLE_FILTER_MODE on target side; with TARGET_POS='consistent', uses current position
            tgt_is_correct = tgt_entry.get('is_correct')
            if tgt_is_correct is None:
                continue
            if effective_sample_filter_mode == 'correct':
                if not tgt_is_correct:
                    continue
            elif effective_sample_filter_mode == 'incorrect':
                if tgt_is_correct:
                    continue
            # 'all': no filter

        # Target position for raw_sum / C_potential etc.
        label_pos = tgt_entry["pos"]
        
        # Label for train_target
        if effective_train_target == 'pred':
            label_val = map_label_to_id(tgt_entry.get('pred', ''))
        elif effective_train_target == 'gt':
            label_val = map_label_to_id(tgt_entry.get('gt', ''))
        elif effective_train_target == 'incoming_carry':
            carry_val = tgt_entry.get('incoming_carry', -1)
            if carry_val == -1:
                continue
            label_val = carry_val
        elif effective_train_target == 'outgoing_carry':
            carry_val = tgt_entry.get('outgoing_carry', -1)
            if carry_val == -1:
                continue
            label_val = carry_val
        elif effective_train_target == 'pred_offset_direction':
            pred_char = tgt_entry.get('pred')
            gt_char = tgt_entry.get('gt')
            tgt_is_correct = bool(tgt_entry.get('is_correct', False))
            
            if effective_sample_filter_mode == 'correct':
                raise ValueError("TRAIN_TARGET = 'pred_offset_direction': SAMPLE_FILTER_MODE cannot be 'correct'")
            elif effective_sample_filter_mode == 'incorrect':
                if tgt_is_correct:
                    continue
                if not is_hard_sample(pred_char, gt_char):
                    continue
                try:
                    offset = get_pred_offset_direction(pred_char, gt_char, allow_equal=False)
                    label_val = 0 if offset == 0 else 1
                except ValueError:
                    continue
            elif effective_sample_filter_mode == 'all':
                is_hard = is_hard_sample(pred_char, gt_char)
                if not (tgt_is_correct or is_hard):
                    continue
                try:
                    if tgt_is_correct:
                        label_val = 1
                    else:
                        offset = get_pred_offset_direction(pred_char, gt_char, allow_equal=False)
                        label_val = offset
                except ValueError:
                    continue
            else:
                raise ValueError(f"TRAIN_TARGET = 'pred_offset_direction': SAMPLE_FILTER_MODE must be 'incorrect' or 'all'; got '{effective_sample_filter_mode}'")
        elif effective_train_target in ['raw_sum_classify', 'raw_sum_regress']:
            if sample_idx_list is None or samples_meta is None:
                raise ValueError("raw_sum requires sample_idx_list and samples_meta")
            sample_idx = sample_idx_list[idx]
            sample_info = samples_meta.get(sample_idx)
            if sample_info is None:
                continue
            question = sample_info.get("question", "")
            
            # Local digit sum at label_pos
            raw_sum_val = compute_raw_sum(question, label_pos)
            if raw_sum_val == -1:
                continue
            
            if effective_raw_sum_mod_10:
                raw_sum_val = raw_sum_val % 10
            
            label_val = raw_sum_val
        elif effective_train_target == 'C_potential':
            if sample_idx_list is None or samples_meta is None:
                raise ValueError("C_potential requires sample_idx_list and samples_meta")
            sample_info = samples_meta.get(sample_idx_cur)
            if sample_info is None:
                continue
            question = sample_info.get("question", "")
            
            # C_potential at label_pos
            try:
                c_potential_val = compute_c_potential(question, label_pos)
            except Exception:
                continue
                
            label_val = c_potential_val
        elif effective_train_target == 'pos':
            if not isinstance(original_pos, int):
                continue
            label_val = int(original_pos)
        elif effective_train_target == 'is_correct':
            label_val = 1 if tgt_entry.get('is_correct', False) else 0
        elif effective_train_target == 'prefix_is_correct':
            # Whether all prefix positions before TARGET_POS are correct (same as filter_prefix_correct)
            if isinstance(tgt_entry["pos"], (int, np.integer)):
                if int(tgt_entry["pos"]) == 0:
                    # pos=0 has no prefix; skip
                    drop_counts['prefix_is_correct_pos0'] += 1
                    continue
                
                prefix_all_correct = True
                for p in range(0, int(tgt_entry["pos"])):
                    p_map = prefix_correct_map.get(p)
                    if p_map is None:
                        prefix_all_correct = False
                        break
                    if not p_map.get(sample_idx_cur, False):
                        prefix_all_correct = False
                        break
                label_val = 1 if prefix_all_correct else 0
            else:
                # Non-integer TARGET_POS (e.g. extra); skip
                continue
        else:
            label_val = 1 if tgt_entry.get('is_correct', False) else 0
            
        # y_binary always reflects target-side correctness (for BALANCE_MODE)
        y_binary_val = 1 if tgt_entry.get('is_correct', False) else 0

        feat = compute_feature_from_flow(actual_flow, raw_flow_feature_type)
        # feat shape: (seq_len, dim) or (seq_len-1/-2, dim) for derivatives
        
        # Pooling
        if pooling_type == 'avg':
            # Average over seq_len
            feat_processed = feat.mean(axis=0)  # shape: (feature_dim,)
        elif pooling_type == 'max':
            # Max over seq_len
            feat_processed = feat.max(axis=0)  # shape: (feature_dim,)
        else:
            # No pooling: flatten
            feat_processed = feat.flatten()  # shape: (seq_len * feature_dim,)
        
        X_list.append(feat_processed)
        y_list.append(label_val)
        y_binary_list.append(y_binary_val)
        position_idx_list_out.append(pos_idx)
        if sample_idx_list is not None:
             sample_idx_list_out.append(sample_idx_list[idx])
        else:
             sample_idx_list_out.append(-1)
        
        # --- META ---
        # META keeps target-side fields for alignment
        pred_char = tgt_entry.get("pred")
        gt_char = tgt_entry.get("gt")
        inc_carry = tgt_entry.get("incoming_carry", -1)
        out_carry = tgt_entry.get("outgoing_carry", -1)
        preds_list_out.append(pred_char)
        gts_list_out.append(gt_char)
        incoming_carries_out.append(inc_carry)
        outgoing_carries_out.append(out_carry)
        
        # Target-position input digit for external filter analysis
        input_tokens_out.append(str(get_filter_input_digit(question, tgt_entry["pos"])))    
    # To tensors
    if len(X_list) == 0:
        # Detailed error when no samples remain
        error_msg = "No samples left after filtering."
        error_msg += f"\n  Train target: {effective_train_target}"
        if is_carry_target:
            # Sort numeric then string excluded positions
            numeric_excluded = sorted([p for p in positions_to_exclude if isinstance(p, int)])
            string_excluded = sorted([p for p in positions_to_exclude if isinstance(p, str)])
            excluded_list = numeric_excluded + string_excluded
            error_msg += f"\n  Excluded positions: {excluded_list}"
            if incoming_carries_list is not None:
                valid_carries = [c for c in incoming_carries_list if c != -1]
                error_msg += f"\n  Incoming carry: total {len(incoming_carries_list)}, valid {len(valid_carries)}"
            if outgoing_carries_list is not None:
                valid_carries = [c for c in outgoing_carries_list if c != -1]
                error_msg += f"\n  Outgoing carry: total {len(outgoing_carries_list)}, valid {len(valid_carries)}"
            if incoming_carries_list is not None and all(c == -1 for c in incoming_carries_list):
                error_msg += "\n  Warning: all incoming carries are -1; carry data may be missing from file."
            if outgoing_carries_list is not None and all(c == -1 for c in outgoing_carries_list):
                error_msg += "\n  Warning: all outgoing carries are -1; carry data may be missing from file."
        elif effective_train_target == 'pred_offset_direction':
            if preds_list is not None and gt_chars_list is not None:
                total_samples = len(preds_list)
                error_samples = sum(1 for l in labels_list if not l)
                correct_samples = sum(1 for l in labels_list if l)
                hard_samples = sum(1 for p, g in zip(preds_list, gt_chars_list) if is_hard_sample(p, g))
            else:
                total_samples = 0
                error_samples = 0
                correct_samples = 0
                hard_samples = 0

            if total_samples > 0:
                error_msg += f"\n  Total samples: {total_samples}"
                error_msg += f"\n  Correct predictions: {correct_samples}"
                error_msg += f"\n  Wrong predictions: {error_samples}"
                error_msg += f"\n  Off-by-one samples: {hard_samples}"
                if effective_sample_filter_mode == 'incorrect':
                    error_msg += f"\n  Hint: need wrong prediction and off-by-one (binary mode)"
                elif effective_sample_filter_mode == 'all':
                    error_msg += f"\n  Hint: need correct or off-by-one (three-class mode)"
        raise ValueError(error_msg)
    
    X_all = torch.tensor(np.stack(X_list), dtype=torch.float32)
    # Regression: float32; classification: long
    y_all = torch.tensor(y_list, dtype=torch.float32 if is_regression_task else torch.long)
    y_binary = torch.tensor(y_binary_list, dtype=torch.long)
    position_indices = torch.tensor(position_idx_list_out, dtype=torch.long)
    sample_idx_all = torch.tensor(sample_idx_list_out, dtype=torch.long)
    
    # Infer seq_len and feature_dim from first flow / derived feature shape
    sample_flow = flows_list[0]
    sample_feat = compute_feature_from_flow(sample_flow, raw_flow_feature_type)
    original_seq_len = sample_feat.shape[0]
    feature_dim = sample_feat.shape[1]
    
    # Effective seq_len after pooling
    if pooling_type in ['avg', 'max']:
        seq_len = 1  # After pooling, seq_len is 1
    else:
        seq_len = original_seq_len
    
    print(f"\n{'='*50}")
    
    # Section 1: configuration
    print("")
    print(f"[Configuration]")
    print(f"  Train target: {effective_train_target}")
    if effective_train_target in ['pred', 'gt', 'incoming_carry', 'outgoing_carry']:
        if effective_sample_filter_mode == 'correct':
            print(f"  [Correct-only samples enabled]")
        elif effective_sample_filter_mode == 'incorrect':
            print(f"  [Incorrect-only samples enabled]")
    elif effective_train_target == 'pred_offset_direction':
        if effective_sample_filter_mode == 'incorrect':
            print(f"  [Binary mode: wrong and off-by-one only]")
        elif effective_sample_filter_mode == 'all':
            print(f"  [Three-class mode: correct or off-by-one]")
    print(f"  Dataset type: {'addition' if is_addition_dataset else 'multiplication'}")
    print(f"  Task type: {'regression' if is_regression_task else 'classification'}")
    print(f"  Flow dataset: {requested_flow_dataset_key}")
    print(f"  Flow derived feature: {raw_flow_feature_type}")
    
    print("")
    print(f"[Filter]")
    
    if is_carry_target:
        numeric_excluded = sorted([p for p in positions_to_exclude if isinstance(p, int)])
        string_excluded = sorted([p for p in positions_to_exclude if isinstance(p, str)])
        print(f"  [Excluded positions] {numeric_excluded + string_excluded}")
        
    print(f"  Samples before extra filters: {total_reached_additional_filters}")
    if (
        ALLOWED_INPUT_DIGITS is not None or ALLOWED_OUTPUT_DIGITS is not None or ALLOWED_INCOMING_CARRIES is not None or
        ALLOWED_TARGET_INPUT_DIGITS is not None or ALLOWED_TARGET_OUTPUT_DIGITS is not None or ALLOWED_TARGET_INCOMING_CARRIES is not None or
        FILTER_BY_RESULT_LEN
    ):
        current_count = total_reached_additional_filters
        if FILTER_BY_RESULT_LEN:
            current_count -= drop_counts['result_len']
            print(f"  [Result length filter]")
            if 'result_len_counter' in locals():
                print(f"    - Length distribution: {dict(result_len_counter)}")
            print(f"    - Dropped {drop_counts['result_len']}, kept {current_count}")
        if ALLOWED_INPUT_DIGITS is not None:
            current_count -= drop_counts['src_input_digits']
            print(f"  [Source input digit filter]")
            print(f"    - Dropped {drop_counts['src_input_digits']}, kept {current_count}")
        if ALLOWED_OUTPUT_DIGITS is not None:
            current_count -= drop_counts['src_output_digits']
            print(f"  [Source output digit filter]")
            print(f"    - Dropped {drop_counts['src_output_digits']}, kept {current_count}")
        if ALLOWED_INCOMING_CARRIES is not None:
            current_count -= drop_counts['src_incoming_carry']
            print(f"  [Source incoming carry filter]")
            print(f"    - Dropped {drop_counts['src_incoming_carry']}, kept {current_count}")
        if ALLOWED_TARGET_INPUT_DIGITS is not None:
            current_count -= drop_counts['tgt_input_digits']
            print(f"  [Target input digit filter]")
            print(f"    - Dropped {drop_counts['tgt_input_digits']}, kept {current_count}")
        if ALLOWED_TARGET_OUTPUT_DIGITS is not None:
            current_count -= drop_counts['tgt_output_digits']
            print(f"  [Target output digit filter]")
            print(f"    - Dropped {drop_counts['tgt_output_digits']}, kept {current_count}")
        if ALLOWED_TARGET_INCOMING_CARRIES is not None:
            current_count -= drop_counts['tgt_incoming_carry']
            print(f"  [Target incoming carry filter]")
            print(f"  Target incoming carry dropped: {drop_counts['tgt_incoming_carry']}")
            
        print(f"  --------------------------")
        total_dropped = sum(drop_counts.values())
        print(f"  Total extra dropped: {total_dropped}, kept {len(X_all)}")
    else:
        total_dropped = sum(drop_counts.values())
        if total_dropped > 0:
            print(f"  Total extra dropped: {total_dropped}, kept {len(X_all)}")
        else:
            print(f"  (no extra filters)")

    if drop_counts.get('prefix_is_correct_pos0', 0) > 0:
        print(f"  {'-'*40}")
        print(f"  Warning: dropped {drop_counts['prefix_is_correct_pos0']} samples with TARGET_POS=0 (no prefix for prefix_is_correct).")
        print(f"  {'-'*40}")
    else:
        print(f"  (no extra filter drops)")
    
    print("")
    print(f"[Final stats]")
    
    print(f"  is_correct distribution: positive {sum(y_binary == 1).item()}, negative {sum(y_binary == 0).item()}")
    
    if effective_train_target != 'is_correct':
        if is_regression_task:
            # Regression summary
            y_np = y_all.numpy()
            print(f"  Carry value stats:")
            print(f"    min: {y_np.min():.2f}, max: {y_np.max():.2f}, mean: {y_np.mean():.2f}, std: {y_np.std():.2f}")
            unique, counts = np.unique(y_np, return_counts=True)
            print(f"  Carry value distribution:")
            for u, c in zip(unique[:20], counts[:20]):
                print(f"    value {u:.2f}: {c} samples")
            if len(unique) > 20:
                print(f"    ... ({len(unique)} distinct values)")
        else:
            unique, counts = torch.unique(y_all, return_counts=True)
            print(f"  Target class distribution:")
            for u, c in zip(unique, counts):
                print(f"    class {u.item()}: {c.item()} samples")
    
    print(f"  X shape: {X_all.shape}")
    print(f"  y shape: {y_all.shape}")
    print(f"  Original seq length: {original_seq_len}")
    print(f"  Pooling type: {pooling_type}")
    print(f"  Processed seq length (seq_len): {seq_len}")
    print(f"  Feature dim (feature_dim): {feature_dim}")
    print(f"  Feature type: {feature_type}")
    
    if use_pca:
        effective_pca_mode = pca_mode
        if pca_mode == 'per_layer' and pooling_type != 'none':
            print("Note: pooling merges layers; per-layer PCA unavailable, using concat_first.")
            effective_pca_mode = 'concat_first'
        
        if effective_pca_mode == 'per_layer':
            print(f"\nPer-layer PCA to {pca_dim} dims (reshape then PCA)...")
            X_reshaped = X_all.view(len(X_all), seq_len, feature_dim)
            X_stacked_np = X_reshaped.reshape(-1, feature_dim).numpy()
            pca = PCA(n_components=pca_dim, random_state=SEED)
            X_reduced_np = pca.fit_transform(X_stacked_np)
            explained_variance_ratio = pca.explained_variance_ratio_.sum()
            X_reduced = torch.tensor(X_reduced_np, dtype=torch.float32).view(len(X_all), seq_len, pca_dim)
            X_all = X_reduced.reshape(len(X_all), -1)
            feature_dim = pca_dim
            print(f"PCA done. New X shape: {X_all.shape}, variance retained: {explained_variance_ratio:.4f}")
        else:
            print(f"\nPCA to {pca_dim} dims...")
            X_numpy = X_all.numpy()
            pca = PCA(n_components=pca_dim, random_state=SEED)
            X_reduced = pca.fit_transform(X_numpy)
            X_all = torch.tensor(X_reduced, dtype=torch.float32)
            explained_variance_ratio = pca.explained_variance_ratio_.sum()
            print(f"PCA done. New X shape: {X_all.shape}, variance retained: {explained_variance_ratio:.4f}")
            # Update dims after PCA
            seq_len = 1
            feature_dim = pca_dim
            
    meta_dict = {
        'preds': np.array(preds_list_out),
        'gts': np.array(gts_list_out),
        'input_tokens': input_tokens_out,
        'incoming_carries': np.array(incoming_carries_out),
        'outgoing_carries': np.array(outgoing_carries_out),
    }
    
    return X_all, y_all, y_binary, position_indices, selected_positions, seq_len, feature_dim, is_regression_task, sample_idx_all, meta_dict
