import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import ProbeMLP, ProbeMLPRegressor
from src.utils import flow_utils as flow_utils_module
from src.utils.cli import parse_bool_arg
from src.utils.metrics import get_off_by_one_direction
from src.utils.probe_data import compute_c_potential, compute_raw_sum, load_dataset


DEFAULT_H5 = Path("results/activations/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5")
DEFAULT_OUTDIR = Path("results/error_decomposition")
DEFAULT_POS = 4
DEFAULT_LAYER = 36
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 20
DEFAULT_LR = 1e-4
DEFAULT_MLP_DIRECT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CORRECT_TRAIN_RATIO = 0.8
DEFAULT_PROBE_MODE = "dual_probe"
DEFAULT_SAMPLE_FILTER_MODE = "all"
DEFAULT_BALANCE_MODE = "none"
DEFAULT_BALANCE_TARGET_CLASSES = False
DEFAULT_MLP_HIDDEN_DIM = 512
DEFAULT_MLP_DROPOUT = 0.2

BUCKET_ORDER = [
    "both_correct",
    "raw_correct_carry_wrong",
    "raw_wrong_carry_correct",
    "both_wrong",
]


@dataclass
class PositionData:
    features: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray
    pred_digits: np.ndarray
    gt_digits: np.ndarray
    raw_sum_full: np.ndarray
    c_potential: np.ndarray
    result_lens: np.ndarray
    gt_char_match: np.ndarray
    questions: List[str]
    gts: List[str]
    pred_tokens: List[str]
    gt_chars_h5: List[str]

    def subset(self, mask: np.ndarray) -> "PositionData":
        indices = np.flatnonzero(mask)
        return self.take(indices)

    def take(self, indices: np.ndarray) -> "PositionData":
        idx = np.asarray(indices, dtype=np.int64)
        return PositionData(
            features=self.features[idx],
            sample_ids=self.sample_ids[idx],
            labels=self.labels[idx],
            pred_digits=self.pred_digits[idx],
            gt_digits=self.gt_digits[idx],
            raw_sum_full=self.raw_sum_full[idx],
            c_potential=self.c_potential[idx],
            result_lens=self.result_lens[idx],
            gt_char_match=self.gt_char_match[idx],
            questions=[self.questions[i] for i in idx.tolist()],
            gts=[self.gts[i] for i in idx.tolist()],
            pred_tokens=[self.pred_tokens[i] for i in idx.tolist()],
            gt_chars_h5=[self.gt_chars_h5[i] for i in idx.tolist()],
        )

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass
class ClassifierResult:
    model: nn.Module
    trained_epochs: int
    best_metric_name: str
    best_test_metric: Optional[float]
    train_metrics: Dict[str, Optional[float]]
    test_metrics: Dict[str, Optional[float]]
    split_strategy: str
    num_classes: int
    test_preds: np.ndarray
    test_probs: np.ndarray


@dataclass
class RegressorResult:
    model: nn.Module
    trained_epochs: int
    best_metric_name: str
    best_test_metric: Optional[float]
    train_metrics: Dict[str, Optional[float]]
    test_metrics: Dict[str, Optional[float]]
    split_strategy: str
    test_preds: np.ndarray


def normalize_outdir(raw_outdir: str) -> Path:
    text = str(raw_outdir)
    if os.name != "nt":
        text = text.replace("\\", "/")
    return Path(text)


def decode_to_str_list(values) -> List[str]:
    if values is None:
        return []
    out: List[str] = []
    for item in values:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        elif hasattr(item, "decode"):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def parse_digit(token: str) -> int:
    token = token.strip()
    return int(token) if len(token) == 1 and token.isdigit() else -1


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_get_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def load_sample_reference(h5f: h5py.File, dataset_path: Optional[Path]) -> Tuple[Dict[int, Dict[str, str]], str]:
    if "samples" in h5f:
        samples_group = h5f["samples"]
        required = {"question", "gt", "sample_idx"}
        if required.issubset(samples_group.keys()):
            questions = decode_to_str_list(samples_group["question"][:])
            gts = decode_to_str_list(samples_group["gt"][:])
            sample_ids = np.asarray(samples_group["sample_idx"][:], dtype=np.int64)
            meta: Dict[int, Dict[str, str]] = {}
            for sample_id, question, gt in zip(sample_ids.tolist(), questions, gts):
                meta[int(sample_id)] = {"question": question, "gt": gt}
            return meta, "h5_samples"

    if dataset_path is None:
        raise RuntimeError("The H5 file does not contain samples/question|gt|sample_idx and no --dataset fallback was provided.")

    dataset = load_dataset(dataset_path)
    meta = {}
    for sample_id, operands in enumerate(dataset):
        question = " + ".join(str(x) for x in operands)
        meta[sample_id] = {"question": question, "gt": str(sum(operands))}
    return meta, "dataset_pickle"


def load_position_sample_ids(pos_group: h5py.Group, row_count: int) -> np.ndarray:
    sample_ds = pos_group.get("sample_indices")
    if sample_ds is None:
        sample_ds = pos_group.get("sample_ids")
    if sample_ds is None:
        return np.arange(row_count, dtype=np.int64)
    return np.asarray(sample_ds[:], dtype=np.int64)


def collect_first_error_positions(
    h5_path: Path,
    sample_meta: Dict[int, Dict[str, str]],
) -> Dict[int, int]:
    first_error_pos: Dict[int, int] = {}
    with h5py.File(h5_path, "r") as h5f:
        all_results = h5f.get("all_token_results")
        if all_results is None:
            raise RuntimeError("The H5 file is missing the all_token_results group.")

        numeric_positions: List[Tuple[int, str]] = []
        for pos_name in all_results.keys():
            if not pos_name.startswith("pos_"):
                continue
            pos_suffix = pos_name[4:]
            if pos_suffix.lstrip("-").isdigit():
                numeric_positions.append((int(pos_suffix), pos_name))

        for pos, pos_name in sorted(numeric_positions, key=lambda item: item[0]):
            pos_group = all_results[pos_name]
            preds_ds = pos_group.get("preds")
            gt_chars_ds = pos_group.get("gt_chars")
            if preds_ds is None or gt_chars_ds is None:
                continue

            pred_tokens = decode_to_str_list(preds_ds[:])
            gt_chars_h5 = decode_to_str_list(gt_chars_ds[:])
            sample_ids = load_position_sample_ids(pos_group, len(pred_tokens))
            max_rows = min(len(pred_tokens), len(gt_chars_h5), len(sample_ids))
            for row_idx in range(max_rows):
                sample_id = int(sample_ids[row_idx])
                meta = sample_meta.get(sample_id)
                if meta is None:
                    continue

                gt = meta["gt"]
                if pos >= len(gt):
                    continue

                expected_gt_char = gt[pos]
                if gt_chars_h5[row_idx] != expected_gt_char:
                    continue

                pred_digit = parse_digit(pred_tokens[row_idx])
                gt_digit = parse_digit(expected_gt_char)
                if pred_digit == gt_digit:
                    continue

                previous_pos = first_error_pos.get(sample_id)
                if previous_pos is None or pos < previous_pos:
                    first_error_pos[sample_id] = pos

    return first_error_pos


def load_position_data(
    h5_path: Path,
    pos: int,
    layer: int,
    dataset_path: Optional[Path],
) -> Tuple[PositionData, Dict[str, int], str]:
    with h5py.File(h5_path, "r") as h5f:
        sample_meta, source = load_sample_reference(h5f, dataset_path)

        all_results = h5f.get("all_token_results")
        if all_results is None:
            raise RuntimeError("The H5 file is missing the all_token_results group.")

        pos_name = f"pos_{pos}"
        if pos_name not in all_results:
            raise RuntimeError(f"Position group {pos_name} does not exist in the H5 file.")
        pos_group = all_results[pos_name]

        flows_ds = pos_group.get("flows")
        if flows_ds is None:
            raise RuntimeError(f"{pos_name} is missing the flows dataset.")
        if flows_ds.ndim != 3:
            raise RuntimeError(f"{pos_name}/flows expects 3 dimensions, got {flows_ds.ndim}。")
        if not (0 <= layer < flows_ds.shape[1]):
            raise RuntimeError(f"Requested layer={layer}, but flows has only {flows_ds.shape[1]} layers.")

        features = np.asarray(flows_ds[:, layer, :], dtype=np.float32)
        labels = np.asarray(pos_group["labels"][:], dtype=np.bool_)
        pred_tokens = decode_to_str_list(pos_group["preds"][:])
        gt_chars_h5 = decode_to_str_list(pos_group["gt_chars"][:])
        sample_ids = load_position_sample_ids(pos_group, features.shape[0])

    stats = {
        "rows_total": int(features.shape[0]),
        "rows_missing_sample_meta": 0,
        "rows_pos_out_of_range": 0,
        "rows_invalid_raw_sum": 0,
    }

    kept_features: List[np.ndarray] = []
    kept_sample_ids: List[int] = []
    kept_labels: List[bool] = []
    kept_pred_digits: List[int] = []
    kept_gt_digits: List[int] = []
    kept_raw_sum_full: List[int] = []
    kept_c_potential: List[float] = []
    kept_result_lens: List[int] = []
    kept_gt_char_match: List[bool] = []
    kept_questions: List[str] = []
    kept_gts: List[str] = []
    kept_pred_tokens: List[str] = []
    kept_gt_chars_h5: List[str] = []

    max_rows = min(
        features.shape[0],
        len(labels),
        len(pred_tokens),
        len(gt_chars_h5),
        len(sample_ids),
    )

    for row_idx in range(max_rows):
        sample_id = int(sample_ids[row_idx])
        meta = sample_meta.get(sample_id)
        if meta is None:
            stats["rows_missing_sample_meta"] += 1
            continue

        question = meta["question"]
        gt = meta["gt"]
        if pos >= len(gt):
            stats["rows_pos_out_of_range"] += 1
            continue

        raw_sum_full = compute_raw_sum(question, pos)
        if raw_sum_full < 0:
            stats["rows_invalid_raw_sum"] += 1
            continue

        expected_gt_char = gt[pos]
        pred_token = pred_tokens[row_idx]
        gt_char_h5 = gt_chars_h5[row_idx]

        kept_features.append(features[row_idx])
        kept_sample_ids.append(sample_id)
        kept_labels.append(bool(labels[row_idx]))
        kept_pred_digits.append(parse_digit(pred_token))
        kept_gt_digits.append(parse_digit(expected_gt_char))
        kept_raw_sum_full.append(int(raw_sum_full))
        kept_c_potential.append(float(compute_c_potential(question, pos)))
        kept_result_lens.append(len(gt))
        kept_gt_char_match.append(gt_char_h5 == expected_gt_char)
        kept_questions.append(question)
        kept_gts.append(gt)
        kept_pred_tokens.append(pred_token)
        kept_gt_chars_h5.append(gt_char_h5)

    if not kept_features:
        raise RuntimeError("No valid samples were built; check whether the H5 file matches sample metadata.")

    data = PositionData(
        features=np.stack(kept_features, axis=0).astype(np.float32),
        sample_ids=np.asarray(kept_sample_ids, dtype=np.int64),
        labels=np.asarray(kept_labels, dtype=np.bool_),
        pred_digits=np.asarray(kept_pred_digits, dtype=np.int64),
        gt_digits=np.asarray(kept_gt_digits, dtype=np.int64),
        raw_sum_full=np.asarray(kept_raw_sum_full, dtype=np.int64),
        c_potential=np.asarray(kept_c_potential, dtype=np.float32),
        result_lens=np.asarray(kept_result_lens, dtype=np.int64),
        gt_char_match=np.asarray(kept_gt_char_match, dtype=np.bool_),
        questions=kept_questions,
        gts=kept_gts,
        pred_tokens=kept_pred_tokens,
        gt_chars_h5=kept_gt_chars_h5,
    )
    return data, stats, source


def load_probe_model(model_path: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
    )
    model.eval()
    return model


def load_processed_task_data(
    h5_path: Path,
    pos: int,
    model_path: str,
    seed: int,
    train_target: str,
    balance_mode: str,
    sample_filter_mode: Optional[str] = None,
    raw_sum_mod_10: bool = False,
    balance_target_classes: bool = DEFAULT_BALANCE_TARGET_CLASSES,
    loaded_model_obj=None,
):
    if sample_filter_mode is None:
        sample_filter_mode = DEFAULT_SAMPLE_FILTER_MODE
    return flow_utils_module.load_and_process_data(
        file_path=h5_path,
        position_select=pos,
        feature_type="post_ffn",
        pooling_type="none",
        use_pca=False,
        apply_carry_filter=None,
        train_target=train_target,
        sample_filter_mode=sample_filter_mode,
        apply_model_norm=True,
        model_path=model_path,
        raw_sum_mod_10=raw_sum_mod_10,
        filter_by_result_len=True,
        allowed_input_digits=None,
        allowed_output_digits=None,
        allowed_incoming_carries=None,
        allowed_target_input_digits=None,
        allowed_target_output_digits=None,
        allowed_target_incoming_carries=None,
        balance_mode=balance_mode,
        balance_target_classes=balance_target_classes,
        seed=seed,
        model_type="mlp",
        has_h5py=getattr(flow_utils_module, "HAS_H5PY", True),
        target_pos=pos,
        input_pos="consistent",
        filter_prefix_correct=None,
        loaded_model_obj=loaded_model_obj,
    )


def ensure_shared_processed_alignment(
    raw_processed,
    carry_processed,
) -> None:
    raw_sample_ids = np.asarray(raw_processed[8].cpu().numpy(), dtype=np.int64)
    carry_sample_ids = np.asarray(carry_processed[8].cpu().numpy(), dtype=np.int64)
    if raw_sample_ids.shape != carry_sample_ids.shape or not np.array_equal(raw_sample_ids, carry_sample_ids):
        raise RuntimeError("raw-sum and carry sample orders differ; cannot share the same error decomposition.")

    raw_positions = np.asarray(raw_processed[3].cpu().numpy(), dtype=np.int64)
    carry_positions = np.asarray(carry_processed[3].cpu().numpy(), dtype=np.int64)
    if not np.array_equal(raw_positions, carry_positions):
        raise RuntimeError("raw-sum and carry position_indices differ; cannot share the same split.")


def select_layer_features(X_all: torch.Tensor, seq_len: int, feature_dim: int, layer: int) -> torch.Tensor:
    if layer < 0:
        layer = seq_len + layer
    if not (0 <= layer < seq_len):
        raise RuntimeError(f"Requested layer={layer}, but external flow seq_len={seq_len}.")
    X_reshaped = X_all.view(len(X_all), seq_len, feature_dim)
    return X_reshaped[:, layer, :].contiguous()


def load_h5_position_row_count(h5_path: Path, pos: int) -> int:
    with h5py.File(h5_path, "r") as h5f:
        pos_group = h5f["all_token_results"][f"pos_{pos}"]
        return int(len(pos_group["labels"]))


def build_position_data_from_processed(
    features: torch.Tensor,
    y_binary: torch.Tensor,
    raw_labels: torch.Tensor,
    carry_labels: torch.Tensor,
    sample_idx_all: torch.Tensor,
    meta: Dict[str, np.ndarray],
    sample_meta: Dict[int, Dict[str, str]],
    pos: int,
) -> PositionData:
    features_np = features.detach().cpu().numpy().astype(np.float32)
    sample_ids = np.asarray(sample_idx_all.detach().cpu().numpy(), dtype=np.int64)
    labels = np.asarray(y_binary.detach().cpu().numpy(), dtype=np.int64).astype(np.bool_)
    raw_sum_full_list: List[int] = []
    c_potential = np.asarray(carry_labels.detach().cpu().numpy(), dtype=np.float32)
    pred_tokens = decode_to_str_list(meta["preds"])
    gt_chars_h5 = decode_to_str_list(meta["gts"])

    questions: List[str] = []
    gts: List[str] = []
    result_lens: List[int] = []
    gt_char_match: List[bool] = []
    gt_digits: List[int] = []

    for sample_id, gt_char_h5 in zip(sample_ids.tolist(), gt_chars_h5):
        sample_record = sample_meta.get(int(sample_id))
        if sample_record is None:
            raise RuntimeError(f"external flow returned sample_id={sample_id}, which does not exist in H5 samples.")
        question = sample_record["question"]
        gt = sample_record["gt"]
        if pos >= len(gt):
            raise RuntimeError(f"sample_id={sample_id} has gt that is too short for pos={pos}。")
        raw_sum_value = compute_raw_sum(question, pos)
        if raw_sum_value < 0:
            raise RuntimeError(f"sample_id={sample_id} cannot reconstruct raw_sum at pos={pos}.")
        questions.append(question)
        gts.append(gt)
        result_lens.append(len(gt))
        gt_char_match.append(gt[pos] == gt_char_h5)
        gt_digits.append(parse_digit(gt[pos]))
        raw_sum_full_list.append(int(raw_sum_value))

    pred_digits = np.asarray([parse_digit(token) for token in pred_tokens], dtype=np.int64)
    return PositionData(
        features=features_np,
        sample_ids=sample_ids,
        labels=labels,
        pred_digits=pred_digits,
        gt_digits=np.asarray(gt_digits, dtype=np.int64),
        raw_sum_full=np.asarray(raw_sum_full_list, dtype=np.int64),
        c_potential=c_potential,
        result_lens=np.asarray(result_lens, dtype=np.int64),
        gt_char_match=np.asarray(gt_char_match, dtype=np.bool_),
        questions=questions,
        gts=gts,
        pred_tokens=pred_tokens,
        gt_chars_h5=gt_chars_h5,
    )


def subset_meta_dict(meta: Dict[str, object], indices: np.ndarray) -> Dict[str, object]:
    idx = np.asarray(indices, dtype=np.int64)
    out: Dict[str, object] = {}
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=value.device)
            out[key] = value[idx_t]
        elif isinstance(value, np.ndarray):
            out[key] = value[idx]
        elif isinstance(value, list):
            out[key] = [value[i] for i in idx.tolist()]
        else:
            out[key] = value
    return out


def build_first_error_keep_mask(
    labels: np.ndarray,
    sample_ids: np.ndarray,
    target_pos: int,
    first_error_pos: Dict[int, int],
) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.bool_)
    sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
    keep = np.ones(labels_np.shape[0], dtype=np.bool_)
    wrong_mask = ~labels_np
    if not np.any(wrong_mask):
        return keep

    wrong_indices = np.flatnonzero(wrong_mask)
    keep[wrong_indices] = np.asarray(
        [first_error_pos.get(int(sample_ids_np[idx])) == target_pos for idx in wrong_indices],
        dtype=np.bool_,
    )
    return keep


def take_tensor_rows(tensor: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor[idx_t]


def pick_target_result_len(result_lens: np.ndarray) -> int:
    counter = Counter(int(v) for v in result_lens.tolist())
    return counter.most_common(1)[0][0]


def split_train_test_indices(
    y_all: np.ndarray,
    y_binary: np.ndarray,
    seed: int,
    test_size: float,
    split_name: str = "train/test",
) -> Tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(len(y_all))
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y_all,
        )
        return train_idx, test_idx, "y_all"
    except ValueError as exc:
        warnings.warn(f"{split_name} stratified split by y_all failed; falling back to y_binary: {exc}")
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y_binary,
        )
        return train_idx, test_idx, "y_binary"
    except ValueError as exc:
        warnings.warn(f"{split_name} stratified split by y_binary failed; falling back to unstratified split: {exc}")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        return train_idx, test_idx, "none"


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensor_x = torch.from_numpy(features.astype(np.float32))
    tensor_y = torch.from_numpy(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_multiclass_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> Optional[float]:
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            try:
                y_true_onehot = np.eye(num_classes)[y_true]
                auc = roc_auc_score(y_true_onehot.ravel(), probs.ravel())
            except Exception:
                auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        auc = float(auc)
        if math.isnan(auc) or math.isinf(auc):
            return 0.0
        return auc
    except Exception:
        return 0.0


def evaluate_classifier(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    loader = make_loader(features, labels.astype(np.int64), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    losses: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            losses.append(float(loss.item()))
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    preds_np = np.concatenate(all_preds, axis=0)
    acc = float(np.mean(preds_np == labels))
    auc = compute_multiclass_auc(labels, probs_np, num_classes)

    return {
        "loss": float(np.mean(losses)),
        "acc": acc,
        "auc": auc,
        "preds": preds_np,
        "probs": probs_np,
    }


def evaluate_regressor(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    loader = make_loader(features, labels.astype(np.float32), batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    preds_all: List[np.ndarray] = []
    losses: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            losses.append(float(loss.item()))
            preds_all.append(preds.cpu().numpy())

    preds_np = np.concatenate(preds_all, axis=0)
    mae = float(np.mean(np.abs(preds_np - labels)))
    pred_floor = np.floor(np.maximum(preds_np, 0.0)).astype(np.int64)
    label_floor = np.floor(np.maximum(labels, 0.0)).astype(np.int64)
    floor_acc = float(np.mean(pred_floor == label_floor))

    return {
        "loss": float(np.mean(losses)),
        "mae": mae,
        "floor_acc": floor_acc,
        "preds": preds_np,
    }


def train_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    split_strategy: str,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    dropout: float = DEFAULT_MLP_DROPOUT,
) -> ClassifierResult:
    num_classes = int(max(train_labels.max(), test_labels.max())) + 1
    model = ProbeMLP(
        input_dim=train_features.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(train_features, train_labels.astype(np.int64), batch_size=batch_size, shuffle=True)

    best_metric_name = "auc"
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        test_metrics_epoch = evaluate_classifier(model, test_features, test_labels, batch_size, device, num_classes)
        metric_value = test_metrics_epoch["auc"]
        metric_score = float(metric_value) if metric_value is not None else float(test_metrics_epoch["acc"])
        if metric_score > best_score:
            best_score = metric_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics_raw = evaluate_classifier(model, train_features, train_labels, batch_size, device, num_classes)
    test_metrics_raw = evaluate_classifier(model, test_features, test_labels, batch_size, device, num_classes)

    train_metrics = {
        "loss": float(train_metrics_raw["loss"]),
        "acc": float(train_metrics_raw["acc"]),
        "auc": None if train_metrics_raw["auc"] is None else float(train_metrics_raw["auc"]),
    }
    test_metrics = {
        "loss": float(test_metrics_raw["loss"]),
        "acc": float(test_metrics_raw["acc"]),
        "auc": None if test_metrics_raw["auc"] is None else float(test_metrics_raw["auc"]),
    }

    return ClassifierResult(
        model=model,
        trained_epochs=best_epoch if best_epoch > 0 else epochs,
        best_metric_name=best_metric_name,
        best_test_metric=None if best_score == float("-inf") else float(best_score),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        split_strategy=split_strategy,
        num_classes=num_classes,
        test_preds=np.asarray(test_metrics_raw["preds"], dtype=np.int64),
        test_probs=np.asarray(test_metrics_raw["probs"], dtype=np.float32),
    )


def train_regressor(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    split_strategy: str,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    dropout: float = DEFAULT_MLP_DROPOUT,
) -> RegressorResult:
    model = ProbeMLPRegressor(
        input_dim=train_features.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(train_features, train_labels.astype(np.float32), batch_size=batch_size, shuffle=True)

    best_metric_name = "mae"
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        test_metrics_epoch = evaluate_regressor(model, test_features, test_labels, batch_size, device)
        metric_value = float(test_metrics_epoch["mae"])
        if metric_value < best_score:
            best_score = metric_value
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics_raw = evaluate_regressor(model, train_features, train_labels, batch_size, device)
    test_metrics_raw = evaluate_regressor(model, test_features, test_labels, batch_size, device)

    train_metrics = {
        "loss": float(train_metrics_raw["loss"]),
        "mae": float(train_metrics_raw["mae"]),
        "floor_acc": float(train_metrics_raw["floor_acc"]),
    }
    test_metrics = {
        "loss": float(test_metrics_raw["loss"]),
        "mae": float(test_metrics_raw["mae"]),
        "floor_acc": float(test_metrics_raw["floor_acc"]),
    }

    return RegressorResult(
        model=model,
        trained_epochs=best_epoch if best_epoch > 0 else epochs,
        best_metric_name=best_metric_name,
        best_test_metric=float(best_score) if best_epoch > 0 else None,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        split_strategy=split_strategy,
        test_preds=np.asarray(test_metrics_raw["preds"], dtype=np.float32),
    )


def class_distribution(values: np.ndarray) -> Dict[str, int]:
    counter = Counter(int(v) for v in values.tolist())
    return {str(k): int(v) for k, v in sorted(counter.items())}


def regression_floor_distribution(values: np.ndarray) -> Dict[str, int]:
    floored = np.floor(np.maximum(values, 0.0)).astype(np.int64)
    return class_distribution(floored)


def make_empty_error_bucket_stats() -> Dict[str, Dict[str, float]]:
    return {
        bucket: {
            "count": 0,
            "ratio": 0.0,
            "fixed_count": 0,
            "fixed_rate": 0.0,
        }
        for bucket in BUCKET_ORDER
    }


def compute_corrected_digits_with_inertia_delta(
    raw_pred_labels: np.ndarray,
    carry_preds_phi: np.ndarray,
    pred_digits: np.ndarray,
    inertia_delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_hat_labels = np.asarray(raw_pred_labels, dtype=np.int64)
    raw_hat_mod = raw_hat_labels % 10
    carry_phi = np.maximum(np.asarray(carry_preds_phi, dtype=np.float32), 0.0)
    carry_hat = np.floor(carry_phi).astype(np.int64)
    pred_arr = np.asarray(pred_digits, dtype=np.int64)
    corrected_digits = np.empty_like(raw_hat_mod, dtype=np.int64)

    for idx in range(raw_hat_mod.shape[0]):
        raw_d = int(raw_hat_mod[idx])
        phi = float(carry_phi[idx])
        pred_digit = int(pred_arr[idx]) if 0 <= pred_arr[idx] <= 9 else -1

        low = math.floor(phi - inertia_delta)
        high = math.floor(phi + inertia_delta)
        intervene = True
        if pred_digit != -1:
            for carry_candidate in range(low, high + 1):
                if (raw_d + carry_candidate) % 10 == pred_digit:
                    intervene = False
                    break

        if intervene:
            corrected_digits[idx] = (raw_d + int(carry_hat[idx])) % 10
        else:
            corrected_digits[idx] = pred_digit if pred_digit != -1 else (raw_d + int(carry_hat[idx])) % 10

    return corrected_digits, carry_hat


def summarize_error_type_repairs(
    pred_digits: np.ndarray,
    gt_digits: np.ndarray,
    corrected_digits: np.ndarray,
    is_correct_mask: np.ndarray,
) -> Dict[str, object]:
    pred_arr = np.asarray(pred_digits, dtype=np.int64)
    gt_arr = np.asarray(gt_digits, dtype=np.int64)
    corrected_arr = np.asarray(corrected_digits, dtype=np.int64)
    incorrect_mask = ~np.asarray(is_correct_mask, dtype=bool)

    orig_error_count = int(np.sum(incorrect_mask))
    off_by_one_count = 0
    off_by_one_fix_count = 0
    other_error_count = 0
    other_error_fix_count = 0

    for idx in np.flatnonzero(incorrect_mask):
        direction = get_off_by_one_direction(int(pred_arr[idx]), int(gt_arr[idx]))
        is_fixed = int(corrected_arr[idx]) == int(gt_arr[idx])
        if direction is not None:
            off_by_one_count += 1
            if is_fixed:
                off_by_one_fix_count += 1
        else:
            other_error_count += 1
            if is_fixed:
                other_error_fix_count += 1

    denom = orig_error_count if orig_error_count > 0 else 1
    return {
        "orig_error_count": int(orig_error_count),
        "off_by_one_count": int(off_by_one_count),
        "other_error_count": int(other_error_count),
        "off_by_one_ratio": float(off_by_one_count / denom) if orig_error_count > 0 else 0.0,
        "other_error_ratio": float(other_error_count / denom) if orig_error_count > 0 else 0.0,
        "off_by_one_fix_count": int(off_by_one_fix_count),
        "off_by_one_fix_rate": float(off_by_one_fix_count / off_by_one_count) if off_by_one_count > 0 else 0.0,
        "other_error_fix_count": int(other_error_fix_count),
        "other_error_fix_rate": float(other_error_fix_count / other_error_count) if other_error_count > 0 else 0.0,
    }


def summarize_error_type_bucket_stats(
    rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats = {
        "off_by_one_bucket_stats": make_empty_error_bucket_stats(),
        "other_error_bucket_stats": make_empty_error_bucket_stats(),
    }
    totals = {
        "off_by_one_bucket_stats": 0,
        "other_error_bucket_stats": 0,
    }

    for row in rows:
        if int(row["is_correct"]) != 0:
            continue
        error_type = str(row["error_type"])
        bucket = str(row["bucket"])
        if bucket not in BUCKET_ORDER:
            continue
        if error_type == "off_by_one":
            stat_key = "off_by_one_bucket_stats"
        elif error_type == "other_error":
            stat_key = "other_error_bucket_stats"
        else:
            continue

        bucket_stats = stats[stat_key][bucket]
        bucket_stats["count"] += 1
        totals[stat_key] += 1
        if int(row["probe_fixed_to_gt"]) == 1:
            bucket_stats["fixed_count"] += 1

    for stat_key, bucket_stats in stats.items():
        total_count = totals[stat_key]
        denom = total_count if total_count > 0 else 1
        for bucket in BUCKET_ORDER:
            bucket_count = int(bucket_stats[bucket]["count"])
            fixed_count = int(bucket_stats[bucket]["fixed_count"])
            bucket_stats[bucket]["count"] = bucket_count
            bucket_stats[bucket]["ratio"] = float(bucket_count / denom) if total_count > 0 else 0.0
            bucket_stats[bucket]["fixed_count"] = fixed_count
            bucket_stats[bucket]["fixed_rate"] = float(fixed_count / bucket_count) if bucket_count > 0 else 0.0

    return stats


def format_error_bucket_stats(
    label: str,
    bucket_stats: Dict[str, Dict[str, float]],
) -> str:
    parts = []
    for bucket in BUCKET_ORDER:
        stats = bucket_stats[bucket]
        parts.append(
            f"{bucket}={int(stats['count'])} ({float(stats['ratio']):.4f}), "
            f"fixed={int(stats['fixed_count'])} ({float(stats['fixed_rate']):.4f})"
        )
    return f"{label}: " + " | ".join(parts)


def summarize_split(data: PositionData) -> Dict[str, object]:
    raw_sum_mod10 = data.raw_sum_full % 10
    return {
        "rows": int(len(data)),
        "unique_sample_ids": int(np.unique(data.sample_ids).size),
        "label_correct_distribution": class_distribution(data.labels.astype(np.int64)),
        "raw_sum_distribution": class_distribution(data.raw_sum_full),
        "raw_sum_full_distribution": class_distribution(data.raw_sum_full),
        "raw_sum_mod10_distribution": class_distribution(raw_sum_mod10),
        "carry_floor_distribution": regression_floor_distribution(data.c_potential),
    }


def decompose_test_errors(
    test_data: PositionData,
    raw_pred_labels: np.ndarray,
    carry_preds_phi: np.ndarray,
    raw_sum_mod_10: bool,
    inertia_delta: float,
) -> Tuple[List[Dict[str, object]], Dict[str, int], np.ndarray]:
    rows: List[Dict[str, object]] = []
    counts = {bucket: 0 for bucket in BUCKET_ORDER}
    counts["correct"] = 0

    raw_gt_full = test_data.raw_sum_full
    raw_gt_mod = raw_gt_full % 10
    raw_gt_labels = raw_gt_mod if raw_sum_mod_10 else raw_gt_full
    raw_hat_labels = np.asarray(raw_pred_labels, dtype=np.int64)
    carry_gt = np.floor(np.maximum(test_data.c_potential, 0.0)).astype(np.int64)
    raw_hat_mod = raw_hat_labels % 10
    corrected_digits, carry_hat = compute_corrected_digits_with_inertia_delta(
        raw_pred_labels=raw_hat_labels,
        carry_preds_phi=carry_preds_phi,
        pred_digits=test_data.pred_digits,
        inertia_delta=inertia_delta,
    )

    for idx in range(len(test_data)):
        is_correct = bool(test_data.labels[idx])
        pred_digit = int(test_data.pred_digits[idx])
        explained_digit = int(corrected_digits[idx])
        off_by_one_direction = None if is_correct else get_off_by_one_direction(pred_digit, int(test_data.gt_digits[idx]))
        error_type = "correct" if is_correct else ("off_by_one" if off_by_one_direction is not None else "other_error")
        probe_fixed_to_gt = int((not is_correct) and explained_digit == int(test_data.gt_digits[idx]))

        if is_correct:
            bucket = "correct"
        else:
            raw_ok = int(raw_hat_labels[idx]) == int(raw_gt_labels[idx])
            carry_ok = int(carry_hat[idx]) == int(carry_gt[idx])
            if raw_ok and carry_ok:
                bucket = "both_correct"
            elif raw_ok:
                bucket = "raw_correct_carry_wrong"
            elif carry_ok:
                bucket = "raw_wrong_carry_correct"
            else:
                bucket = "both_wrong"

        counts[bucket] = counts.get(bucket, 0) + 1
        rows.append(
            {
                "sample_id": int(test_data.sample_ids[idx]),
                "question": test_data.questions[idx],
                "gt": test_data.gts[idx],
                "gt_char_h5": test_data.gt_chars_h5[idx],
                "gt_digit": int(test_data.gt_digits[idx]),
                "pred_token": test_data.pred_tokens[idx],
                "pred_digit": pred_digit,
                "is_correct": int(is_correct),
                "error_type": error_type,
                "off_by_one_direction": off_by_one_direction,
                "raw_gt_full": int(raw_gt_full[idx]),
                "raw_gt_mod10": int(raw_gt_mod[idx]),
                "raw_gt_label": int(raw_gt_labels[idx]),
                "raw_hat_full": None if raw_sum_mod_10 else int(raw_hat_labels[idx]),
                "raw_hat_label": int(raw_hat_labels[idx]),
                "raw_hat_mod10": int(raw_hat_mod[idx]),
                "c_potential_gt": float(test_data.c_potential[idx]),
                "c_potential_hat": float(carry_preds_phi[idx]),
                "carry_gt": int(carry_gt[idx]),
                "carry_hat": int(carry_hat[idx]),
                "probe_explained_digit": explained_digit,
                "probe_fixed_to_gt": probe_fixed_to_gt,
                "bucket": bucket,
            }
        )

    return rows, counts, corrected_digits


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise RuntimeError("No test details are available for CSV output.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def split_correct_training_subset(
    labels_correct_mask: np.ndarray,
    train_target_labels: np.ndarray,
    seed: int,
    test_size: float,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    correct_indices = np.flatnonzero(labels_correct_mask)
    if correct_indices.size < 2:
        raise RuntimeError(f"{split_name} After enabling train_on_correct, there are too few correct samples for a train/validation split.")

    correct_target_labels = np.asarray(train_target_labels[correct_indices])
    correct_binary = np.ones(correct_indices.size, dtype=np.int64)
    inner_train_rel, inner_val_rel, strategy = split_train_test_indices(
        y_all=correct_target_labels,
        y_binary=correct_binary,
        seed=seed,
        test_size=test_size,
        split_name=split_name,
    )
    return correct_indices[inner_train_rel], correct_indices[inner_val_rel], strategy



def validate_digit_labels(labels: np.ndarray, label_name: str) -> None:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.size == 0:
        raise RuntimeError(f"{label_name} is empty; cannot train/evaluate mlp_direct.")
    invalid_mask = (arr < 0) | (arr > 9)
    if np.any(invalid_mask):
        bad_values = np.unique(arr[invalid_mask]).tolist()
        raise RuntimeError(f"{label_name} contains labels outside 0-9: {bad_values[:10]}")


def train_mlp_direct_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    split_strategy: str,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    dropout: float = DEFAULT_MLP_DROPOUT,
) -> Tuple[ClassifierResult, Dict[str, Optional[float]]]:
    validate_digit_labels(train_labels, "train_labels")
    validate_digit_labels(val_labels, "val_labels")
    validate_digit_labels(test_labels, "test_labels")

    num_classes = 10
    model = ProbeMLP(
        input_dim=train_features.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_loader(train_features, train_labels.astype(np.int64), batch_size=batch_size, shuffle=True)

    best_metric_name = "acc"
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_metrics_epoch = evaluate_classifier(model, val_features, val_labels, batch_size, device, num_classes)
        metric_score = float(val_metrics_epoch["acc"])
        if metric_score > best_score:
            best_score = metric_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics_raw = evaluate_classifier(model, train_features, train_labels, batch_size, device, num_classes)
    val_metrics_raw = evaluate_classifier(model, val_features, val_labels, batch_size, device, num_classes)
    test_metrics_raw = evaluate_classifier(model, test_features, test_labels, batch_size, device, num_classes)

    train_metrics = {
        "loss": float(train_metrics_raw["loss"]),
        "acc": float(train_metrics_raw["acc"]),
        "auc": None if train_metrics_raw["auc"] is None else float(train_metrics_raw["auc"]),
    }
    val_metrics = {
        "loss": float(val_metrics_raw["loss"]),
        "acc": float(val_metrics_raw["acc"]),
        "auc": None if val_metrics_raw["auc"] is None else float(val_metrics_raw["auc"]),
    }
    test_metrics = {
        "loss": float(test_metrics_raw["loss"]),
        "acc": float(test_metrics_raw["acc"]),
        "auc": None if test_metrics_raw["auc"] is None else float(test_metrics_raw["auc"]),
    }

    result = ClassifierResult(
        model=model,
        trained_epochs=best_epoch if best_epoch > 0 else epochs,
        best_metric_name=best_metric_name,
        best_test_metric=None if best_score == float("-inf") else float(best_score),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        split_strategy=split_strategy,
        num_classes=num_classes,
        test_preds=np.asarray(test_metrics_raw["preds"], dtype=np.int64),
        test_probs=np.asarray(test_metrics_raw["probs"], dtype=np.float32),
    )
    return result, val_metrics


def compute_mlp_direct_correction_metrics(
    corrected_digits: np.ndarray,
    pred_orig_digits: np.ndarray,
    gt_digits: np.ndarray,
) -> Dict[str, float]:
    corrected = np.asarray(corrected_digits, dtype=np.int64)
    pred_orig = np.asarray(pred_orig_digits, dtype=np.int64)
    gt = np.asarray(gt_digits, dtype=np.int64)
    if corrected.shape[0] != pred_orig.shape[0] or corrected.shape[0] != gt.shape[0]:
        raise RuntimeError("mlp_direct evaluation lengths are inconsistent.")

    n = corrected.shape[0]
    if n == 0:
        return {
            "modified_rate": 0.0,
            "tp_correction": 0.0,
            "fp_preservation": 0.0,
        }

    modified_rate = float(np.sum(corrected != pred_orig) / n)
    orig_errors = pred_orig != gt
    tp_total = int(np.sum(orig_errors))
    tp_correction = float(np.sum(orig_errors & (corrected == gt)) / tp_total) if tp_total > 0 else 0.0

    orig_correct = pred_orig == gt
    fp_total = int(np.sum(orig_correct))
    fp_preservation = float(np.sum(orig_correct & (corrected == gt)) / fp_total) if fp_total > 0 else 0.0
    return {
        "modified_rate": modified_rate,
        "tp_correction": tp_correction,
        "fp_preservation": fp_preservation,
    }


def decompose_test_errors_mlp_direct(
    test_data: PositionData,
    mlp_pred_digits: np.ndarray,
) -> List[Dict[str, object]]:
    pred_arr = np.asarray(mlp_pred_digits, dtype=np.int64)
    if pred_arr.shape[0] != len(test_data):
        raise RuntimeError("mlp_direct prediction length differs from test_data.")

    rows: List[Dict[str, object]] = []
    for idx in range(len(test_data)):
        is_correct = bool(test_data.labels[idx])
        pred_digit = int(test_data.pred_digits[idx])
        gt_digit = int(test_data.gt_digits[idx])
        mlp_digit = int(pred_arr[idx])

        off_by_one_direction = None if is_correct else get_off_by_one_direction(pred_digit, gt_digit)
        error_type = "correct" if is_correct else ("off_by_one" if off_by_one_direction is not None else "other_error")
        rows.append(
            {
                "sample_id": int(test_data.sample_ids[idx]),
                "question": test_data.questions[idx],
                "gt": test_data.gts[idx],
                "gt_char_h5": test_data.gt_chars_h5[idx],
                "gt_digit": gt_digit,
                "pred_token": test_data.pred_tokens[idx],
                "pred_digit": pred_digit,
                "is_correct": int(is_correct),
                "error_type": error_type,
                "off_by_one_direction": off_by_one_direction,
                "mlp_pred_digit": mlp_digit,
                "mlp_changed_pred": int(mlp_digit != pred_digit),
                "mlp_fixed_to_gt": int((not is_correct) and (mlp_digit == gt_digit)),
            }
        )
    return rows

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train raw-sum and carry probes from H5 samples, then decompose test errors into four buckets.",
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--dataset", type=Path, default=Path("data/num3len10-10000.pkl"), help="Fallback dataset used only when the H5 file lacks sample metadata.")
    parser.add_argument("--pos", type=int, default=DEFAULT_POS)
    parser.add_argument("--model", type=str, default="/data/Models/Qwen3-4b", help="Model path used to extract final-norm parameters.")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--probe-mode", type=str, choices=["dual_probe", "mlp_direct"], default=DEFAULT_PROBE_MODE, help="Probe mode: dual_probe uses raw+carry probes; mlp_direct predicts the ground-truth digit directly.")
    parser.add_argument("--raw-sum-mod-10", action="store_true", help="Use 10-class raw-sum labels (raw_sum mod 10).")
    parser.add_argument("--train-on-correct", action="store_true", help="Use only correct samples for train/validation probe training.")
    parser.add_argument("--correct-train-ratio", type=float, default=DEFAULT_CORRECT_TRAIN_RATIO, help="When --train-on-correct is enabled, fraction of correct samples assigned to train; the rest go to validation.")
    parser.add_argument("--sample-filter", choices=["all", "correct", "incorrect"], default=DEFAULT_SAMPLE_FILTER_MODE)
    parser.add_argument("--balance-mode", choices=["none", "normal", "strong"], default=DEFAULT_BALANCE_MODE)
    parser.add_argument("--balance-target-classes", type=parse_bool_arg, nargs="?", const=True, default=DEFAULT_BALANCE_TARGET_CLASSES)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--mlp-hidden-dim", type=int, default=DEFAULT_MLP_HIDDEN_DIM)
    parser.add_argument("--mlp-dropout", type=float, default=DEFAULT_MLP_DROPOUT)
    parser.add_argument(
        "--first-error",
        type=parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
        help="Keep only samples whose target position is the first error for that sample; default true.",
    )
    parser.add_argument("--inertia-delta", type=float, default=0.1, help="Phi-neighborhood gate copied from dualstream_probe.")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (0.0 < args.correct_train_ratio < 1.0):
        raise ValueError("--correct-train-ratio must be in the open interval (0, 1).")
    if args.inertia_delta < 0:
        raise ValueError("--inertia-delta must be >= 0.")
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    if args.probe_mode == "mlp_direct":
        if "--raw-sum-mod-10" in cli_args:
            raise ValueError("mlp_direct mode does not support --raw-sum-mod-10.")
        if "--inertia-delta" in cli_args:
            raise ValueError("mlp_direct mode does not support --inertia-delta.")
        if "--lr" not in cli_args:
            args.lr = DEFAULT_MLP_DIRECT_LR

    set_global_seed(args.seed)
    device = maybe_get_device(args.device)
    outdir = normalize_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading position data from {args.h5} (pos={args.pos}, layer={args.layer})...")
    with h5py.File(args.h5, "r") as h5f:
        sample_meta, sample_reference_source = load_sample_reference(h5f, args.dataset)
    if not sample_meta:
        raise RuntimeError("The target H5 lacks usable sample metadata; provide --dataset as a sample_id -> question/gt fallback.")
    print(f"Using sample metadata source: {sample_reference_source}")

    print(f"Applying final norm using model: {args.model}")
    loaded_model_obj = load_probe_model(args.model)
    try:
        raw_unbalanced = load_processed_task_data(
            h5_path=args.h5,
            pos=args.pos,
            model_path=args.model,
            seed=args.seed,
            train_target="raw_sum_classify",
            balance_mode="none",
            raw_sum_mod_10=args.raw_sum_mod_10,
            balance_target_classes=args.balance_target_classes,
            loaded_model_obj=loaded_model_obj,
        )
        carry_unbalanced = load_processed_task_data(
            h5_path=args.h5,
            pos=args.pos,
            model_path=args.model,
            seed=args.seed,
            train_target="C_potential",
            balance_mode="none",
            balance_target_classes=args.balance_target_classes,
            loaded_model_obj=loaded_model_obj,
        )
    finally:
        del loaded_model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensure_shared_processed_alignment(raw_unbalanced, carry_unbalanced)

    (
        raw_X_unbalanced,
        raw_y_unbalanced,
        raw_y_binary_unbalanced,
        raw_position_indices_unbalanced,
        selected_positions,
        seq_len,
        feature_dim,
        _is_regression_raw,
        raw_sample_ids_unbalanced,
        raw_meta_unbalanced,
    ) = raw_unbalanced
    (
        carry_X_unbalanced,
        carry_y_unbalanced,
        carry_y_binary_unbalanced,
        carry_position_indices_unbalanced,
        _selected_positions_carry,
        carry_seq_len,
        carry_feature_dim,
        _is_regression_carry,
        carry_sample_ids_unbalanced,
        carry_meta_unbalanced,
    ) = carry_unbalanced

    raw_fit_pool_rows = None
    carry_fit_pool_rows = None
    fit_sample_filter_mode = args.sample_filter
    if seq_len != carry_seq_len or feature_dim != carry_feature_dim:
        raise RuntimeError("external flow_utils returned inconsistent seq_len/feature_dim values.")
    if not torch.equal(raw_y_binary_unbalanced, carry_y_binary_unbalanced):
        raise RuntimeError("raw-sum and carry y_binary differ; cannot align classifier.py balancing logic.")
    if not torch.equal(raw_position_indices_unbalanced, carry_position_indices_unbalanced):
        raise RuntimeError("raw-sum and carry position_indices differ.")

    raw_data_alignment = build_position_data_from_processed(
        features=select_layer_features(raw_X_unbalanced, seq_len, feature_dim, args.layer),
        y_binary=raw_y_binary_unbalanced,
        raw_labels=raw_y_unbalanced,
        carry_labels=torch.tensor(
            [
                compute_c_potential(sample_meta[int(sample_id)]["question"], args.pos)
                for sample_id in raw_sample_ids_unbalanced.detach().cpu().numpy().tolist()
            ],
            dtype=torch.float32,
        ),
        sample_idx_all=raw_sample_ids_unbalanced,
        meta=raw_meta_unbalanced,
        sample_meta=sample_meta,
        pos=args.pos,
    )
    total_rows = load_h5_position_row_count(args.h5, args.pos)
    mismatch_count = int((~raw_data_alignment.gt_char_match).sum())
    target_result_len = pick_target_result_len(raw_data_alignment.result_lens)
    dropped_by_result_len = int(total_rows - len(raw_data_alignment))
    pre_balance_rows = len(raw_data_alignment)
    pre_balance_correct = int(raw_data_alignment.labels.sum())
    pre_balance_wrong = int((~raw_data_alignment.labels).sum())

    if args.first_error:
        first_error_pos = collect_first_error_positions(args.h5, sample_meta)
        first_error_keep = build_first_error_keep_mask(
            labels=raw_data_alignment.labels,
            sample_ids=raw_data_alignment.sample_ids,
            target_pos=args.pos,
            first_error_pos=first_error_pos,
        )
        kept_after_first_error = int(first_error_keep.sum())
        dropped_after_first_error = int((~first_error_keep).sum())
        print(
            "Applying first-error filter: "
            f"kept={kept_after_first_error} | dropped={dropped_after_first_error} | target_pos={args.pos}"
        )
        if kept_after_first_error == 0:
            raise RuntimeError("No samples remain after enabling --first-error.")

        first_error_indices = np.flatnonzero(first_error_keep)
        raw_X_unbalanced = take_tensor_rows(raw_X_unbalanced, first_error_indices)
        raw_y_unbalanced = take_tensor_rows(raw_y_unbalanced, first_error_indices)
        raw_y_binary_unbalanced = take_tensor_rows(raw_y_binary_unbalanced, first_error_indices)
        raw_position_indices_unbalanced = take_tensor_rows(raw_position_indices_unbalanced, first_error_indices)
        raw_sample_ids_unbalanced = take_tensor_rows(raw_sample_ids_unbalanced, first_error_indices)
        raw_meta_unbalanced = subset_meta_dict(raw_meta_unbalanced, first_error_indices)

        carry_X_unbalanced = take_tensor_rows(carry_X_unbalanced, first_error_indices)
        carry_y_unbalanced = take_tensor_rows(carry_y_unbalanced, first_error_indices)
        carry_y_binary_unbalanced = take_tensor_rows(carry_y_binary_unbalanced, first_error_indices)
        carry_position_indices_unbalanced = take_tensor_rows(carry_position_indices_unbalanced, first_error_indices)
        carry_sample_ids_unbalanced = take_tensor_rows(carry_sample_ids_unbalanced, first_error_indices)
        carry_meta_unbalanced = subset_meta_dict(carry_meta_unbalanced, first_error_indices)

    effective_balance_mode = "none" if args.train_on_correct else args.balance_mode

    if effective_balance_mode == "strong":
        kept_indices = flow_utils_module.balance_indices_per_position_numpy(
            raw_y_binary_unbalanced.detach().cpu().numpy(),
            raw_position_indices_unbalanced.detach().cpu().numpy(),
            selected_positions,
            seed=args.seed,
        )
    elif effective_balance_mode == "normal":
        kept_indices = flow_utils_module.balance_indices_numpy(
            raw_y_binary_unbalanced.detach().cpu().numpy(),
            seed=args.seed,
        )
    elif effective_balance_mode == "none":
        kept_indices = np.arange(len(raw_y_binary_unbalanced), dtype=np.int64)
    else:
        raise RuntimeError(f"Unsupported balance_mode: {effective_balance_mode}")

    raw_X_balanced = take_tensor_rows(raw_X_unbalanced, kept_indices)
    raw_y_balanced = take_tensor_rows(raw_y_unbalanced, kept_indices)
    raw_y_binary_balanced = take_tensor_rows(raw_y_binary_unbalanced, kept_indices)
    raw_position_indices = take_tensor_rows(raw_position_indices_unbalanced, kept_indices)
    raw_sample_ids_balanced = take_tensor_rows(raw_sample_ids_unbalanced, kept_indices)
    raw_meta_balanced = subset_meta_dict(raw_meta_unbalanced, kept_indices)

    carry_X_balanced = take_tensor_rows(carry_X_unbalanced, kept_indices)
    carry_y_balanced = take_tensor_rows(carry_y_unbalanced, kept_indices)
    carry_y_binary_balanced = take_tensor_rows(carry_y_binary_unbalanced, kept_indices)
    carry_position_indices = take_tensor_rows(carry_position_indices_unbalanced, kept_indices)
    carry_sample_ids_balanced = take_tensor_rows(carry_sample_ids_unbalanced, kept_indices)
    carry_meta_balanced = subset_meta_dict(carry_meta_unbalanced, kept_indices)
    if not torch.equal(raw_y_binary_balanced, carry_y_binary_balanced):
        raise RuntimeError("After balancing, raw-sum and carry y_binary differ.")
    if not torch.equal(raw_sample_ids_balanced, carry_sample_ids_balanced):
        raise RuntimeError("After balancing, raw-sum and carry sample_id orders differ.")
    data = build_position_data_from_processed(
        features=select_layer_features(raw_X_balanced, seq_len, feature_dim, args.layer),
        y_binary=raw_y_binary_balanced,
        raw_labels=raw_y_balanced,
        carry_labels=carry_y_balanced,
        sample_idx_all=raw_sample_ids_balanced,
        meta=raw_meta_balanced,
        sample_meta=sample_meta,
        pos=args.pos,
    )

    print(
        "Filtered rows: "
        f"kept={pre_balance_rows} | "
        f"target_result_len={target_result_len} | "
        f"dropped_by_result_len={dropped_by_result_len}"
    )
    print(
        "Working rows: "
        f"rows={len(data)} | correct={int(data.labels.sum())} | wrong={int((~data.labels).sum())}"
    )

    raw_split_labels = data.raw_sum_full % 10 if args.raw_sum_mod_10 else data.raw_sum_full
    if args.train_on_correct:
        correct_idx = np.flatnonzero(data.labels.astype(bool))
        wrong_idx = np.flatnonzero(~data.labels.astype(bool))
        if correct_idx.size < 2:
            raise RuntimeError("Too few correct samples for a train/validation split.")
        if wrong_idx.size == 0:
            raise RuntimeError("There are zero error samples; cannot build the test set.")

        val_size = 1.0 - float(args.correct_train_ratio)
        train_rel_idx, val_rel_idx, split_strategy = split_train_test_indices(
            y_all=raw_split_labels[correct_idx].astype(np.int64),
            y_binary=np.ones(correct_idx.size, dtype=np.int64),
            seed=args.seed,
            test_size=val_size,
            split_name="correct train/val split",
        )
        train_idx = correct_idx[train_rel_idx]
        val_idx = correct_idx[val_rel_idx]
        test_idx = wrong_idx
        train_data = data.take(train_idx)
        val_data = data.take(val_idx)
        test_data = data.take(test_idx)
        print(
            "Correct/Wrong split: "
            f"correct_rows={correct_idx.size} | wrong_rows={wrong_idx.size}"
        )
        print(
            "Train/Val/Test split: "
            f"train_rows={len(train_data)} ({np.unique(train_data.sample_ids).size} unique) | "
            f"val_rows={len(val_data)} ({np.unique(val_data.sample_ids).size} unique) | "
            f"test_rows={len(test_data)} ({np.unique(test_data.sample_ids).size} unique) | "
            f"strategy={split_strategy}"
        )
    else:
        train_idx, test_idx, split_strategy = split_train_test_indices(
            y_all=raw_split_labels.astype(np.int64),
            y_binary=data.labels.astype(np.int64),
            seed=args.seed,
            test_size=args.test_size,
            split_name="outer train/test split",
        )
        train_data = data.take(train_idx)
        test_data = data.take(test_idx)
        val_data = test_data
        print(
            "Train/Test split: "
            f"train_rows={len(train_data)} ({np.unique(train_data.sample_ids).size} unique) | "
            f"test_rows={len(test_data)} ({np.unique(test_data.sample_ids).size} unique) | "
            f"strategy={split_strategy}"
        )

    raw_X_layer = select_layer_features(raw_X_balanced, seq_len, feature_dim, args.layer)
    carry_X_layer = select_layer_features(carry_X_balanced, seq_len, feature_dim, args.layer)
    if raw_X_layer.shape != carry_X_layer.shape or not torch.allclose(raw_X_layer, carry_X_layer):
        warnings.warn("raw-sum and carry layer features are not identical; subsequent steps will use features returned by each external loader separately.")

    train_idx_t = torch.from_numpy(train_idx.astype(np.int64))
    val_idx_t = torch.from_numpy(val_idx.astype(np.int64)) if args.train_on_correct else None
    test_idx_t = torch.from_numpy(test_idx.astype(np.int64))
    pos_train = raw_position_indices[train_idx_t].long()
    pos_test = raw_position_indices[test_idx_t].long()
    pos_val = raw_position_indices[val_idx_t].long() if val_idx_t is not None else None

    X_raw_train = raw_X_layer[train_idx_t].float()
    X_raw_val = raw_X_layer[val_idx_t].float() if val_idx_t is not None else None
    X_raw_test = raw_X_layer[test_idx_t].float()
    y_raw_train = raw_y_balanced[train_idx_t].long()
    y_raw_val = raw_y_balanced[val_idx_t].long() if val_idx_t is not None else None
    y_raw_test = raw_y_balanced[test_idx_t].long()

    X_carry_train = carry_X_layer[train_idx_t].float()
    X_carry_val = carry_X_layer[val_idx_t].float() if val_idx_t is not None else None
    X_carry_test = carry_X_layer[test_idx_t].float()
    y_carry_train = carry_y_balanced[train_idx_t].float()
    y_carry_val = carry_y_balanced[val_idx_t].float() if val_idx_t is not None else None
    y_carry_test = carry_y_balanced[test_idx_t].float()

    raw_num_classes = 10 if args.raw_sum_mod_10 else int(torch.max(raw_y_balanced).item()) + 1

    raw_fit_train_idx_t = None
    raw_fit_val_idx_t = None
    carry_fit_train_idx_t = None
    carry_fit_val_idx_t = None
    if not args.train_on_correct:
        raw_early_stop_split_strategy = split_strategy
        carry_early_stop_split_strategy = split_strategy

    if args.train_on_correct:
        raw_early_stop_split_strategy = split_strategy
        carry_early_stop_split_strategy = split_strategy
        raw_fit_pool_rows = int(len(train_data) + len(val_data))
        carry_fit_pool_rows = int(len(train_data) + len(val_data))
        fit_sample_filter_mode = "correct_train_val__wrong_test"
        X_raw_fit_train = X_raw_train
        y_raw_fit_train = y_raw_train
        pos_raw_fit_train = pos_train
        X_raw_fit_val = X_raw_val
        y_raw_fit_val = y_raw_val
        pos_raw_fit_val = pos_val

        X_carry_fit_train = X_carry_train
        y_carry_fit_train = y_carry_train
        pos_carry_fit_train = pos_train
        X_carry_fit_val = X_carry_val
        y_carry_fit_val = y_carry_val
        pos_carry_fit_val = pos_val
        print(
            "Train-on-correct split: "
            f"correct_pool={raw_fit_pool_rows} | train_rows={len(X_raw_fit_train)} | val_rows={len(X_raw_fit_val)} | "
            f"wrong_test_rows={len(X_raw_test)} | strategy={raw_early_stop_split_strategy}"
        )
    else:
        X_raw_fit_train = X_raw_train
        y_raw_fit_train = y_raw_train
        pos_raw_fit_train = pos_train
        X_raw_fit_val = X_raw_test
        y_raw_fit_val = y_raw_test
        pos_raw_fit_val = pos_test

        X_carry_fit_train = X_carry_train
        y_carry_fit_train = y_carry_train
        pos_carry_fit_train = pos_train
        X_carry_fit_val = X_carry_test
        y_carry_fit_val = y_carry_test
        pos_carry_fit_val = pos_test

    if args.probe_mode == "mlp_direct":
        validate_digit_labels(train_data.gt_digits, "train_data.gt_digits")
        validate_digit_labels(val_data.gt_digits, "val_data.gt_digits")
        validate_digit_labels(test_data.gt_digits, "test_data.gt_digits")

        mlp_probe, mlp_val_metrics = train_mlp_direct_classifier(
            train_features=train_data.features,
            train_labels=train_data.gt_digits,
            val_features=val_data.features,
            val_labels=val_data.gt_digits,
            test_features=test_data.features,
            test_labels=test_data.gt_digits,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            split_strategy=split_strategy,
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.mlp_dropout,
        )
        corrected_test_digits = mlp_probe.test_preds
        error_rows = decompose_test_errors_mlp_direct(test_data=test_data, mlp_pred_digits=corrected_test_digits)
        error_type_stats = summarize_error_type_repairs(
            pred_digits=test_data.pred_digits,
            gt_digits=test_data.gt_digits,
            corrected_digits=corrected_test_digits,
            is_correct_mask=test_data.labels,
        )
        correction_metrics = compute_mlp_direct_correction_metrics(
            corrected_digits=corrected_test_digits,
            pred_orig_digits=test_data.pred_digits,
            gt_digits=test_data.gt_digits,
        )

        test_count = len(test_data)
        incorrect_test_count = int((~test_data.labels).sum())
        if int(error_type_stats["orig_error_count"]) != int(error_type_stats["off_by_one_count"]) + int(error_type_stats["other_error_count"]):
            raise RuntimeError("Inconsistent mlp_direct error stats: orig_error_count != off_by_one_count + other_error_count")

        summary = {
            "task": "probe_error_decomposition",
            "probe_mode": args.probe_mode,
            "target_h5": str(args.h5),
            "sample_reference_source": sample_reference_source,
            "split_source": "generated_in_memory",
            "target_pos": int(args.pos),
            "layer": int(args.layer),
            "input_pos": "consistent",
            "feature_type": "post_ffn",
            "pooling_type": "none",
            "sample_filter_mode": args.sample_filter,
            "balance_mode": effective_balance_mode,
            "balance_target_classes": args.balance_target_classes,
            "filter_by_result_len": True,
            "raw_sum_mod_10": bool(args.raw_sum_mod_10),
            "inertia_delta": float(args.inertia_delta),
            "train_on_correct": bool(args.train_on_correct),
            "test_size": None if args.train_on_correct else args.test_size,
            "correct_train_ratio": float(args.correct_train_ratio),
            "correct_val_ratio": float(1.0 - args.correct_train_ratio),
            "seed": int(args.seed),
            "device": str(device),
            "load_stats": {
                "rows_total": int(total_rows),
                "rows_missing_sample_meta": 0,
                "rows_pos_out_of_range": 0,
                "rows_invalid_raw_sum": 0,
            },
            "alignment": {
                "gt_char_mismatch_dropped": mismatch_count,
                "target_result_len": int(target_result_len),
                "result_len_dropped": int(dropped_by_result_len),
                "pre_balance_rows": int(pre_balance_rows),
                "pre_balance_correct": int(pre_balance_correct),
                "pre_balance_wrong": int(pre_balance_wrong),
                "balanced_rows": int(len(data)),
                "balanced_unique_sample_ids": int(np.unique(data.sample_ids).size),
            },
            "splits": {
                "all": summarize_split(data),
                "train": summarize_split(train_data),
                "val": summarize_split(val_data),
                "test": summarize_split(test_data),
            },
            "probe_training": {
                "train_on_correct": bool(args.train_on_correct),
                "fit_sample_filter_mode": fit_sample_filter_mode,
                "split_mode": "correct_train_val__wrong_test" if args.train_on_correct else "train_test",
                "correct_train_ratio": float(args.correct_train_ratio) if args.train_on_correct else None,
                "correct_val_ratio": float(1.0 - args.correct_train_ratio) if args.train_on_correct else None,
                "train_rows": int(len(train_data)),
                "val_rows": int(len(val_data)),
                "test_rows": int(len(test_data)),
                "outer_train_rows": int(len(train_data)),
                "outer_train_correct_rows": int(train_data.labels.sum()),
                "outer_train_wrong_rows": int((~train_data.labels).sum()),
                "outer_test_rows": int(len(test_data)),
                "mlp_early_stop_split_strategy": split_strategy,
                "mlp_fit_train_rows": int(len(train_data)),
                "mlp_fit_val_rows": int(len(val_data)),
            },
            "mlp_direct_probe": {
                "target": "gt_digit_classify",
                "num_classes": int(mlp_probe.num_classes),
                "train_test_split_strategy": mlp_probe.split_strategy,
                "early_stop_split_strategy": split_strategy,
                "trained_epochs": int(mlp_probe.trained_epochs),
                "best_metric_name": mlp_probe.best_metric_name,
                "best_test_metric": safe_float(mlp_probe.best_test_metric),
                "train": {
                    "loss": safe_float(mlp_probe.train_metrics["loss"]),
                    "acc": safe_float(mlp_probe.train_metrics["acc"]),
                    "auc": safe_float(mlp_probe.train_metrics["auc"]),
                },
                "val": {
                    "loss": safe_float(mlp_val_metrics["loss"]),
                    "acc": safe_float(mlp_val_metrics["acc"]),
                    "auc": safe_float(mlp_val_metrics["auc"]),
                },
                "test": {
                    "loss": safe_float(mlp_probe.test_metrics["loss"]),
                    "acc": safe_float(mlp_probe.test_metrics["acc"]),
                    "auc": safe_float(mlp_probe.test_metrics["auc"]),
                },
            },
            "mlp_error_repair": {
                "test_count": int(test_count),
                "incorrect_test_count": incorrect_test_count,
                "orig_error_count": int(error_type_stats["orig_error_count"]),
                "correct_test_count": int(np.sum(test_data.labels.astype(np.int64))),
                "off_by_one_count": int(error_type_stats["off_by_one_count"]),
                "other_error_count": int(error_type_stats["other_error_count"]),
                "off_by_one_ratio": float(error_type_stats["off_by_one_ratio"]),
                "other_error_ratio": float(error_type_stats["other_error_ratio"]),
                "off_by_one_fix_count": int(error_type_stats["off_by_one_fix_count"]),
                "off_by_one_fix_rate": float(error_type_stats["off_by_one_fix_rate"]),
                "other_error_fix_count": int(error_type_stats["other_error_fix_count"]),
                "other_error_fix_rate": float(error_type_stats["other_error_fix_rate"]),
                "modified_rate": float(correction_metrics["modified_rate"]),
                "tp_correction": float(correction_metrics["tp_correction"]),
                "fp_preservation": float(correction_metrics["fp_preservation"]),
            },
        }

        json_path = outdir / f"probe_error_decomposition_pos{args.pos}_mlp_direct.json"
        csv_path = outdir / f"probe_error_decomposition_pos{args.pos}_mlp_direct.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        write_csv(error_rows, csv_path)

        print(f"Saved json to: {json_path}")
        print(f"Saved csv to:  {csv_path}")
        print(
            "Summary: "
            f"probe_mode={args.probe_mode} | "
            f"mlp_val_acc={0.0 if mlp_val_metrics['acc'] is None else mlp_val_metrics['acc']:.4f} | "
            f"mlp_test_acc={mlp_probe.test_metrics['acc']:.4f} | "
            f"off_by_one={error_type_stats['off_by_one_count']} ({error_type_stats['off_by_one_ratio']:.4f}) | "
            f"off_by_one_fixed={error_type_stats['off_by_one_fix_count']} ({error_type_stats['off_by_one_fix_rate']:.4f}) | "
            f"other_error={error_type_stats['other_error_count']} ({error_type_stats['other_error_ratio']:.4f}) | "
            f"other_fixed={error_type_stats['other_error_fix_count']} ({error_type_stats['other_error_fix_rate']:.4f}) | "
            f"modified_rate={correction_metrics['modified_rate']:.4f} | "
            f"tp_correction={correction_metrics['tp_correction']:.4f} | "
            f"fp_preservation={correction_metrics['fp_preservation']:.4f}"
        )
        return

    raw_probe = train_classifier(
        train_features=X_raw_fit_train.detach().cpu().numpy().astype(np.float32),
        train_labels=y_raw_fit_train.detach().cpu().numpy().astype(np.int64),
        test_features=X_raw_fit_val.detach().cpu().numpy().astype(np.float32),
        test_labels=y_raw_fit_val.detach().cpu().numpy().astype(np.int64),
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        split_strategy=raw_early_stop_split_strategy,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    )
    raw_train_metrics_raw = evaluate_classifier(
        raw_probe.model,
        X_raw_train.detach().cpu().numpy().astype(np.float32),
        y_raw_train.detach().cpu().numpy().astype(np.int64),
        args.batch_size,
        device,
        raw_num_classes,
    )
    raw_val_metrics_raw = evaluate_classifier(
        raw_probe.model,
        X_raw_fit_val.detach().cpu().numpy().astype(np.float32),
        y_raw_fit_val.detach().cpu().numpy().astype(np.int64),
        args.batch_size,
        device,
        raw_num_classes,
    )
    raw_test_metrics_raw = evaluate_classifier(
        raw_probe.model,
        X_raw_test.detach().cpu().numpy().astype(np.float32),
        y_raw_test.detach().cpu().numpy().astype(np.int64),
        args.batch_size,
        device,
        raw_num_classes,
    )
    raw_train_metrics = {
        "loss": float(raw_train_metrics_raw["loss"]),
        "acc": float(raw_train_metrics_raw["acc"]),
        "auc": safe_float(raw_train_metrics_raw["auc"]),
    }
    raw_val_metrics = {
        "loss": float(raw_val_metrics_raw["loss"]),
        "acc": float(raw_val_metrics_raw["acc"]),
        "auc": safe_float(raw_val_metrics_raw["auc"]),
    }
    raw_test_metrics = {
        "loss": float(raw_test_metrics_raw["loss"]),
        "acc": float(raw_test_metrics_raw["acc"]),
        "auc": safe_float(raw_test_metrics_raw["auc"]),
    }
    raw_probe.train_metrics = raw_train_metrics
    raw_probe.test_metrics = raw_test_metrics
    raw_probe.test_preds = np.asarray(raw_test_metrics_raw["preds"], dtype=np.int64)
    raw_probe.test_probs = np.asarray(raw_test_metrics_raw["probs"], dtype=np.float32)

    carry_probe = train_regressor(
        train_features=X_carry_fit_train.detach().cpu().numpy().astype(np.float32),
        train_labels=y_carry_fit_train.detach().cpu().numpy().astype(np.float32),
        test_features=X_carry_fit_val.detach().cpu().numpy().astype(np.float32),
        test_labels=y_carry_fit_val.detach().cpu().numpy().astype(np.float32),
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        split_strategy=carry_early_stop_split_strategy,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    )
    carry_train_metrics_raw = evaluate_regressor(
        carry_probe.model,
        X_carry_train.detach().cpu().numpy().astype(np.float32),
        y_carry_train.detach().cpu().numpy().astype(np.float32),
        args.batch_size,
        device,
    )
    carry_val_metrics_raw = evaluate_regressor(
        carry_probe.model,
        X_carry_fit_val.detach().cpu().numpy().astype(np.float32),
        y_carry_fit_val.detach().cpu().numpy().astype(np.float32),
        args.batch_size,
        device,
    )
    carry_test_metrics_raw = evaluate_regressor(
        carry_probe.model,
        X_carry_test.detach().cpu().numpy().astype(np.float32),
        y_carry_test.detach().cpu().numpy().astype(np.float32),
        args.batch_size,
        device,
    )
    carry_train_metrics = {
        "loss": float(carry_train_metrics_raw["loss"]),
        "mae": float(carry_train_metrics_raw["mae"]),
        "floor_acc": float(carry_train_metrics_raw["floor_acc"]),
    }
    carry_val_metrics = {
        "loss": float(carry_val_metrics_raw["loss"]),
        "mae": float(carry_val_metrics_raw["mae"]),
        "floor_acc": float(carry_val_metrics_raw["floor_acc"]),
    }
    carry_test_metrics = {
        "loss": float(carry_test_metrics_raw["loss"]),
        "mae": float(carry_test_metrics_raw["mae"]),
        "floor_acc": float(carry_test_metrics_raw["floor_acc"]),
    }
    carry_probe.train_metrics = carry_train_metrics
    carry_probe.test_metrics = carry_test_metrics
    carry_probe.test_preds = np.asarray(carry_test_metrics_raw["preds"], dtype=np.float32)

    raw_val_acc = safe_float(raw_val_metrics["acc"])
    carry_val_floor_acc = safe_float(carry_val_metrics["floor_acc"])
    raw_test_mod10_acc = float(np.mean((raw_probe.test_preds % 10) == (test_data.raw_sum_full % 10)))
    error_rows, bucket_counts, corrected_test_digits = decompose_test_errors(
        test_data=test_data,
        raw_pred_labels=raw_probe.test_preds,
        carry_preds_phi=carry_probe.test_preds,
        raw_sum_mod_10=args.raw_sum_mod_10,
        inertia_delta=args.inertia_delta,
    )
    error_type_stats = summarize_error_type_repairs(
        pred_digits=test_data.pred_digits,
        gt_digits=test_data.gt_digits,
        corrected_digits=corrected_test_digits,
        is_correct_mask=test_data.labels,
    )
    error_type_bucket_stats = summarize_error_type_bucket_stats(error_rows)

    incorrect_test_count = int((~test_data.labels).sum())
    decomposed_incorrect_count = int(sum(bucket_counts.get(bucket, 0) for bucket in BUCKET_ORDER))
    if decomposed_incorrect_count != incorrect_test_count:
        raise RuntimeError(
            f"Inconsistent four-way error bucket counts: bucket_sum={decomposed_incorrect_count}, incorrect_test_count={incorrect_test_count}"
        )
    off_by_one_bucket_count = int(
        sum(error_type_bucket_stats["off_by_one_bucket_stats"][bucket]["count"] for bucket in BUCKET_ORDER)
    )
    other_error_bucket_count = int(
        sum(error_type_bucket_stats["other_error_bucket_stats"][bucket]["count"] for bucket in BUCKET_ORDER)
    )
    if off_by_one_bucket_count != int(error_type_stats["off_by_one_count"]):
        raise RuntimeError(
            f"Inconsistent off_by_one bucket stats: bucket_sum={off_by_one_bucket_count}, off_by_one_count={error_type_stats['off_by_one_count']}"
        )
    if other_error_bucket_count != int(error_type_stats["other_error_count"]):
        raise RuntimeError(
            f"Inconsistent other_error bucket stats: bucket_sum={other_error_bucket_count}, other_error_count={error_type_stats['other_error_count']}"
        )
    test_count = len(test_data)
    bucket_pct_of_test = {}
    bucket_pct_of_incorrect = {}
    for bucket in BUCKET_ORDER:
        bucket_pct_of_test[bucket] = float(bucket_counts.get(bucket, 0) / test_count) if test_count > 0 else 0.0
        denom = incorrect_test_count if incorrect_test_count > 0 else 1
        bucket_pct_of_incorrect[bucket] = float(bucket_counts.get(bucket, 0) / denom)

    if (
        not args.raw_sum_mod_10
        and "plus_num3len10" in str(args.h5).lower()
        and args.pos == 4
        and raw_probe.num_classes != 28
    ):
        warnings.warn(
            f"The filtered data implies raw-sum class count {raw_probe.num_classes}，"
            "which differs from the expected 28 classes; check the H5 file or filters."
        )

    summary = {
        "task": "probe_error_decomposition",
        "probe_mode": args.probe_mode,
        "target_h5": str(args.h5),
        "sample_reference_source": sample_reference_source,
        "split_source": "generated_in_memory",
        "target_pos": int(args.pos),
        "layer": int(args.layer),
        "input_pos": "consistent",
        "feature_type": "post_ffn",
        "pooling_type": "none",
        "sample_filter_mode": args.sample_filter,
        "balance_mode": effective_balance_mode,
        "balance_target_classes": args.balance_target_classes,
        "filter_by_result_len": True,
        "raw_sum_mod_10": bool(args.raw_sum_mod_10),
        "inertia_delta": float(args.inertia_delta),
        "train_on_correct": bool(args.train_on_correct),
        "test_size": None if args.train_on_correct else args.test_size,
        "correct_train_ratio": float(args.correct_train_ratio),
        "correct_val_ratio": float(1.0 - args.correct_train_ratio),
        "seed": int(args.seed),
        "device": str(device),
        "load_stats": {
            "rows_total": int(total_rows),
            "rows_missing_sample_meta": 0,
            "rows_pos_out_of_range": 0,
            "rows_invalid_raw_sum": 0,
        },
        "alignment": {
            "gt_char_mismatch_dropped": mismatch_count,
            "target_result_len": int(target_result_len),
            "result_len_dropped": int(dropped_by_result_len),
            "pre_balance_rows": int(pre_balance_rows),
            "pre_balance_correct": int(pre_balance_correct),
            "pre_balance_wrong": int(pre_balance_wrong),
            "balanced_rows": int(len(data)),
            "balanced_unique_sample_ids": int(np.unique(data.sample_ids).size),
        },
        "splits": {
            "all": summarize_split(data),
            "train": summarize_split(train_data),
            "val": summarize_split(val_data),
            "test": summarize_split(test_data),
        },
        "probe_training": {
            "train_on_correct": bool(args.train_on_correct),
            "fit_sample_filter_mode": fit_sample_filter_mode,
            "split_mode": "correct_train_val__wrong_test" if args.train_on_correct else "train_test",
            "correct_train_ratio": float(args.correct_train_ratio) if args.train_on_correct else None,
            "correct_val_ratio": float(1.0 - args.correct_train_ratio) if args.train_on_correct else None,
            "train_rows": int(len(train_data)),
            "val_rows": int(len(val_data)),
            "test_rows": int(len(test_data)),
            "outer_train_rows": int(len(train_data)),
            "outer_train_correct_rows": int(train_data.labels.sum()),
            "outer_train_wrong_rows": int((~train_data.labels).sum()),
            "outer_test_rows": int(len(test_data)),
            "raw_early_stop_split_strategy": raw_early_stop_split_strategy,
            "carry_early_stop_split_strategy": carry_early_stop_split_strategy,
            "raw_fit_pool_rows": None if raw_fit_pool_rows is None else int(raw_fit_pool_rows),
            "carry_fit_pool_rows": None if carry_fit_pool_rows is None else int(carry_fit_pool_rows),
            "raw_fit_train_rows": int(len(X_raw_fit_train)),
            "raw_fit_val_rows": int(len(X_raw_fit_val)),
            "carry_fit_train_rows": int(len(X_carry_fit_train)),
            "carry_fit_val_rows": int(len(X_carry_fit_val)),
        },
        "raw_sum_probe": {
            "target": "raw_sum_classify",
            "raw_sum_mod_10": bool(args.raw_sum_mod_10),
            "num_classes": int(raw_probe.num_classes),
            "train_test_split_strategy": raw_probe.split_strategy,
            "early_stop_split_strategy": raw_early_stop_split_strategy,
            "trained_epochs": int(raw_probe.trained_epochs),
            "best_metric_name": raw_probe.best_metric_name,
            "best_test_metric": safe_float(raw_probe.best_test_metric),
            "train": {
                "loss": safe_float(raw_probe.train_metrics["loss"]),
                "acc": safe_float(raw_probe.train_metrics["acc"]),
                "auc": safe_float(raw_probe.train_metrics["auc"]),
            },
            "val": {
                "loss": safe_float(raw_val_metrics["loss"]),
                "acc": raw_val_acc,
                "auc": safe_float(raw_val_metrics["auc"]),
            },
            "test": {
                "loss": safe_float(raw_probe.test_metrics["loss"]),
                "acc": safe_float(raw_probe.test_metrics["acc"]),
                "auc": safe_float(raw_probe.test_metrics["auc"]),
                "mod10_acc": safe_float(raw_test_mod10_acc),
            },
        },
        "carry_probe": {
            "target": "C_potential",
            "train_test_split_strategy": carry_probe.split_strategy,
            "early_stop_split_strategy": carry_early_stop_split_strategy,
            "trained_epochs": int(carry_probe.trained_epochs),
            "best_metric_name": carry_probe.best_metric_name,
            "best_test_metric": safe_float(carry_probe.best_test_metric),
            "train": {
                "loss": safe_float(carry_probe.train_metrics["loss"]),
                "mae": safe_float(carry_probe.train_metrics["mae"]),
                "floor_acc": safe_float(carry_probe.train_metrics["floor_acc"]),
            },
            "val": {
                "loss": safe_float(carry_val_metrics["loss"]),
                "mae": safe_float(carry_val_metrics["mae"]),
                "floor_acc": carry_val_floor_acc,
            },
            "test": {
                "loss": safe_float(carry_probe.test_metrics["loss"]),
                "mae": safe_float(carry_probe.test_metrics["mae"]),
                "floor_acc": safe_float(carry_probe.test_metrics["floor_acc"]),
            },
        },
        "error_decomposition": {
            "inertia_delta": float(args.inertia_delta),
            "test_count": int(test_count),
            "incorrect_test_count": incorrect_test_count,
            "orig_error_count": int(error_type_stats["orig_error_count"]),
            "correct_test_count": int(bucket_counts.get("correct", 0)),
            "off_by_one_count": int(error_type_stats["off_by_one_count"]),
            "other_error_count": int(error_type_stats["other_error_count"]),
            "off_by_one_ratio": float(error_type_stats["off_by_one_ratio"]),
            "other_error_ratio": float(error_type_stats["other_error_ratio"]),
            "off_by_one_fix_count": int(error_type_stats["off_by_one_fix_count"]),
            "off_by_one_fix_rate": float(error_type_stats["off_by_one_fix_rate"]),
            "other_error_fix_count": int(error_type_stats["other_error_fix_count"]),
            "other_error_fix_rate": float(error_type_stats["other_error_fix_rate"]),
            "off_by_one_bucket_stats": error_type_bucket_stats["off_by_one_bucket_stats"],
            "other_error_bucket_stats": error_type_bucket_stats["other_error_bucket_stats"],
            "bucket_counts": {k: int(bucket_counts.get(k, 0)) for k in BUCKET_ORDER},
            "bucket_pct_of_test": bucket_pct_of_test,
            "bucket_pct_of_incorrect": bucket_pct_of_incorrect,
        },
    }

    json_path = outdir / f"probe_error_decomposition_pos{args.pos}.json"
    csv_path = outdir / f"probe_error_decomposition_pos{args.pos}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(error_rows, csv_path)

    print(f"Saved json to: {json_path}")
    print(f"Saved csv to:  {csv_path}")
    print(
        "Summary: "
        f"inertia_delta={args.inertia_delta:.4f} | "
        f"raw_val_acc={0.0 if raw_val_acc is None else raw_val_acc:.4f} | "
        f"raw_test_acc={raw_probe.test_metrics['acc']:.4f} | "
        f"raw_test_mod10_acc={raw_test_mod10_acc:.4f} | "
        f"carry_val_floor_acc={0.0 if carry_val_floor_acc is None else carry_val_floor_acc:.4f} | "
        f"carry_test_floor_acc={carry_probe.test_metrics['floor_acc']:.4f} | "
        f"both_correct={bucket_counts.get('both_correct', 0)} ({bucket_pct_of_test.get('both_correct', 0.0):.4f}) | "
        f"raw_correct_carry_wrong={bucket_counts.get('raw_correct_carry_wrong', 0)} ({bucket_pct_of_test.get('raw_correct_carry_wrong', 0.0):.4f}) | "
        f"raw_wrong_carry_correct={bucket_counts.get('raw_wrong_carry_correct', 0)} ({bucket_pct_of_test.get('raw_wrong_carry_correct', 0.0):.4f}) | "
        f"both_wrong={bucket_counts.get('both_wrong', 0)} ({bucket_pct_of_test.get('both_wrong', 0.0):.4f}) | "
        f"off_by_one={error_type_stats['off_by_one_count']} ({error_type_stats['off_by_one_ratio']:.4f}) | "
        f"off_by_one_fixed={error_type_stats['off_by_one_fix_count']} ({error_type_stats['off_by_one_fix_rate']:.4f}) | "
        f"other_error={error_type_stats['other_error_count']} ({error_type_stats['other_error_ratio']:.4f}) | "
        f"other_fixed={error_type_stats['other_error_fix_count']} ({error_type_stats['other_error_fix_rate']:.4f})"
    )
    print(format_error_bucket_stats("Off-by-one bucket stats", error_type_bucket_stats["off_by_one_bucket_stats"]))
    print(format_error_bucket_stats("Other-error bucket stats", error_type_bucket_stats["other_error_bucket_stats"]))


if __name__ == "__main__":
    main()
