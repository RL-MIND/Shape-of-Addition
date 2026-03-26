import argparse
import csv
import importlib.util
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

from model_utils import detect_model_type, get_norm_weight_from_model, rms_norm
from probe_data import compute_c_potential, compute_raw_sum, load_dataset


DEFAULT_H5 = Path("results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5")
DEFAULT_OUTDIR = Path("results/error_decomposition")
DEFAULT_POS = 4
DEFAULT_LAYER = 36
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 20
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CORRECT_TRAIN_RATIO = 0.8
CURRENT_SAMPLE_FILTER_MODE = "all"
CURRENT_BALANCE_MODE = "normal"
CURRENT_BALANCE_TARGET_CLASSES = False
DEFAULT_MLP_HIDDEN_DIM = 512
DEFAULT_MLP_DROPOUT = 0.2
EXTERNAL_VERTICAL_FLOW_DIR = Path("/home/wenliuyuan/vertical-flow")

BUCKET_ORDER = [
    "carry_wrong_raw_correct",
    "raw_wrong_carry_correct",
    "both_wrong",
    "unexplained_probe_failure",
    "internal_anomaly",
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


class ProbeMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProbeMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
        warnings.warn("CUDA 不可用，自动回退到 CPU。")
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
        raise RuntimeError("H5 不包含 samples/question|gt|sample_idx，且未提供 --dataset 回退。")

    dataset = load_dataset(dataset_path)
    meta = {}
    for sample_id, operands in enumerate(dataset):
        question = " + ".join(str(x) for x in operands)
        meta[sample_id] = {"question": question, "gt": str(sum(operands))}
    return meta, "dataset_pickle"


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
            raise RuntimeError("H5 缺少 all_token_results 组。")

        pos_name = f"pos_{pos}"
        if pos_name not in all_results:
            raise RuntimeError(f"H5 中不存在 {pos_name}。")
        pos_group = all_results[pos_name]

        flows_ds = pos_group.get("flows")
        if flows_ds is None:
            raise RuntimeError(f"{pos_name} 缺少 flows 数据集。")
        if flows_ds.ndim != 3:
            raise RuntimeError(f"{pos_name}/flows 期望 3 维，实际为 {flows_ds.ndim}。")
        if not (0 <= layer < flows_ds.shape[1]):
            raise RuntimeError(f"请求 layer={layer}，但 flows 只有 {flows_ds.shape[1]} 层。")

        features = np.asarray(flows_ds[:, layer, :], dtype=np.float32)
        labels = np.asarray(pos_group["labels"][:], dtype=np.bool_)
        pred_tokens = decode_to_str_list(pos_group["preds"][:])
        gt_chars_h5 = decode_to_str_list(pos_group["gt_chars"][:])

        sample_ds = pos_group.get("sample_indices")
        if sample_ds is None:
            sample_ds = pos_group.get("sample_ids")
        if sample_ds is None:
            sample_ids = np.arange(features.shape[0], dtype=np.int64)
        else:
            sample_ids = np.asarray(sample_ds[:], dtype=np.int64)

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
        raise RuntimeError("没有构建出任何有效样本，请检查 H5 与样本元数据是否匹配。")

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


def _load_module_from_file(module_name: str, module_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path and Path(existing_path).resolve() == module_path.resolve():
            return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法从 {module_path} 加载模块 {module_name}。")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_vertical_flow_modules():
    external_dir = EXTERNAL_VERTICAL_FLOW_DIR
    if not external_dir.exists():
        raise RuntimeError(f"未找到 external vertical-flow 目录: {external_dir}")
    external_path = str(external_dir)
    if external_path not in sys.path:
        sys.path.insert(0, external_path)

    import builtins

    builtin_print = builtins.print
    _load_module_from_file("models", external_dir / "models.py")
    flow_utils_module = _load_module_from_file("flow_utils", external_dir / "flow_utils.py")
    classifier_module = _load_module_from_file("classifier", external_dir / "classifier.py")
    builtins.print = builtin_print
    return flow_utils_module, classifier_module


def install_flow_utils_sample_meta_override(
    flow_utils_module,
    sample_meta: Dict[int, Dict[str, str]],
) -> None:
    cached_meta = {
        int(sample_id): {
            "question": str(meta["question"]),
            "gt": str(meta["gt"]),
        }
        for sample_id, meta in sample_meta.items()
    }

    def _load_samples_meta_override(_h5_path):
        return {
            int(sample_id): {
                "question": meta["question"],
                "gt": meta["gt"],
            }
            for sample_id, meta in cached_meta.items()
        }

    flow_utils_module.load_samples_meta = _load_samples_meta_override


def configure_vertical_flow_classifier(
    classifier_module,
    device: torch.device,
    batch_size: int,
    seed: int,
    lr: float,
    weight_decay: float,
    patience: int,
    train_target: str,
    raw_sum_mod_10: bool = False,
) -> None:
    classifier_module.DEVICE = device
    classifier_module.SEED = seed
    classifier_module.BATCH_SIZE = batch_size
    classifier_module.MODEL_TYPE = "mlp"
    classifier_module.MLP_HIDDEN_DIM = DEFAULT_MLP_HIDDEN_DIM
    classifier_module.MLP_DROPOUT = DEFAULT_MLP_DROPOUT
    classifier_module.LEARNING_RATE = lr
    classifier_module.WEIGHT_DECAY = weight_decay
    classifier_module.EARLY_STOP_PATIENCE = patience
    classifier_module.BEST_METRIC = "auc"
    classifier_module.TRAIN_TARGET = train_target
    classifier_module.RAW_SUM_MOD_10 = raw_sum_mod_10
    classifier_module.SAVE_MODEL = False
    classifier_module.CUMULATIVE_LAYERS = False
    classifier_module.SPECIFIC_LAYER_INDEX = None


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
    flow_utils_module,
    h5_path: Path,
    pos: int,
    model_path: str,
    seed: int,
    train_target: str,
    balance_mode: str,
    sample_filter_mode: Optional[str] = None,
    raw_sum_mod_10: bool = False,
    loaded_model_obj=None,
):
    if sample_filter_mode is None:
        sample_filter_mode = CURRENT_SAMPLE_FILTER_MODE
    return flow_utils_module.load_and_process_data(
        file_path=h5_path,
        position_select=pos,
        feature_type="flows",
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
        balance_target_classes=CURRENT_BALANCE_TARGET_CLASSES,
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
        raise RuntimeError("raw-sum 与 carry 的样本顺序不一致，无法共享同一套错误分解。")

    raw_positions = np.asarray(raw_processed[3].cpu().numpy(), dtype=np.int64)
    carry_positions = np.asarray(carry_processed[3].cpu().numpy(), dtype=np.int64)
    if not np.array_equal(raw_positions, carry_positions):
        raise RuntimeError("raw-sum 与 carry 的 position_indices 不一致，无法共享同一套切分。")


def select_layer_features(X_all: torch.Tensor, seq_len: int, feature_dim: int, layer: int) -> torch.Tensor:
    if not (0 <= layer < seq_len):
        raise RuntimeError(f"请求 layer={layer}，但 external flow seq_len={seq_len}。")
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
            raise RuntimeError(f"external flow 返回的 sample_id={sample_id} 在 H5 samples 中不存在。")
        question = sample_record["question"]
        gt = sample_record["gt"]
        if pos >= len(gt):
            raise RuntimeError(f"sample_id={sample_id} 的 gt 长度不足以访问 pos={pos}。")
        raw_sum_value = compute_raw_sum(question, pos)
        if raw_sum_value < 0:
            raise RuntimeError(f"sample_id={sample_id} 在 pos={pos} 上无法重建 raw_sum。")
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
        if isinstance(value, np.ndarray):
            out[key] = value[idx]
        elif isinstance(value, list):
            out[key] = [value[i] for i in idx.tolist()]
        else:
            out[key] = value
    return out


def make_external_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(features, labels, positions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_with_external_classifier(
    classifier_module,
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    is_regression: bool,
    train_target: str,
    batch_size: int,
) -> Dict[str, float]:
    loader = make_external_loader(features, labels, positions, batch_size)
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    metrics = classifier_module.evaluate(
        model,
        loader,
        criterion,
        is_regression=is_regression,
        train_target=train_target,
    )
    if is_regression:
        return {
            "loss": float(metrics["loss"]),
            "mae": float(metrics["mae"]),
            "floor_acc": float(metrics["acc"]),
        }
    return {
        "loss": float(metrics["loss"]),
        "acc": float(metrics["acc"]),
        "auc": safe_float(metrics["auc"]),
    }


def predict_with_external_classifier(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    preds_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
    return (
        np.concatenate(preds_all, axis=0).astype(np.int64),
        np.concatenate(probs_all, axis=0).astype(np.float32),
    )


def predict_with_external_regressor(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    preds_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            preds = model(batch_x.to(device)).squeeze(-1)
            preds_all.append(preds.cpu().numpy())
    return np.concatenate(preds_all, axis=0).astype(np.float32)


def pick_target_result_len(result_lens: np.ndarray) -> int:
    counter = Counter(int(v) for v in result_lens.tolist())
    return counter.most_common(1)[0][0]


def apply_final_norm(
    features: np.ndarray,
    model_path: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model_type = detect_model_type(model_path)
    add_unit_offset = model_type == "gemma3"
    norm_weight, norm_eps = get_norm_weight_from_model(model_path, device, model_type=model_type)
    norm_weight = norm_weight.to(device)

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            batch = rms_norm(batch, norm_weight, eps=norm_eps, add_unit_offset=add_unit_offset)
            outputs.append(batch.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def balance_indices_numpy(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return np.arange(len(labels), dtype=np.int64)

    min_count = counts.min()
    kept_indices: List[int] = []
    for cls in unique:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) > min_count:
            selected = rng.choice(cls_indices, size=min_count, replace=False)
        else:
            selected = cls_indices
        kept_indices.extend(selected.tolist())

    rng.shuffle(kept_indices)
    return np.asarray(kept_indices, dtype=np.int64)


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
        warnings.warn(f"{split_name} 按 y_all 分层切分失败，改为 y_binary: {exc}")
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y_binary,
        )
        return train_idx, test_idx, "y_binary"
    except ValueError as exc:
        warnings.warn(f"{split_name} 按 y_binary 分层切分失败，改为非分层切分: {exc}")
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
) -> ClassifierResult:
    num_classes = int(max(train_labels.max(), test_labels.max())) + 1
    model = ProbeMLPClassifier(
        train_features.shape[1],
        num_classes,
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
        dropout=DEFAULT_MLP_DROPOUT,
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
) -> RegressorResult:
    model = ProbeMLPRegressor(
        train_features.shape[1],
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
        dropout=DEFAULT_MLP_DROPOUT,
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
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    counts = {bucket: 0 for bucket in BUCKET_ORDER}
    counts["correct"] = 0

    raw_gt_full = test_data.raw_sum_full
    raw_gt_mod = raw_gt_full % 10
    raw_gt_labels = raw_gt_mod if raw_sum_mod_10 else raw_gt_full
    raw_hat_labels = np.asarray(raw_pred_labels, dtype=np.int64)
    raw_hat_mod = raw_hat_labels % 10
    carry_gt = np.floor(np.maximum(test_data.c_potential, 0.0)).astype(np.int64)
    carry_hat = np.floor(np.maximum(carry_preds_phi, 0.0)).astype(np.int64)

    for idx in range(len(test_data)):
        is_correct = bool(test_data.labels[idx])
        pred_digit = int(test_data.pred_digits[idx])
        explained_digit = int((raw_hat_mod[idx] + carry_hat[idx]) % 10)

        if is_correct:
            bucket = "correct"
        else:
            probe_explained = pred_digit >= 0 and pred_digit == explained_digit
            if not probe_explained:
                bucket = "unexplained_probe_failure"
            else:
                raw_ok = int(raw_hat_labels[idx]) == int(raw_gt_labels[idx])
                carry_ok = int(carry_hat[idx]) == int(carry_gt[idx])
                if raw_ok and carry_ok:
                    bucket = "internal_anomaly"
                elif raw_ok and not carry_ok:
                    bucket = "carry_wrong_raw_correct"
                elif (not raw_ok) and carry_ok:
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
                "bucket": bucket,
            }
        )

    return rows, counts


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise RuntimeError("没有可写入 CSV 的测试明细。")
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
        raise RuntimeError(f"{split_name} 开启 train_on_correct 后，正确样本不足以再切分训练/验证集。")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train raw-sum and carry probes from H5 samples, then decompose test errors into four buckets.",
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--dataset", type=Path, default=Path("num3len10-10000.pkl"), help="仅在 H5 缺少 samples 元数据时作为回退。")
    parser.add_argument("--pos", type=int, default=DEFAULT_POS)
    parser.add_argument("--model", type=str, default="/data/Models/Qwen3-4b", help="用于提取 final norm 参数的模型路径。")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--raw-sum-mod-10", action="store_true", help="将 raw-sum probe 改为 10 分类（raw_sum mod 10）。")
    parser.add_argument("--train-on-correct", action="store_true", help="按 Y:/vertical-flow/classifier.py 的口径，仅使用 correct-only 样本池切分 train/val 并训练 probe。")
    parser.add_argument("--correct-train-ratio", type=float, default=DEFAULT_CORRECT_TRAIN_RATIO, help="开启 --train-on-correct 时，正确样本中分给 train 的比例，其余进入 val。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not (0.0 < args.correct_train_ratio < 1.0):
        raise ValueError("--correct-train-ratio 必须在 (0, 1) 区间内。")

    set_global_seed(args.seed)
    device = maybe_get_device(args.device)
    outdir = normalize_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading position data from {args.h5} (pos={args.pos}, layer={args.layer})...")
    flow_utils_module, classifier_module = load_vertical_flow_modules()
    with h5py.File(args.h5, "r") as h5f:
        sample_meta, sample_reference_source = load_sample_reference(h5f, args.dataset)
    if not sample_meta:
        raise RuntimeError("目标 H5 缺少可用样本元数据；请提供 --dataset 作为 sample_id -> question/gt 回退。")
    install_flow_utils_sample_meta_override(flow_utils_module, sample_meta)
    print(f"Using sample metadata source: {sample_reference_source}")

    print(f"Applying final norm using model: {args.model}")
    loaded_model_obj = load_probe_model(args.model)
    try:
        raw_unbalanced = load_processed_task_data(
            flow_utils_module=flow_utils_module,
            h5_path=args.h5,
            pos=args.pos,
            model_path=args.model,
            seed=args.seed,
            train_target="raw_sum_classify",
            balance_mode="none",
            raw_sum_mod_10=args.raw_sum_mod_10,
            loaded_model_obj=loaded_model_obj,
        )
        carry_unbalanced = load_processed_task_data(
            flow_utils_module=flow_utils_module,
            h5_path=args.h5,
            pos=args.pos,
            model_path=args.model,
            seed=args.seed,
            train_target="C_potential",
            balance_mode="none",
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
    fit_sample_filter_mode = CURRENT_SAMPLE_FILTER_MODE
    if seq_len != carry_seq_len or feature_dim != carry_feature_dim:
        raise RuntimeError("external flow_utils 返回的 seq_len/feature_dim 不一致。")
    if not torch.equal(raw_y_binary_unbalanced, carry_y_binary_unbalanced):
        raise RuntimeError("raw-sum 与 carry 的 y_binary 不一致，无法对齐 classifier.py 的平衡逻辑。")
    if not torch.equal(raw_position_indices_unbalanced, carry_position_indices_unbalanced):
        raise RuntimeError("raw-sum 与 carry 的 position_indices 不一致。")

    effective_balance_mode = "none" if args.train_on_correct else CURRENT_BALANCE_MODE

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
        raise RuntimeError(f"不支持的 balance_mode: {effective_balance_mode}")

    kept_indices_t = torch.from_numpy(kept_indices.astype(np.int64))
    raw_X_balanced = raw_X_unbalanced[kept_indices_t]
    raw_y_balanced = raw_y_unbalanced[kept_indices_t]
    raw_y_binary_balanced = raw_y_binary_unbalanced[kept_indices_t]
    raw_position_indices = raw_position_indices_unbalanced[kept_indices_t]
    raw_sample_ids_balanced = raw_sample_ids_unbalanced[kept_indices_t]
    raw_meta_balanced = subset_meta_dict(raw_meta_unbalanced, kept_indices)

    carry_X_balanced = carry_X_unbalanced[kept_indices_t]
    carry_y_balanced = carry_y_unbalanced[kept_indices_t]
    carry_y_binary_balanced = carry_y_binary_unbalanced[kept_indices_t]
    carry_position_indices = carry_position_indices_unbalanced[kept_indices_t]
    carry_sample_ids_balanced = carry_sample_ids_unbalanced[kept_indices_t]
    carry_meta_balanced = subset_meta_dict(carry_meta_unbalanced, kept_indices)

    raw_data_unbalanced = build_position_data_from_processed(
        features=select_layer_features(raw_X_unbalanced, seq_len, feature_dim, args.layer),
        y_binary=raw_y_binary_unbalanced,
        raw_labels=raw_y_unbalanced,
        carry_labels=torch.tensor(
            [
                flow_utils_module.compute_c_potential(sample_meta[int(sample_id)]["question"], args.pos)
                for sample_id in raw_sample_ids_unbalanced.detach().cpu().numpy().tolist()
            ],
            dtype=torch.float32,
        ),
        sample_idx_all=raw_sample_ids_unbalanced,
        meta=raw_meta_unbalanced,
        sample_meta=sample_meta,
        pos=args.pos,
    )
    if not torch.equal(raw_y_binary_balanced, carry_y_binary_balanced):
        raise RuntimeError("平衡后 raw-sum 与 carry 的 y_binary 不一致。")
    if not torch.equal(raw_sample_ids_balanced, carry_sample_ids_balanced):
        raise RuntimeError("平衡后 raw-sum 与 carry 的 sample_id 顺序不一致。")
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

    total_rows = load_h5_position_row_count(args.h5, args.pos)
    mismatch_count = int((~raw_data_unbalanced.gt_char_match).sum())
    target_result_len = pick_target_result_len(raw_data_unbalanced.result_lens)
    dropped_by_result_len = int(total_rows - len(raw_data_unbalanced))
    pre_balance_rows = len(raw_data_unbalanced)
    pre_balance_correct = int(raw_data_unbalanced.labels.sum())
    pre_balance_wrong = int((~raw_data_unbalanced.labels).sum())

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
            raise RuntimeError("正确样本不足以切分 train/val。")
        if wrong_idx.size == 0:
            raise RuntimeError("错误样本数为 0，无法构造 test 集。")

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
            test_size=DEFAULT_TEST_SIZE,
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
        warnings.warn("raw-sum 与 carry 的 layer 特征不完全一致，后续将分别使用各自 external loader 返回的特征。")

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

    configure_vertical_flow_classifier(
        classifier_module=classifier_module,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        train_target="raw_sum_classify",
        raw_sum_mod_10=args.raw_sum_mod_10,
    )
    raw_model, raw_result = classifier_module.train_single_run(
        X_raw_fit_train,
        y_raw_fit_train,
        pos_raw_fit_train,
        X_raw_fit_val,
        y_raw_fit_val,
        pos_raw_fit_val,
        seq_len=1,
        feature_dim=X_raw_fit_train.shape[1],
        num_classes=raw_num_classes,
        is_regression=False,
        label_prefix="[raw] ",
    )
    raw_train_metrics = evaluate_with_external_classifier(
        classifier_module,
        raw_model,
        X_raw_train,
        y_raw_train,
        pos_train,
        is_regression=False,
        train_target="raw_sum_classify",
        batch_size=args.batch_size,
    )
    raw_test_metrics = evaluate_with_external_classifier(
        classifier_module,
        raw_model,
        X_raw_test,
        y_raw_test,
        pos_test,
        is_regression=False,
        train_target="raw_sum_classify",
        batch_size=args.batch_size,
    )
    raw_test_preds, raw_test_probs = predict_with_external_classifier(
        raw_model,
        X_raw_test,
        batch_size=args.batch_size,
        device=device,
    )
    raw_probe = ClassifierResult(
        model=raw_model,
        trained_epochs=int(raw_result["best_epoch"]),
        best_metric_name=str(raw_result["best_metric_name"]),
        best_test_metric=safe_float(raw_result["best_metric_display"]),
        train_metrics=raw_train_metrics,
        test_metrics=raw_test_metrics,
        split_strategy=split_strategy,
        num_classes=raw_num_classes,
        test_preds=raw_test_preds,
        test_probs=raw_test_probs,
    )

    configure_vertical_flow_classifier(
        classifier_module=classifier_module,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        train_target="C_potential",
        raw_sum_mod_10=False,
    )
    carry_model, carry_result = classifier_module.train_single_run(
        X_carry_fit_train,
        y_carry_fit_train,
        pos_carry_fit_train,
        X_carry_fit_val,
        y_carry_fit_val,
        pos_carry_fit_val,
        seq_len=1,
        feature_dim=X_carry_fit_train.shape[1],
        num_classes=1,
        is_regression=True,
        label_prefix="[carry] ",
    )
    carry_train_metrics = evaluate_with_external_classifier(
        classifier_module,
        carry_model,
        X_carry_train,
        y_carry_train,
        pos_train,
        is_regression=True,
        train_target="C_potential",
        batch_size=args.batch_size,
    )
    carry_test_metrics = evaluate_with_external_classifier(
        classifier_module,
        carry_model,
        X_carry_test,
        y_carry_test,
        pos_test,
        is_regression=True,
        train_target="C_potential",
        batch_size=args.batch_size,
    )
    carry_test_preds = predict_with_external_regressor(
        carry_model,
        X_carry_test,
        batch_size=args.batch_size,
        device=device,
    )
    carry_probe = RegressorResult(
        model=carry_model,
        trained_epochs=int(carry_result["best_epoch"]),
        best_metric_name=str(carry_result["best_metric_name"]),
        best_test_metric=safe_float(carry_result["best_metric_display"]),
        train_metrics=carry_train_metrics,
        test_metrics=carry_test_metrics,
        split_strategy=split_strategy,
        test_preds=carry_test_preds,
    )

    raw_test_mod10_acc = float(np.mean((raw_probe.test_preds % 10) == (test_data.raw_sum_full % 10)))
    error_rows, bucket_counts = decompose_test_errors(
        test_data=test_data,
        raw_pred_labels=raw_probe.test_preds,
        carry_preds_phi=carry_probe.test_preds,
        raw_sum_mod_10=args.raw_sum_mod_10,
    )

    incorrect_test_count = int((~test_data.labels).sum())
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
            f"按当前过滤后的数据自动检测到 raw-sum 类别数为 {raw_probe.num_classes}，"
            "与预期的 28 类不一致，请进一步核查 H5 或筛选条件。"
        )

    summary = {
        "task": "probe_error_decomposition",
        "target_h5": str(args.h5),
        "sample_reference_source": sample_reference_source,
        "split_source": "generated_in_memory",
        "target_pos": int(args.pos),
        "layer": int(args.layer),
        "input_pos": "consistent",
        "feature_type": "flows",
        "pooling_type": "none",
        "sample_filter_mode": CURRENT_SAMPLE_FILTER_MODE,
        "balance_mode": effective_balance_mode,
        "balance_target_classes": CURRENT_BALANCE_TARGET_CLASSES,
        "filter_by_result_len": True,
        "raw_sum_mod_10": bool(args.raw_sum_mod_10),
        "train_on_correct": bool(args.train_on_correct),
        "test_size": None if args.train_on_correct else DEFAULT_TEST_SIZE,
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
            "test": {
                "loss": safe_float(carry_probe.test_metrics["loss"]),
                "mae": safe_float(carry_probe.test_metrics["mae"]),
                "floor_acc": safe_float(carry_probe.test_metrics["floor_acc"]),
            },
        },
        "error_decomposition": {
            "test_count": int(test_count),
            "incorrect_test_count": incorrect_test_count,
            "correct_test_count": int(bucket_counts.get("correct", 0)),
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
        f"raw_test_acc={raw_probe.test_metrics['acc']:.4f} | "
        f"raw_test_mod10_acc={raw_test_mod10_acc:.4f} | "
        f"carry_test_floor_acc={carry_probe.test_metrics['floor_acc']:.4f} | "
        f"carry_wrong_raw_correct={bucket_counts.get('carry_wrong_raw_correct', 0)} ({bucket_pct_of_test.get('carry_wrong_raw_correct', 0.0):.4f}) | "
        f"raw_wrong_carry_correct={bucket_counts.get('raw_wrong_carry_correct', 0)} ({bucket_pct_of_test.get('raw_wrong_carry_correct', 0.0):.4f}) | "
        f"both_wrong={bucket_counts.get('both_wrong', 0)} ({bucket_pct_of_test.get('both_wrong', 0.0):.4f}) | "
        f"unexplained_probe_failure={bucket_counts.get('unexplained_probe_failure', 0)} ({bucket_pct_of_test.get('unexplained_probe_failure', 0.0):.4f}) | "
        f"internal_anomaly={bucket_counts.get('internal_anomaly', 0)} ({bucket_pct_of_test.get('internal_anomaly', 0.0):.4f})"
    )


if __name__ == "__main__":
    main()
