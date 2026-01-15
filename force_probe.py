import argparse
import copy
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===========================
# Carry potential helpers
# ===========================

def compute_raw_sum(question: str, sum_pos: int) -> int:
    """
    计算指定位置的本位和（各加数在该位置的数字之和，不取模）。

    Args:
        question: 算式字符串，如 "123 + 456 + 789"
        sum_pos: 和的位置索引（从最高位开始，0=最高位）

    Returns:
        int: 本位和（>=0），如果无法计算则返回 -1

    Note:
        和的 pos 与加数的 pos 可能不一致。如果最高位有进位，和的总位数会比加数更多。
        加数的 pos = 和的 pos - (和的位数 - 加数的位数)
    """
    try:
        operands = []
        for part in question.split('+'):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))

        if not operands:
            return -1

        result = sum(operands)
        result_str = str(result)
        result_len = len(result_str)
        max_operand_len = max(len(str(op)) for op in operands)
        extra_digits = result_len - max_operand_len
        operand_pos = sum_pos - extra_digits

        if operand_pos < 0:
            return 0

        digit_sum = 0
        for op in operands:
            op_str = str(op)
            op_len = len(op_str)
            right_idx = result_len - 1 - sum_pos
            op_digit_idx = op_len - 1 - right_idx

            if 0 <= op_digit_idx < op_len:
                digit_sum += int(op_str[op_digit_idx])

        return digit_sum
    except Exception:
        return -1


def compute_c_potential(question: str, current_pos: int) -> float:
    """
    计算 C_potential (Potential of Truth)
    公式: C_potential(pos) = sum( raw_sum(k) / 10^(k-pos) ) for k = pos+1 to end
    """
    try:
        operands = []
        for part in question.split('+'):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))
        if not operands:
            return 0.0

        result_len = len(str(sum(operands)))
        c_potential = 0.0
        for k in range(current_pos + 1, result_len + 5):
            raw_sum_k = compute_raw_sum(question, k)
            if raw_sum_k == -1:
                break

            exponent = k - current_pos
            term = raw_sum_k / (10 ** exponent)
            c_potential += term

            if term < 1e-9:
                break

        return c_potential
    except Exception:
        return 0.0


# ===========================
# Data loading
# ===========================

def load_dataset(path: Path) -> List[List[int]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [list(item) for item in data]


def _to_str_list(arr) -> List[str]:
    if arr is None:
        return []
    out: List[str] = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def load_positions(h5_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    positions: Dict[int, Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as hf:
        positions_group = hf["all_token_results"]
        for pos_name, pos_group in positions_group.items():
            if not pos_name.startswith("pos_"):
                continue
            try:
                pos_idx = int(pos_name.split("_", 1)[1])
            except Exception:
                continue

            flows = np.asarray(pos_group.get("flows"))
            labels = np.asarray(pos_group.get("labels"), dtype=np.bool_)
            preds = _to_str_list(pos_group.get("preds"))
            gt_chars = _to_str_list(pos_group.get("gt_chars"))
            true_carry = pos_group.get("true_in_carry")
            pred_carry = pos_group.get("pred_in_carry")
            positions[pos_idx] = {
                "flows": flows,
                "labels": labels,
                "preds": preds,
                "gt_chars": gt_chars,
                "true_carry": np.asarray(true_carry) if true_carry is not None else None,
                "pred_carry": np.asarray(pred_carry) if pred_carry is not None else None,
            }
    return positions


# ===========================
# Probe models
# ===========================

class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10, hidden_dim: int = 512, dropout: float = 0.2):
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
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
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
        return self.net(x)


def build_probe(probe_type: str, input_dim: int, num_classes: int) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(input_dim, num_classes)
    if probe_type == "mlp":
        return ProbeMLP(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported probe_type: {probe_type}")


def build_regressor(probe_type: str, input_dim: int) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(input_dim, 1)
    if probe_type == "mlp":
        return ProbeMLPRegressor(input_dim=input_dim)
    raise ValueError(f"Unsupported probe_type for regression: {probe_type}")


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    seed: int,
    test_size: float,
    device: torch.device,
    probe_type: str,
) -> Tuple[nn.Module, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    strat = y if len(np.unique(y)) > 1 else None
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = build_probe(probe_type, input_dim=X.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds: List[int] = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_acc = float(np.mean(val_preds == y_val.numpy())) if len(y_val) else float("nan")
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        full_logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        full_preds = torch.argmax(full_logits, dim=1).cpu().numpy()
    full_acc = float(np.mean(full_preds == y)) if len(y) else float("nan")
    return model, best_val, full_acc


def train_carry_regressor(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    seed: int,
    test_size: float,
    device: torch.device,
    probe_type: str,
) -> Tuple[nn.Module, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = build_regressor(probe_type, input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb).squeeze(1)
                val_losses.append(float(criterion(preds, yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds_full = model(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(1)
        full_loss = float(nn.functional.mse_loss(preds_full, torch.tensor(y, dtype=torch.float32, device=device)).item())
    return model, best_val_loss, full_loss


LOG_DIR = Path("VerticalFlow/log/log_interia")
SAVE_DIR = Path("VerticalFlow/saved_models")


# ===========================
# Core logic
# ===========================

def build_flat_dataset(
    dataset: List[List[int]],
    positions: Dict[int, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每个位置的 flows 逐样本对齐 dataset 计算标签。"""
    flows_list: List[np.ndarray] = []
    raw_labels: List[int] = []
    carry_labels: List[float] = []
    gt_digits: List[int] = []
    pred_digits: List[int] = []
    sample_ids: List[int] = []

    for pos_idx, data in positions.items():
        flows = np.asarray(data["flows"])
        preds = data["preds"]
        gt_chars = data["gt_chars"]
        max_rows = min(len(dataset), len(flows), len(preds), len(gt_chars))
        for sample_idx in range(max_rows):
            operands = dataset[sample_idx]
            question = " + ".join(str(x) for x in operands)
            gt_value = sum(operands)
            gt_str = str(gt_value)
            if pos_idx >= len(gt_str):
                continue
            if gt_chars[sample_idx] != gt_str[pos_idx]:
                continue

            raw_sum_val = compute_raw_sum(question, pos_idx)
            if raw_sum_val < 0:
                continue
            raw_mod = raw_sum_val % 10
            c_potential = compute_c_potential(question, pos_idx)

            pred_token = preds[sample_idx]
            pred_digit = int(pred_token) if str(pred_token).isdigit() else -1

            flows_list.append(flows[sample_idx])
            raw_labels.append(raw_mod)
            carry_labels.append(float(c_potential))
            gt_digits.append(int(gt_str[pos_idx]))
            pred_digits.append(pred_digit)
            sample_ids.append(sample_idx)

    if not flows_list:
        raise RuntimeError("No samples built from H5/dataset alignment.")

    flows_all = np.stack(flows_list, axis=0)
    return (
        flows_all,
        np.asarray(raw_labels, dtype=np.int64),
        np.asarray(carry_labels, dtype=np.float32),
        np.asarray(gt_digits, dtype=np.int64),
        np.asarray(pred_digits, dtype=np.int64),
        np.asarray(sample_ids, dtype=np.int64),
    )


def evaluate_correction(
    raw_model: nn.Module,
    carry_model: nn.Module,
    flows: np.ndarray,
    raw_layer: int,
    inertia_layer: int,
    gt_digits: np.ndarray,
    pred_digits: np.ndarray,
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    with torch.no_grad():
        X_raw = torch.tensor(flows[:, raw_layer, :], dtype=torch.float32, device=device)
        X_inertia = torch.tensor(flows[:, inertia_layer, :], dtype=torch.float32, device=device)
        raw_hat = torch.argmax(raw_model(X_raw), dim=1).cpu().numpy()
        carry_pred = carry_model(X_inertia).squeeze(1)
        carry_hat = torch.floor(torch.clamp(carry_pred, min=0.0)).to(torch.int64).cpu().numpy()

    corrected = (raw_hat + carry_hat) % 10

    orig_token_acc = float(np.mean(pred_digits == gt_digits))
    corrected_token_acc = float(np.mean(corrected == gt_digits))

    fixed_mask = np.logical_and(pred_digits != gt_digits, corrected == gt_digits)
    harmed_mask = np.logical_and(pred_digits == gt_digits, corrected != gt_digits)

    # sample-level
    unique_samples = np.unique(sample_ids)
    orig_sample_correct = 0
    corrected_sample_correct = 0
    for sid in unique_samples:
        mask = sample_ids == sid
        if np.all(pred_digits[mask] == gt_digits[mask]):
            orig_sample_correct += 1
        if np.all(corrected[mask] == gt_digits[mask]):
            corrected_sample_correct += 1
    sample_acc_orig = orig_sample_correct / len(unique_samples) if len(unique_samples) else 0.0
    sample_acc_corrected = corrected_sample_correct / len(unique_samples) if len(unique_samples) else 0.0

    return {
        "total": float(len(gt_digits)),
        "orig_token_acc": orig_token_acc,
        "corrected_token_acc": corrected_token_acc,
        "orig_sample_acc": sample_acc_orig,
        "corrected_sample_acc": sample_acc_corrected,
        "fixed_count": float(fixed_mask.sum()),
        "harmed_count": float(harmed_mask.sum()),
    }


def get_digit_token_ids(tokenizer: AutoTokenizer) -> Dict[int, int]:
    digit_ids = {}
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {d} is not a single token: {ids}")
        digit_ids[d] = ids[0]
    return digit_ids


def run_intervention_eval(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    raw_model: nn.Module,
    carry_model: nn.Module,
    raw_layer: int,
    inertia_layer: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, float]:
    digit_ids = get_digit_token_ids(tokenizer)

    raw_dtype = next(raw_model.parameters()).dtype
    carry_dtype = next(carry_model.parameters()).dtype

    total_samples = len(dataset)
    print(f"[online] Start intervention eval on {total_samples} samples", flush=True)

    token_total = 0
    token_correct = 0
    sample_total = 0
    sample_correct = 0

    raw_model.eval()
    carry_model.eval()

    with torch.no_grad():
        for idx, operands in enumerate(dataset):
            gt_val = sum(operands)
            gt_str = str(gt_val)

            expr = " + ".join(str(x) for x in operands)
            messages = [{"role": "user", "content": f"Calculate {expr}. Only output a number."}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            text = text + expr + " = "

            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values

            generated_digits: List[int] = []
            hidden_states = outputs.hidden_states
            raw_h = hidden_states[raw_layer + 1][:, -1, :].to(raw_dtype)
            inertia_h = hidden_states[inertia_layer + 1][:, -1, :].to(carry_dtype)
            raw_hat = torch.argmax(raw_model(raw_h), dim=1)
            carry_pred = carry_model(inertia_h).squeeze(1)
            carry_hat = torch.floor(torch.clamp(carry_pred, min=0.0)).to(torch.int64)
            corrected_digit = int((raw_hat + carry_hat) % 10)
            next_token_id = digit_ids[corrected_digit]
            generated_digits.append(corrected_digit)

            next_input = torch.tensor([[next_token_id]], device=device)

            for _ in range(max_new_tokens - 1):
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = outputs.past_key_values
                hidden_states = outputs.hidden_states
                raw_h = hidden_states[raw_layer + 1][:, -1, :].to(raw_dtype)
                inertia_h = hidden_states[inertia_layer + 1][:, -1, :].to(carry_dtype)
                raw_hat = torch.argmax(raw_model(raw_h), dim=1)
                carry_pred = carry_model(inertia_h).squeeze(1)
                carry_hat = torch.floor(torch.clamp(carry_pred, min=0.0)).to(torch.int64)
                corrected_digit = int((raw_hat + carry_hat) % 10)
                next_token_id = digit_ids[corrected_digit]
                generated_digits.append(corrected_digit)
                next_input = torch.tensor([[next_token_id]], device=device)
                if len(generated_digits) >= len(gt_str) + 1:
                    break

            # Accuracy accounting
            sample_total += 1
            g_len = len(generated_digits)
            t_len = len(gt_str)
            for i in range(max(g_len, t_len)):
                if i < t_len and i < g_len:
                    token_total += 1
                    if generated_digits[i] == int(gt_str[i]):
                        token_correct += 1
                else:
                    token_total += 1
            if g_len == t_len and all(generated_digits[i] == int(gt_str[i]) for i in range(t_len)):
                sample_correct += 1

            if (idx + 1) % 100 == 0:
                token_acc_so_far = token_correct / token_total if token_total else 0.0
                sample_acc_so_far = sample_correct / sample_total if sample_total else 0.0
                print(
                    f"[online] Progress: {idx + 1}/{total_samples} samples | token_acc={token_acc_so_far:.4f} | sample_acc={sample_acc_so_far:.4f}",
                    flush=True,
                )

    token_acc = token_correct / token_total if token_total else 0.0
    sample_acc = sample_correct / sample_total if sample_total else 0.0
    return {"token_acc": token_acc, "sample_acc": sample_acc, "token_total": token_total, "sample_total": sample_total}


def balance_carry_data(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """平衡所有 carry 标签：各类过采样到最大类数量。"""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) <= 1:
        return X, y

    counts = {int(c): int((y == c).sum()) for c in classes}
    target = max(counts.values())

    chunks = []
    labels = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) < target:
            extra = rng.choice(idx, size=target - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        chunks.append(X[idx])
        labels.append(y[idx])

    X_bal = np.concatenate(chunks, axis=0)
    y_bal = np.concatenate(labels, axis=0)
    shuffle_idx = rng.permutation(len(X_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]


def main():
    parser = argparse.ArgumentParser(description="Force-correction probe based on triangle consistency.")
    parser.add_argument("--h5", type=Path, default=Path("VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"), help="Path to HDF5 results")
    parser.add_argument("--dataset", type=Path, default=Path("VerticalFlow/num3len10-10000.pkl"), help="Dataset used for generation")
    parser.add_argument("--model", type=str, default="/data/Models/Qwen3-4b", help="Model path for online intervention generation")
    parser.add_argument(
        "--layers",
        type=str,
        nargs=2,
        metavar=("S_RAW_LAYER", "INERTIA_LAYER"),
        default=[-1, 24],
        help="Layer pair for S_raw and inertia probes; use 'none' to auto-search best layer",
    )
    parser.add_argument("--dataset-size", type=int, default=10000, help="Number of samples to load from dataset (<=0 for all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["offline", "online", "both"], default="both", help="Evaluation mode")
    parser.add_argument("--probe-type", type=str, choices=["linear", "mlp"], default="mlp", help="Probe architecture")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    dataset_full = load_dataset(args.dataset)
    dataset_online = dataset_full if args.dataset_size <= 0 else dataset_full[:args.dataset_size]
    positions = load_positions(args.h5)
    def _parse_layer(val: str):
        if isinstance(val, str) and val.lower() == "none":
            return None
        return int(val)

    layer_pair = (_parse_layer(args.layers[0]), _parse_layer(args.layers[1]))

    flows_all, raw_labels, carry_labels, gt_digits, pred_digits, sample_ids = build_flat_dataset(dataset_full, positions)

    raw_classes = int(raw_labels.max()) + 1
    num_layers = flows_all.shape[1]
    sample_dim = flows_all.shape[2]
    print(
        f"Loaded {len(raw_labels)} tokens | feature_dim={sample_dim} | layers={num_layers} | "
        f"raw_layer={layer_pair[0]} | inertia_layer={layer_pair[1]} | "
        f"raw_classes={raw_classes}"
    )

    # ---------- Raw probe (S_raw) ----------
    if layer_pair[0] is not None:
        raw_layer_best = layer_pair[0]
        raw_vecs = flows_all[:, raw_layer_best, :]
        raw_model, raw_val_acc, raw_full_acc = train_probe(
            raw_vecs,
            raw_labels,
            num_classes=raw_classes,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            test_size=args.test_size,
            device=device,
            probe_type=args.probe_type,
        )
    else:
        raw_best = (-1.0, -1.0, None, None)  # val_acc, full_acc, layer, model
        for l in range(num_layers):
            raw_vecs = flows_all[:, l, :]
            model_l, val_l, full_l = train_probe(
                raw_vecs,
                raw_labels,
                num_classes=raw_classes,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                test_size=args.test_size,
                device=device,
                probe_type=args.probe_type,
            )
            if val_l > raw_best[0]:
                raw_best = (val_l, full_l, l, model_l)
        raw_val_acc, raw_full_acc, raw_layer_best, raw_model = raw_best
        raw_vecs = flows_all[:, raw_layer_best, :]
    print(f"Raw-sum probe (layer={raw_layer_best}): val_acc={raw_val_acc:.4f}, full_acc={raw_full_acc:.4f}")

    # ---------- Carry probe (inertia) regression ----------
    if layer_pair[1] is not None:
        inertia_layer_best = layer_pair[1]
        inertia_vecs = flows_all[:, inertia_layer_best, :]
        carry_model, carry_val_loss, carry_full_loss = train_carry_regressor(
            inertia_vecs,
            carry_labels,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            test_size=args.test_size,
            device=device,
            probe_type=args.probe_type,
        )
        inertia_vecs = flows_all[:, inertia_layer_best, :]
    else:
        carry_best = (float("inf"), float("inf"), None, None, None)  # val_loss, full_loss, layer, model, vecs
        for l in range(num_layers):
            inertia_vecs_l = flows_all[:, l, :]
            model_l, val_l, full_l = train_carry_regressor(
                inertia_vecs_l,
                carry_labels,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                test_size=args.test_size,
                device=device,
                probe_type=args.probe_type,
            )
            if val_l < carry_best[0]:
                carry_best = (val_l, full_l, l, model_l, inertia_vecs_l)
        carry_val_loss, carry_full_loss, inertia_layer_best, carry_model, inertia_vecs = carry_best
    print(f"Carry probe (layer={inertia_layer_best}): val_mse={carry_val_loss:.6f}, full_mse={carry_full_loss:.6f}")

    inter_metrics = None
    if args.mode in ("online", "both"):
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype="auto",
            output_hidden_states=True,
        )
        inter_metrics = run_intervention_eval(
            dataset_online,
            tokenizer,
            model,
            raw_model,
            carry_model,
            raw_layer_best,
            inertia_layer_best,
            max_new_tokens=25,
            device=device,
        )
        print("\n=== Intervention online generation ===")
        print(f"Token Acc (corrected): {inter_metrics['token_acc']:.4f} ({inter_metrics['token_total']} tokens)")
        print(f"Sample Acc (corrected): {inter_metrics['sample_acc']:.4f} ({inter_metrics['sample_total']} samples)")

    # Save models
    raw_path = SAVE_DIR / f"force_raw_layer{raw_layer_best}.pt"
    inertia_path = SAVE_DIR / f"force_inertia_layer{inertia_layer_best}.pt"
    torch.save({"state_dict": raw_model.state_dict(), "layer": raw_layer_best}, raw_path)
    torch.save({"state_dict": carry_model.state_dict(), "layer": inertia_layer_best}, inertia_path)
    print(f"Saved raw probe to {raw_path}")
    print(f"Saved inertia probe to {inertia_path}")

    # Materialize final vectors for evaluation
    metrics = None
    if args.mode in ("offline", "both"):
        metrics = evaluate_correction(
            raw_model,
            carry_model,
            flows_all,
            raw_layer_best,
            inertia_layer_best,
            gt_digits,
            pred_digits,
            sample_ids,
            device,
        )
        print("\n=== Correction summary (offline) ===")
        print(f"Tokens: {metrics['total']:.0f}")
        print(f"Original token accuracy: {metrics['orig_token_acc']:.4f}")
        print(f"Corrected token accuracy: {metrics['corrected_token_acc']:.4f}")
        print(f"Original sample accuracy: {metrics['orig_sample_acc']:.4f}")
        print(f"Corrected sample accuracy: {metrics['corrected_sample_acc']:.4f}")
        print(f"Fixed wrong tokens: {metrics['fixed_count']:.0f}")
        print(f"Harmed correct tokens: {metrics['harmed_count']:.0f}")

    # Persist metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"force_probe_{timestamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"h5: {args.h5}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"dataset_size: {len(dataset_full)}\n")
        f.write(f"online_dataset_size: {len(dataset_online)}\n")
        f.write(f"layers: (raw={raw_layer_best}, inertia={inertia_layer_best})\n")
        f.write(f"raw_val_acc: {raw_val_acc:.6f}, raw_full_acc: {raw_full_acc:.6f}\n")
        f.write(f"carry_val_mse: {carry_val_loss:.6f}, carry_full_mse: {carry_full_loss:.6f}\n")
        if metrics is not None:
            f.write(f"tokens: {metrics['total']:.0f}\n")
            f.write(f"orig_token_acc: {metrics['orig_token_acc']:.6f}\n")
            f.write(f"corrected_token_acc: {metrics['corrected_token_acc']:.6f}\n")
            f.write(f"orig_sample_acc: {metrics['orig_sample_acc']:.6f}\n")
            f.write(f"corrected_sample_acc: {metrics['corrected_sample_acc']:.6f}\n")
            f.write(f"fixed_count: {metrics['fixed_count']:.0f}\n")
            f.write(f"harmed_count: {metrics['harmed_count']:.0f}\n")
        if inter_metrics is not None:
            f.write("\n[intervention_online]\n")
            f.write(f"token_acc: {inter_metrics['token_acc']:.6f} ({inter_metrics['token_total']})\n")
            f.write(f"sample_acc: {inter_metrics['sample_acc']:.6f} ({inter_metrics['sample_total']})\n")
    print(f"Logged metrics to {log_path}")


if __name__ == "__main__":
    main()
