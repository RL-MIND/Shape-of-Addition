import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_data import (
    build_flat_dataset,
    compute_token_sample_acc,
    load_dataset,
    load_positions,
    split_sample_ids,
)
from verify import (
    build_dirs_cross_digit,
    collect_records,
    compute_means,
    load_position_arrays,
)


def mask_first_error_positions(
    pred_digits: np.ndarray,
    gt_digits: np.ndarray,
    sample_ids: np.ndarray,
    pos_ids: np.ndarray,
) -> np.ndarray:
    """仅保留每个样本的第一个错误位置（其余错误位置置为 False）。"""
    keep = np.ones_like(pred_digits, dtype=bool)
    incorrect = pred_digits != gt_digits
    if not np.any(incorrect):
        return keep
    for sid in np.unique(sample_ids[incorrect]):
        sid_mask = (sample_ids == sid) & incorrect
        if np.sum(sid_mask) <= 1:
            continue
        min_pos = pos_ids[sid_mask].min()
        keep[sid_mask & (pos_ids != min_pos)] = False
    return keep


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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
    probe_type: str,
) -> Tuple[nn.Module, float]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = build_probe(probe_type, input_dim=X_train.shape[1], num_classes=num_classes).to(device)
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
        val_acc = float(np.mean(val_preds == y_val_t.numpy())) if len(y_val_t) else float("nan")
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

    return model, best_val


def train_carry_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
    probe_type: str,
) -> Tuple[nn.Module, float, float]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = build_regressor(probe_type, input_dim=X_train.shape[1]).to(device)
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

    full_loss = float("nan")  # 不再计算 full_loss
    return model, best_val_loss, full_loss


def evaluate_carry_accuracy_floor(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    if len(X) == 0:
        return float("nan")
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device)).squeeze(1).cpu().numpy()
            target = yb.cpu().numpy()
            preds = np.floor(np.maximum(preds, 0.0))
            target = np.floor(np.maximum(target, 0.0))
            correct += int(np.sum(preds.astype(int) == target.astype(int)))
            total += len(target)
    return correct / total if total else float("nan")


def evaluate_correction(
    raw_model: nn.Module,
    carry_model: nn.Module,
    flows: np.ndarray,
    raw_layer: int,
    inertia_layer: int,
    gt_digits: np.ndarray,
    pred_digits: np.ndarray,
    sample_ids: np.ndarray,
    pos_ids: np.ndarray,
    inertia_delta: float,
    device: torch.device,
) -> Dict[str, object]:
    raw_model.eval()
    carry_model.eval()
    with torch.no_grad():
        X_raw = torch.tensor(flows[:, raw_layer, :], dtype=torch.float32, device=device)
        X_inertia = torch.tensor(flows[:, inertia_layer, :], dtype=torch.float32, device=device)
        raw_hat = torch.argmax(raw_model(X_raw), dim=1).cpu().numpy()
        carry_pred = carry_model(X_inertia).squeeze(1).cpu().numpy()

    corrected = np.zeros_like(raw_hat, dtype=np.int64)
    for i in range(len(raw_hat)):
        raw_d = int(raw_hat[i])
        phi = float(max(carry_pred[i], 0.0))
        d_pred = int(pred_digits[i]) if 0 <= pred_digits[i] <= 9 else -1

        low = math.floor(phi - inertia_delta)
        high = math.floor(phi + inertia_delta)
        intervene = True
        if d_pred != -1:
            for c in range(low, high + 1):
                if (raw_d + c) % 10 == d_pred:
                    intervene = False
                    break

        if intervene:
            carry_hat = math.floor(phi)
            corrected[i] = (raw_d + carry_hat) % 10
        else:
            corrected[i] = d_pred if d_pred != -1 else (raw_d + math.floor(phi)) % 10

    orig_token_acc = float(np.mean(pred_digits == gt_digits))
    corrected_token_acc = float(np.mean(corrected == gt_digits))

    fixed_mask = np.logical_and(pred_digits != gt_digits, corrected == gt_digits)
    harmed_mask = np.logical_and(pred_digits == gt_digits, corrected != gt_digits)

    # 新增统计指标
    # Modified Rate: 探针输出与原始模型输出不同的比例
    modified_count = np.sum(corrected != pred_digits)
    modified_rate = float(modified_count) / len(corrected) if len(corrected) > 0 else 0.0
    
    # TP Correction: 原始错误中被成功修正的比例
    orig_errors = pred_digits != gt_digits
    tp_total = np.sum(orig_errors)
    if tp_total > 0:
        tp_correction = float(fixed_mask.sum()) / float(tp_total)
    else:
        tp_correction = float("nan")
    
    # FP Preservation: 原始正确中保持正确的比例
    orig_correct = pred_digits == gt_digits
    fp_total = np.sum(orig_correct)
    if fp_total > 0:
        fp_preserved = np.sum((orig_correct) & (corrected == gt_digits))
        fp_preservation = float(fp_preserved) / float(fp_total)
    else:
        fp_preservation = float("nan")

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
        "corrected_digits": corrected,
        "modified_rate": modified_rate,
        "tp_correction": tp_correction,
        "fp_preservation": fp_preservation,
    }


def evaluate_correction_vector_steer(
    raw_model: nn.Module,
    carry_model: nn.Module,
    flows: np.ndarray,
    raw_layer: int,
    inertia_layer: int,
    gt_digits: np.ndarray,
    pred_digits: np.ndarray,
    sample_ids: np.ndarray,
    inertia_delta: float,
    digit_id_list: List[int],
    digit_val_list: List[int],
    lm_head: nn.Module,
    dir01: Dict[int, torch.Tensor],
    dir12: Dict[int, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    raw_model.eval()
    carry_model.eval()
    lm_head.eval()
    with torch.no_grad():
        X_raw = torch.tensor(flows[:, raw_layer, :], dtype=torch.float32, device=device)
        X_inertia = torch.tensor(flows[:, inertia_layer, :], dtype=torch.float32, device=device)
        raw_dtype = next(raw_model.parameters()).dtype
        carry_dtype = next(carry_model.parameters()).dtype
        if X_raw.dtype != raw_dtype:
            X_raw = X_raw.to(dtype=raw_dtype)
        if X_inertia.dtype != carry_dtype:
            X_inertia = X_inertia.to(dtype=carry_dtype)
        raw_hat = torch.argmax(raw_model(X_raw), dim=1).cpu().numpy()
        carry_pred = carry_model(X_inertia).squeeze(1).cpu().numpy()

    corrected = np.zeros_like(raw_hat, dtype=np.int64)
    head_dtype = next(lm_head.parameters()).dtype
    h_last_all = torch.tensor(flows[:, -1, :], dtype=head_dtype, device=device)

    for i in range(len(raw_hat)):
        raw_d = int(raw_hat[i])
        phi = float(max(carry_pred[i], 0.0))
        d_pred = int(pred_digits[i]) if 0 <= pred_digits[i] <= 9 else -1

        low = math.floor(phi - inertia_delta)
        high = math.floor(phi + inertia_delta)
        intervene = True
        if d_pred != -1:
            for c in range(low, high + 1):
                if (raw_d + c) % 10 == d_pred:
                    intervene = False
                    break

        if not intervene:
            corrected[i] = d_pred
            continue

        pred_carry = int((d_pred - raw_d) % 10) if d_pred != -1 else -1
        actual_carry = int(math.floor(phi))
        chosen_digit = None
        if pred_carry in (0, 1, 2) and actual_carry in (0, 1, 2):
            diff = actual_carry - pred_carry
            steer_vecs: List[torch.Tensor] = []
            if diff == 1:
                if pred_carry == 0 and dir01.get(d_pred) is not None:
                    steer_vecs.append(dir01[d_pred])
                elif pred_carry == 1 and dir12.get(d_pred) is not None:
                    steer_vecs.append(dir12[d_pred])
            elif diff == -1:
                if pred_carry == 1 and dir01.get((d_pred - 1) % 10) is not None:
                    steer_vecs.append(dir01[(d_pred - 1) % 10])
                elif pred_carry == 2 and dir12.get((d_pred - 1) % 10) is not None:
                    steer_vecs.append(dir12[(d_pred - 1) % 10])
            elif diff == 2:
                if pred_carry == 0:
                    v1 = dir01.get(d_pred)
                    v2 = dir12.get((d_pred + 1) % 10)
                    if v1 is not None and v2 is not None:
                        steer_vecs.extend([v1, v2])
            elif diff == -2:
                if pred_carry == 2:
                    v1 = dir12.get((d_pred - 1) % 10)
                    v2 = dir01.get((d_pred - 2) % 10)
                    if v1 is not None and v2 is not None:
                        steer_vecs.extend([v1, v2])

            if steer_vecs:
                steer_sum = torch.zeros_like(h_last_all[i:i + 1])
                for vec in steer_vecs:
                    steer_sum = steer_sum + vec
                steered = h_last_all[i:i + 1] + steer_sum
                steered_logits = lm_head(steered)
                digit_logits = steered_logits[:, digit_id_list]
                idx = torch.argmax(digit_logits, dim=1).item()
                chosen_digit = int(digit_val_list[idx])

        if chosen_digit is None:
            carry_hat = math.floor(phi)
            chosen_digit = int((raw_d + carry_hat) % 10)

        corrected[i] = chosen_digit

    orig_token_acc = float(np.mean(pred_digits == gt_digits))
    corrected_token_acc = float(np.mean(corrected == gt_digits))

    # 新增统计指标
    # Modified Rate: 探针输出与原始模型输出不同的比例
    modified_count = np.sum(corrected != pred_digits)
    modified_rate = float(modified_count) / len(corrected) if len(corrected) > 0 else 0.0
    
    # TP Correction: 原始错误中被成功修正的比例
    orig_errors = pred_digits != gt_digits
    tp_total = np.sum(orig_errors)
    if tp_total > 0:
        fixed_mask = (orig_errors) & (corrected == gt_digits)
        tp_correction = float(fixed_mask.sum()) / float(tp_total)
    else:
        tp_correction = float("nan")
    
    # FP Preservation: 原始正确中保持正确的比例
    orig_correct = pred_digits == gt_digits
    fp_total = np.sum(orig_correct)
    if fp_total > 0:
        fp_preserved = np.sum((orig_correct) & (corrected == gt_digits))
        fp_preservation = float(fp_preserved) / float(fp_total)
    else:
        fp_preservation = float("nan")

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
        "corrected_digits": corrected,
        "modified_rate": modified_rate,
        "tp_correction": tp_correction,
        "fp_preservation": fp_preservation,
    }


def get_digit_token_ids(tokenizer: AutoTokenizer) -> Tuple[List[int], List[int]]:
    digit_ids = {}
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {d} is not a single token: {ids}")
        digit_ids[d] = ids[0]
    digit_id_list = [digit_ids[d] for d in sorted(digit_ids.keys())]
    digit_val_list = sorted(digit_ids.keys())
    return digit_id_list, digit_val_list


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对输入应用 RMSNorm。"""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def apply_rms_norm_to_flows(
    flows: np.ndarray,
    norm_weight: torch.Tensor,
    layer_idx: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """对 flows 指定层应用 RMSNorm。"""
    flows_tensor = torch.tensor(flows[:, layer_idx, :], dtype=torch.float32)
    weight = norm_weight.float().cpu()
    normed = rms_norm(flows_tensor, weight, eps)
    flows_out = flows.copy()
    flows_out[:, layer_idx, :] = normed.numpy()
    return flows_out


def get_norm_weight_from_model(model_path: str, device: torch.device) -> Tuple[torch.Tensor, float]:
    """从模型中提取 RMSNorm 的 weight 参数。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype="auto",
    )
    norm_module = getattr(model.model, "norm", None)
    if norm_module is None:
        raise RuntimeError("Model does not have model.model.norm")
    weight = norm_module.weight.detach().clone()
    eps = getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6))
    del model
    torch.cuda.empty_cache()
    return weight, eps


def online_force_eval(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    raw_model: nn.Module,
    carry_model: nn.Module,
    raw_layer: int,
    inertia_layer: int,
    max_new_tokens: int,
    inertia_delta: float,
    device: torch.device,
    vector_steer: bool = False,
    dir01: Dict[int, torch.Tensor] | None = None,
    dir12: Dict[int, torch.Tensor] | None = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> Tuple[float, float, float, float, float]:
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    # 用于捕获 pre-norm hidden states 的容器和 hook
    # Qwen3 模型的 hidden_states[-1] 已经过 RMSNorm，需要用 hook 捕获 norm 前的状态
    captured_prenorm: List[torch.Tensor] = []

    def prenorm_hook(module, args, output):
        # args[0] 是 norm 层的输入（pre-norm hidden states）
        captured_prenorm.append(args[0].detach())

    # 注册 hook 到 model.model.norm
    norm_module = getattr(model.model, "norm", None)
    hook_handle = None
    if norm_module is not None:
        hook_handle = norm_module.register_forward_hook(prenorm_hook)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    def should_intervene(raw_hat_int: int, carry_pred_float: float, d_pred_int: int) -> bool:
        if d_pred_int < 0 or d_pred_int > 9:
            return True
        carry_clamped = max(carry_pred_float, 0.0)
        low = math.floor(carry_clamped - inertia_delta)
        high = math.floor(carry_clamped + inertia_delta)
        for c in range(low, high + 1):
            if (raw_hat_int + c) % 10 == d_pred_int:
                return False
        return True

    token_total = 0
    token_correct = 0
    sample_total = 0
    sample_correct = 0

    # 新增统计指标
    modified_count = 0  # 探针修正了多少token
    tp_total = 0  # 模型原始错误的token数
    tp_corrected = 0  # 模型原始错误中被探针修正的数量
    fp_total = 0  # 模型原始正确的token数
    fp_preserved = 0  # 模型原始正确中探针保持不变的数量

    lm_head = None
    if vector_steer:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("vector-steer requires lm_head or output embeddings")
        lm_head = lm_head.to(device)
        lm_head.eval()

    raw_model.eval()
    carry_model.eval()
    model.eval()
    with torch.no_grad():
        for operands in dataset:
            gt_val = sum(operands)
            gt_str = str(gt_val)

            expr = " + ".join(str(x) for x in operands)
            messages = [{"role": "user", "content": f"Calculate {expr}. Only output a number."}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            text = text + expr + " = "

            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            # 清空 pre-norm 捕获（每个样本开始前）
            captured_prenorm.clear()
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            max_layer = len(outputs.hidden_states) - 1
            raw_layer_idx = raw_layer
            inertia_layer_idx = inertia_layer
            if raw_layer_idx < 0:
                raw_layer_idx = max_layer
            if inertia_layer_idx < 0:
                inertia_layer_idx = max_layer
            if raw_layer_idx > max_layer:
                raw_layer_idx = max_layer
            if inertia_layer_idx > max_layer:
                inertia_layer_idx = max_layer

            generated_digits: List[int] = []
            original_digits: List[int] = []  # 记录原始模型预测
            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]
                d_pred = select_digit(logits)
                original_digits.append(d_pred)
                hidden_states = outputs.hidden_states

                # 获取 raw_h 和 inertia_h
                # 如果请求的层是最后一层，使用 pre-norm states（与 generate.py 保存时一致）
                if raw_layer_idx == max_layer and len(captured_prenorm) > 0:
                    raw_h = captured_prenorm[-1][:, -1, :]
                else:
                    raw_h = hidden_states[raw_layer_idx][:, -1, :]

                if inertia_layer_idx == max_layer and len(captured_prenorm) > 0:
                    inertia_h = captured_prenorm[-1][:, -1, :]
                else:
                    inertia_h = hidden_states[inertia_layer_idx][:, -1, :]

                # 应用 RMSNorm（与模型内部归一化步骤相同）
                if norm_weight is not None:
                    raw_h = rms_norm(raw_h, norm_weight.to(raw_h.device).to(raw_h.dtype), norm_eps)
                    inertia_h = rms_norm(inertia_h, norm_weight.to(inertia_h.device).to(inertia_h.dtype), norm_eps)

                raw_dtype = next(raw_model.parameters()).dtype
                carry_dtype = next(carry_model.parameters()).dtype
                if raw_h.dtype != raw_dtype:
                    raw_h = raw_h.to(dtype=raw_dtype)
                if inertia_h.dtype != carry_dtype:
                    inertia_h = inertia_h.to(dtype=carry_dtype)
                raw_hat = int(torch.argmax(raw_model(raw_h), dim=1).item())
                carry_pred_val = float(carry_model(inertia_h).squeeze(1).item())
                intervene = should_intervene(raw_hat, carry_pred_val, d_pred)
                if intervene and vector_steer and dir01 is not None and dir12 is not None and lm_head is not None:
                    pred_carry = int((d_pred - raw_hat) % 10)
                    actual_carry = int(math.floor(max(carry_pred_val, 0.0)))
                    if pred_carry in (0, 1, 2) and actual_carry in (0, 1, 2):
                        diff = actual_carry - pred_carry
                        steer_vecs: List[torch.Tensor] = []
                        if diff == 1:
                            if pred_carry == 0 and dir01.get(d_pred) is not None:
                                steer_vecs.append(dir01[d_pred])
                            elif pred_carry == 1 and dir12.get(d_pred) is not None:
                                steer_vecs.append(dir12[d_pred])
                        elif diff == -1:
                            if pred_carry == 1 and dir01.get((d_pred - 1) % 10) is not None:
                                steer_vecs.append(dir01[(d_pred - 1) % 10])
                            elif pred_carry == 2 and dir12.get((d_pred - 1) % 10) is not None:
                                steer_vecs.append(dir12[(d_pred - 1) % 10])
                        elif diff == 2:
                            if pred_carry == 0:
                                v1 = dir01.get(d_pred)
                                v2 = dir12.get((d_pred + 1) % 10)
                                if v1 is not None and v2 is not None:
                                    steer_vecs.extend([v1, v2])
                        elif diff == -2:
                            if pred_carry == 2:
                                v1 = dir12.get((d_pred - 1) % 10)
                                v2 = dir01.get((d_pred - 2) % 10)
                                if v1 is not None and v2 is not None:
                                    steer_vecs.extend([v1, v2])

                        if steer_vecs:
                            h_last = hidden_states[-1][:, -1, :]
                            head_dtype = next(lm_head.parameters()).dtype
                            if h_last.dtype != head_dtype:
                                h_last = h_last.to(dtype=head_dtype)
                            steer_sum = torch.zeros_like(h_last)
                            for vec in steer_vecs:
                                steer_sum = steer_sum + vec
                            steered = h_last + steer_sum
                            steered_logits = lm_head(steered)
                            digit_logits = steered_logits[:, digit_id_list]
                            idx = torch.argmax(digit_logits, dim=1).item()
                            chosen_digit = int(digit_val_list[idx])
                        else:
                            carry_hat = math.floor(max(carry_pred_val, 0.0))
                            chosen_digit = int((raw_hat + carry_hat) % 10)
                    else:
                        carry_hat = math.floor(max(carry_pred_val, 0.0))
                        chosen_digit = int((raw_hat + carry_hat) % 10)
                elif intervene:
                    carry_hat = math.floor(max(carry_pred_val, 0.0))
                    chosen_digit = int((raw_hat + carry_hat) % 10)
                else:
                    chosen_digit = int(d_pred)
                generated_digits.append(chosen_digit)

                # # 测试：对于错误位置，使用ground truth作为输入token（teacher forcing）
                # current_pos = len(generated_digits) - 1
                # if current_pos < len(gt_str):
                #     gt_digit = int(gt_str[current_pos])
                #     # 如果当前位置预测错误，使用正确答案作为下一个输入
                #     if chosen_digit != gt_digit:
                #         next_token_id = digit_id_list[gt_digit]
                #     else:
                #         next_token_id = digit_id_list[chosen_digit]
                # else:
                #     next_token_id = digit_id_list[chosen_digit]
                next_token_id = digit_id_list[chosen_digit]
                next_input = torch.tensor([[next_token_id]], device=device)
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = outputs.past_key_values

                if len(generated_digits) >= len(gt_str):
                    break

            sample_total += 1
            g_len = len(generated_digits)
            t_len = len(gt_str)
            o_len = len(original_digits)
            for i in range(max(g_len, t_len)):
                token_total += 1
                gt_digit = int(gt_str[i]) if i < t_len else -1
                orig_digit = original_digits[i] if i < o_len else -1
                corr_digit = generated_digits[i] if i < g_len else -1

                if i < t_len and i < g_len and corr_digit == gt_digit:
                    token_correct += 1

                # Modified Rate: 探针输出与原始模型输出不同
                if i < o_len and i < g_len and corr_digit != orig_digit:
                    modified_count += 1

                # TP Correction: 模型原始错误中被探针修正
                if i < t_len and i < o_len and orig_digit != gt_digit:
                    tp_total += 1
                    if i < g_len and corr_digit == gt_digit:
                        tp_corrected += 1

                # FP Preservation: 模型原始正确中探针保持不变
                if i < t_len and i < o_len and orig_digit == gt_digit:
                    fp_total += 1
                    if i < g_len and corr_digit == gt_digit:
                        fp_preserved += 1

            if g_len == t_len and all(generated_digits[i] == int(gt_str[i]) for i in range(t_len)):
                sample_correct += 1

    # 移除 hook
    if hook_handle is not None:
        hook_handle.remove()

    token_acc = token_correct / token_total if token_total else 0.0
    sample_acc = sample_correct / sample_total if sample_total else 0.0
    modified_rate = modified_count / token_total if token_total else 0.0
    tp_correction = tp_corrected / tp_total if tp_total else float("nan")
    fp_preservation = fp_preserved / fp_total if fp_total else float("nan")
    return token_acc, sample_acc, modified_rate, tp_correction, fp_preservation


def main():
    parser = argparse.ArgumentParser(description="Force-correction probe based on triangle consistency.")
    parser.add_argument("--h5", type=Path, default=Path("VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"), help="Path to HDF5 results")
    parser.add_argument("--dataset", type=Path, default=Path("VerticalFlow/num3len10-10000.pkl"), help="Dataset used for generation")
    parser.add_argument(
        "--layers",
        type=str,
        nargs=2,
        metavar=("S_RAW_LAYER", "INERTIA_LAYER"),
        default=[-1, -1],
        help="Layer pair for S_raw and inertia probes; use 'none' to auto-search best layer",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-type", type=str, choices=["linear", "mlp"], default="mlp", help="Probe architecture")
    parser.add_argument("--positions", type=int, nargs="*", default=None, help="Positions to include; default all")
    parser.add_argument(
        "--sample-filter",
        type=str,
        choices=["all", "correct", "incorrect"],
        default="all",
        help="Filter tokens by model correctness: all/correct/incorrect",
    )
    parser.add_argument("--inertia-delta", type=float, default=0, help="Delta window around phi for intervention gating")
    parser.add_argument("--vector-steer", action="store_true", help="Use vector steering instead of raw_sum+incarry correction")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write metrics JSON")
    parser.add_argument("--test-mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--model", type=str, default="/data/Models/Qwen3-4b")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_full = load_dataset(args.dataset)
    positions = load_positions(args.h5)

    # 加载 RMSNorm 参数
    print(f"Loading RMSNorm parameters from {args.model}...")
    norm_weight, norm_eps = get_norm_weight_from_model(args.model, device)
    print(f"RMSNorm eps: {norm_eps}")

    def _parse_layer(val: str):
        if isinstance(val, str) and val.lower() == "none":
            return None
        return int(val)

    layer_pair = (_parse_layer(args.layers[0]), _parse_layer(args.layers[1]))

    flows_all, raw_labels, carry_labels, gt_digits, pred_digits, sample_ids, pos_ids = build_flat_dataset(
        dataset_full,
        positions,
        positions_filter=args.positions,
    )

    # 对 flows 最后一层应用 RMSNorm
    num_layers = flows_all.shape[1]
    print(f"Applying RMSNorm to flows (last layer {num_layers - 1})...")
    flows_all = apply_rms_norm_to_flows(flows_all, norm_weight, num_layers - 1, norm_eps)

    train_ids, val_ids, test_ids = split_sample_ids(
        sample_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_mask = np.isin(sample_ids, list(train_ids))
    val_mask = np.isin(sample_ids, list(val_ids)) if val_ids else np.zeros_like(sample_ids, dtype=bool)
    test_mask = np.isin(sample_ids, list(test_ids)) if test_ids else np.zeros_like(sample_ids, dtype=bool)

    if args.sample_filter != "all":
        if args.sample_filter == "correct":
            keep_mask = pred_digits == gt_digits
        else:
            keep_mask = pred_digits != gt_digits
        train_mask = np.logical_and(train_mask, keep_mask)
        val_mask = np.logical_and(val_mask, keep_mask)

    first_error_keep = mask_first_error_positions(pred_digits, gt_digits, sample_ids, pos_ids)
    train_mask = np.logical_and(train_mask, first_error_keep)
    val_mask = np.logical_and(val_mask, first_error_keep)

    flows_train = flows_all[train_mask]
    raw_labels_train = raw_labels[train_mask]
    carry_labels_train = carry_labels[train_mask]
    gt_digits_train = gt_digits[train_mask]
    pred_digits_train = pred_digits[train_mask]
    sample_ids_train = sample_ids[train_mask]
    pos_ids_train = pos_ids[train_mask]

    flows_val = flows_all[val_mask] if val_mask.any() else flows_all
    raw_labels_val = raw_labels[val_mask] if val_mask.any() else raw_labels
    carry_labels_val = carry_labels[val_mask] if val_mask.any() else carry_labels

    flows_test = flows_all[test_mask] if test_mask.any() else flows_all
    raw_labels_test = raw_labels[test_mask] if test_mask.any() else raw_labels
    carry_labels_test = carry_labels[test_mask] if test_mask.any() else carry_labels
    gt_digits_test = gt_digits[test_mask] if test_mask.any() else gt_digits
    pred_digits_test = pred_digits[test_mask] if test_mask.any() else pred_digits
    sample_ids_test = sample_ids[test_mask] if test_mask.any() else sample_ids
    pos_ids_test = pos_ids[test_mask] if test_mask.any() else pos_ids

    orig_token_acc, orig_sample_acc = compute_token_sample_acc(
        pred_digits_test, gt_digits_test, sample_ids_test
    )
    print(f"Original token accuracy (test): {orig_token_acc:.4f}")
    print(f"Original sample accuracy (test): {orig_sample_acc:.4f}")

    raw_classes = 10
    num_layers = flows_all.shape[1]
    sample_dim = flows_all.shape[2]
    print(
        f"Loaded {len(raw_labels)} tokens | feature_dim={sample_dim} | layers={num_layers} | "
        f"raw_layer={layer_pair[0]} | inertia_layer={layer_pair[1]} | "
        f"raw_classes={raw_classes} | train_samples={len(train_ids)} | val_samples={len(val_ids)} | test_samples={len(test_ids)}"
    )

    def _positions_tag(pos_list: List[int] | None) -> str:
        if not pos_list:
            return "all"
        return "-".join(str(p) for p in pos_list)

    raw_tag = "auto" if layer_pair[0] is None else str(layer_pair[0])
    inertia_tag = "auto" if layer_pair[1] is None else str(layer_pair[1])
    ckpt_name = (
        f"dualstream_probe_{args.h5.stem}_pos{_positions_tag(args.positions)}_"
        f"raw{raw_tag}_in{inertia_tag}_ptype{args.probe_type}_"
        f"sf{args.sample_filter}_seed{args.seed}_"
        f"tr{args.train_ratio}_vr{args.val_ratio}_te{args.test_ratio}.pt"
    )
    save_dir = Path("VerticalFlow/saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / ckpt_name
    use_ckpt = ckpt_path.exists()

    raw_val_acc = float("nan")
    carry_val_loss = float("nan")
    carry_val_acc = float("nan")
    raw_layer_best = None
    inertia_layer_best = None
    raw_model = None
    carry_model = None

    if use_ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_layer_best = int(ckpt["raw_layer"])
        inertia_layer_best = int(ckpt["inertia_layer"])
        raw_model = build_probe(args.probe_type, input_dim=sample_dim, num_classes=raw_classes).to(device)
        raw_model.load_state_dict(ckpt["raw_state"])
        carry_model = build_regressor(args.probe_type, input_dim=sample_dim).to(device)
        carry_model.load_state_dict(ckpt["carry_state"])
        raw_val_acc = float(ckpt.get("raw_val_acc", float("nan")))
        carry_val_loss = float(ckpt.get("carry_val_mse", float("nan")))
        carry_val_acc = float(ckpt.get("carry_val_acc_floor", float("nan")))
        print(f"Loaded probes from {ckpt_path}")

    if not use_ckpt:
        # ---------- Raw probe (S_raw) ----------
        if layer_pair[0] is not None:
            raw_layer_best = layer_pair[0]
            raw_vecs = flows_train[:, raw_layer_best, :]
            raw_model, raw_val_acc = train_probe(
                raw_vecs,
                raw_labels_train,
                flows_val[:, raw_layer_best, :],
                raw_labels_val,
                num_classes=raw_classes,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                probe_type=args.probe_type,
            )
        else:
            raw_best = (-1.0, -1.0, None, None)  # val_acc, full_acc placeholder, layer, model
            for l in range(num_layers):
                raw_vecs = flows_train[:, l, :]
                model_l, val_l = train_probe(
                    raw_vecs,
                    raw_labels_train,
                    flows_val[:, l, :],
                    raw_labels_val,
                    num_classes=raw_classes,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    patience=args.patience,
                    device=device,
                    probe_type=args.probe_type,
                )
                if val_l > raw_best[0]:
                    raw_best = (val_l, float("nan"), l, model_l)
            raw_val_acc, _, raw_layer_best, raw_model = raw_best
            raw_vecs = flows_train[:, raw_layer_best, :]
        if raw_layer_best is None or raw_model is None:
            raise RuntimeError("Failed to train raw probe; no layer selected")
        raw_val_acc = float(raw_val_acc)
        print(f"Raw-sum probe (layer={raw_layer_best}): val_acc={raw_val_acc:.4f}")

        # ---------- Carry probe (inertia) regression ----------
        if layer_pair[1] is not None:
            inertia_layer_best = layer_pair[1]
            inertia_vecs = flows_train[:, inertia_layer_best, :]
            carry_model, carry_val_loss, _ = train_carry_regressor(
                inertia_vecs,
                carry_labels_train,
                flows_val[:, inertia_layer_best, :],
                carry_labels_val,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                probe_type=args.probe_type,
            )
            inertia_vecs = flows_train[:, inertia_layer_best, :]
        else:
            carry_best = (float("inf"), float("nan"), None, None, None)  # val_loss, full_loss placeholder, layer, model, vecs
            for l in range(num_layers):
                inertia_vecs_l = flows_train[:, l, :]
                model_l, val_l, _ = train_carry_regressor(
                    inertia_vecs_l,
                    carry_labels_train,
                    flows_val[:, l, :],
                    carry_labels_val,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    patience=args.patience,
                    device=device,
                    probe_type=args.probe_type,
                )
                if val_l < carry_best[0]:
                    carry_best = (val_l, float("nan"), l, model_l, inertia_vecs_l)
            carry_val_loss, _, inertia_layer_best, carry_model, inertia_vecs = carry_best
        if inertia_layer_best is None or carry_model is None:
            raise RuntimeError("Failed to train carry probe; no layer selected")
        carry_val_acc = evaluate_carry_accuracy_floor(
            carry_model,
            flows_val[:, inertia_layer_best, :],
            carry_labels_val,
            batch_size=args.batch_size,
            device=device,
        )
        print(
            f"Carry probe (layer={inertia_layer_best}): val_mse={carry_val_loss:.6f}, "
            f"val_acc_floor={carry_val_acc:.4f}"
        )

        torch.save(
            {
                "raw_layer": int(raw_layer_best),
                "inertia_layer": int(inertia_layer_best),
                "raw_val_acc": float(raw_val_acc),
                "carry_val_mse": float(carry_val_loss),
                "carry_val_acc_floor": float(carry_val_acc),
                "probe_type": args.probe_type,
                "sample_dim": int(sample_dim),
                "raw_state": raw_model.state_dict(),
                "carry_state": carry_model.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved probes to {ckpt_path}")
    else:
        print(f"Raw-sum probe (layer={raw_layer_best}): val_acc={raw_val_acc:.4f}")
        print(
            f"Carry probe (layer={inertia_layer_best}): val_mse={carry_val_loss:.6f}, "
            f"val_acc_floor={carry_val_acc:.4f}"
        )

    if args.test_mode == "offline":
        if args.vector_steer:
            if args.model is None:
                raise ValueError("--model is required for offline vector-steer mode")
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            lm = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=device,
                torch_dtype="auto",
                output_hidden_states=True,
            )
            labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos = load_position_arrays(args.h5)
            pos_filter = set(args.positions) if args.positions else None
            records = collect_records(labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos, positions=pos_filter)
            means, _ = compute_means(records)
            dir01_np, dir12_np = build_dirs_cross_digit(means)
            head = getattr(lm, "lm_head", None)
            if head is None:
                head = lm.get_output_embeddings()
            if head is None:
                raise RuntimeError("vector-steer requires lm_head or output embeddings")
            head_dtype = next(head.parameters()).dtype
            dir01 = {}
            dir12 = {}
            for d in range(10):
                if dir01_np.get(d) is not None:
                    dir01[d] = torch.tensor(dir01_np[d], dtype=head_dtype, device=device).unsqueeze(0)
                if dir12_np.get(d) is not None:
                    dir12[d] = torch.tensor(dir12_np[d], dtype=head_dtype, device=device).unsqueeze(0)
            digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)
            metrics = evaluate_correction_vector_steer(
                raw_model,
                carry_model,
                flows_test,
                raw_layer_best,
                inertia_layer_best,
                gt_digits_test,
                pred_digits_test,
                sample_ids_test,
                args.inertia_delta,
                digit_id_list,
                digit_val_list,
                head,
                dir01,
                dir12,
                device,
            )
        else:
            metrics = evaluate_correction(
                raw_model,
                carry_model,
                flows_test,
                raw_layer_best,
                inertia_layer_best,
                gt_digits_test,
                pred_digits_test,
                sample_ids_test,
                pos_ids_test,
                args.inertia_delta,
                device,
            )
        corrected_token_acc = float(metrics["corrected_token_acc"])
        corrected_sample_acc = float(metrics["corrected_sample_acc"])
        modified_rate = float(metrics["modified_rate"])
        tp_correction = float(metrics["tp_correction"])
        fp_preservation = float(metrics["fp_preservation"])
    else:
        if args.model is None:
            raise ValueError("--model is required for online test mode")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype="auto",
            output_hidden_states=True,
            do_sample=False,
        )
        dir01 = None
        dir12 = None
        if args.vector_steer:
            labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos = load_position_arrays(args.h5)
            pos_filter = set(args.positions) if args.positions else None
            records = collect_records(labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos, positions=pos_filter)
            means, _ = compute_means(records)
            dir01_np, dir12_np = build_dirs_cross_digit(means)
            head = getattr(lm, "lm_head", None)
            if head is None:
                head = lm.get_output_embeddings()
            if head is None:
                raise RuntimeError("vector-steer requires lm_head or output embeddings")
            head_dtype = next(head.parameters()).dtype
            dir01 = {}
            dir12 = {}
            for d in range(10):
                if dir01_np.get(d) is not None:
                    dir01[d] = torch.tensor(dir01_np[d], dtype=head_dtype, device=device).unsqueeze(0)
                if dir12_np.get(d) is not None:
                    dir12[d] = torch.tensor(dir12_np[d], dtype=head_dtype, device=device).unsqueeze(0)
        dataset_test = [dataset_full[i] for i in sorted(test_ids) if 0 <= i < len(dataset_full)]
        # 从已加载的模型中获取 norm_weight
        norm_module = getattr(lm.model, "norm", None)
        if norm_module is not None:
            online_norm_weight = norm_module.weight.detach().clone()
            online_norm_eps = getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6))
        else:
            online_norm_weight = norm_weight
            online_norm_eps = norm_eps
        corrected_token_acc, corrected_sample_acc, modified_rate, tp_correction, fp_preservation = online_force_eval(
            dataset_test,
            tokenizer,
            lm,
            raw_model,
            carry_model,
            raw_layer_best,
            inertia_layer_best,
            args.max_new_tokens,
            args.inertia_delta,
            device,
            vector_steer=args.vector_steer,
            dir01=dir01,
            dir12=dir12,
            norm_weight=online_norm_weight,
            norm_eps=online_norm_eps,
        )

    print("\n=== Force probe (test) ===")
    print(f"Corrected token accuracy: {corrected_token_acc:.4f}")
    print(f"Corrected sample accuracy: {corrected_sample_acc:.4f}")
    print(f"Modified rate: {modified_rate:.4f}")
    print(f"TP Correction: {tp_correction:.4f}")
    print(f"FP Preservation: {fp_preservation:.4f}")

    payload = {
        "raw_layer": int(raw_layer_best),
        "inertia_layer": int(inertia_layer_best),
        "raw_val_acc": float(raw_val_acc),
        "carry_val_mse": float(carry_val_loss),
        "carry_val_acc_floor": float(carry_val_acc),
        "orig_token_acc": float(orig_token_acc),
        "orig_sample_acc": float(orig_sample_acc),
        "corrected_token_acc": float(corrected_token_acc),
        "corrected_sample_acc": float(corrected_sample_acc),
        "modified_rate": float(modified_rate),
        "tp_correction": float(tp_correction),
        "fp_preservation": float(fp_preservation),
        "test_mode": args.test_mode,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
