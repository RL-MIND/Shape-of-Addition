import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

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


class MLPProbe(nn.Module):
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


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = MLPProbe(X_train.shape[1], num_classes=10).to(device)
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
        preds: List[int] = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_acc = float(np.mean(preds == y_val_t.numpy())) if len(y_val_t) else float("nan")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_mlp(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.int64)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)


def parse_layer_candidates(num_layers: int, layers: List[int] | None, layer_start: int | None, layer_end: int | None) -> List[int]:
    if layers:
        return [int(l) for l in layers]
    if layer_start is None and layer_end is None:
        return [num_layers - 1]
    start = layer_start if layer_start is not None else num_layers - 1
    end = layer_end if layer_end is not None else start
    start = max(0, start)
    end = min(num_layers - 1, end)
    if start > end:
        start, end = end, start
    return list(range(start, end + 1))


def compute_correction_metrics(
    corrected: np.ndarray,
    pred_orig: np.ndarray,
    gt: np.ndarray,
) -> Tuple[float, float, float]:
    """
    计算校正统计指标。
    
    Args:
        corrected: 探针校正后的预测
        pred_orig: 原始模型预测
        gt: ground truth
    
    Returns:
        modified_rate: 被修改的token比例
        tp_correction: 原始错误中被成功修正的比例
        fp_preservation: 原始正确中保持不变的比例
    """
    n = len(corrected)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Modified Rate: 探针输出与原始模型输出不同的比例
    modified_count = np.sum(corrected != pred_orig)
    modified_rate = float(modified_count) / n
    
    # TP Correction: 原始错误中被成功修正的比例
    orig_errors = pred_orig != gt
    tp_total = np.sum(orig_errors)
    if tp_total > 0:
        tp_corrected = np.sum((orig_errors) & (corrected == gt))
        tp_correction = float(tp_corrected) / float(tp_total)
    else:
        tp_correction = float("nan")
    
    # FP Preservation: 原始正确中保持正确的比例
    orig_correct = pred_orig == gt
    fp_total = np.sum(orig_correct)
    if fp_total > 0:
        fp_preserved = np.sum((orig_correct) & (corrected == gt))
        fp_preservation = float(fp_preserved) / float(fp_total)
    else:
        fp_preservation = float("nan")
    
    return modified_rate, tp_correction, fp_preservation


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
    """
    对输入应用 RMSNorm（与 Qwen3 模型内部归一化步骤相同）。
    
    Args:
        x: 输入张量，shape (..., hidden_dim)
        weight: RMSNorm 的 weight 参数，shape (hidden_dim,)
        eps: 防止除零的小常数
    
    Returns:
        归一化后的张量
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def apply_rms_norm_to_flows(
    flows: np.ndarray,
    norm_weight: torch.Tensor,
    layer_idx: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    对 flows 指定层应用 RMSNorm。
    
    Args:
        flows: shape (N, num_layers, hidden_dim)
        norm_weight: RMSNorm 的 weight 参数
        layer_idx: 要归一化的层索引
        eps: 防止除零的小常数
    
    Returns:
        归一化后的 flows
    """
    flows_tensor = torch.tensor(flows[:, layer_idx, :], dtype=torch.float32)
    weight = norm_weight.float().cpu()
    normed = rms_norm(flows_tensor, weight, eps)
    flows_out = flows.copy()
    flows_out[:, layer_idx, :] = normed.numpy()
    return flows_out


def get_norm_weight_from_model(model_path: str, device: torch.device) -> Tuple[torch.Tensor, float]:
    """
    从模型中提取 RMSNorm 的 weight 参数。
    
    Returns:
        (weight, eps): norm 层的 weight 和 eps 参数
    """
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


def online_eval_prompt(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    probe: nn.Module,
    layer: int,
    max_new_tokens: int,
    device: torch.device,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> Tuple[float, float, float, float, float]:
    """
    Prompt correction 模式的 online evaluation。
    
    当探针检测到当前位置可能错误时：
    1. 保留模型的原始输出
    2. 追加修正提示语 "That step looks incorrect. Let's re-do just this step:"
    3. 追加原算式的前半部分（到等号），让模型重新计算
    """
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    # 用于捕获 pre-norm hidden states 的容器和 hook
    captured_prenorm: List[torch.Tensor] = []

    def prenorm_hook(module, args, output):
        captured_prenorm.append(args[0].detach())

    norm_module = getattr(model.model, "norm", None)
    hook_handle = None
    if norm_module is not None:
        hook_handle = norm_module.register_forward_hook(prenorm_hook)

    token_total = 0
    token_correct = 0
    sample_total = 0
    sample_correct = 0

    modified_count = 0
    tp_total = 0
    tp_corrected = 0
    fp_total = 0
    fp_preserved = 0

    correction_prompt = " That step looks incorrect. Let's re-do just this step: "

    probe.eval()
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
            prefix = text + expr + " = "

            model_inputs = tokenizer([prefix], return_tensors="pt").to(device)
            captured_prenorm.clear()
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            max_layer = len(outputs.hidden_states) - 1
            layer_idx = layer
            if layer_idx < 0:
                layer_idx = max_layer
            if layer_idx > max_layer:
                layer_idx = max_layer

            generated_digits: List[int] = []
            original_digits: List[int] = []
            current_output_str = ""  # 当前已输出的数字字符串

            step = 0
            while step < max_new_tokens:
                logits = outputs.logits[:, -1, :]
                d_pred = select_digit(logits)
                original_digits.append(d_pred)
                hidden_states = outputs.hidden_states

                # 获取 hidden state（使用 pre-norm 如果是最后一层）
                if layer_idx == max_layer and len(captured_prenorm) > 0:
                    h = captured_prenorm[-1][:, -1, :]
                else:
                    h = hidden_states[layer_idx][:, -1, :]

                # 应用 RMSNorm（与模型内部归一化步骤相同）
                if norm_weight is not None:
                    h = rms_norm(h, norm_weight.to(h.device), norm_eps)

                probe_dtype = next(probe.parameters()).dtype
                if h.dtype != probe_dtype:
                    h = h.to(dtype=probe_dtype)
                logits_probe = probe(h)
                probe_digit = int(torch.argmax(logits_probe, dim=1).item())

                # 先写入模型预测的数字（无论是否正确）
                next_token_id = digit_id_list[d_pred]
                next_input = torch.tensor([[next_token_id]], device=device)
                captured_prenorm.clear()
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = outputs.past_key_values
                generated_digits.append(d_pred)
                current_output_str += str(d_pred)

                # 检测是否需要干预（在数字已经写入 past 之后）
                if probe_digit != d_pred:
                    # 探针认为模型输出错误，启动 prompt correction
                    modified_count += 1

                    # 追加修正提示，只包含已正确的部分（不包含错误的数字和探针的建议）
                    # 格式: "That step looks incorrect. Let's re-do just this step: {expr} = {已正确的部分}"
                    # 例如：578 -> "That step looks incorrect. Let's re-do just this step: 123 + 456 = 57"
                    correction_text = correction_prompt + expr + " = " + current_output_str[:-1]
                    correction_ids = tokenizer.encode(correction_text, add_special_tokens=False)
                    correction_input = torch.tensor([correction_ids], device=device)
                    captured_prenorm.clear()
                    outputs = model(
                        input_ids=correction_input,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past = outputs.past_key_values

                    # 让模型重新生成这一位的数字
                    logits_new = outputs.logits[:, -1, :]
                    corrected_digit = select_digit(logits_new)
                    
                    # 更新 generated_digits：替换上一个错误的数字为模型重新生成的数字
                    generated_digits[-1] = corrected_digit
                    current_output_str = current_output_str[:-1] + str(corrected_digit)
                    
                    # 将新数字写入 past
                    next_token_id = digit_id_list[corrected_digit]
                    next_input = torch.tensor([[next_token_id]], device=device)
                    captured_prenorm.clear()
                    outputs = model(
                        input_ids=next_input,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past = outputs.past_key_values

                step += 1
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

                if i < t_len and i < o_len and orig_digit != gt_digit:
                    tp_total += 1
                    if i < g_len and corr_digit == gt_digit:
                        tp_corrected += 1

                if i < t_len and i < o_len and orig_digit == gt_digit:
                    fp_total += 1
                    if i < g_len and corr_digit == gt_digit:
                        fp_preserved += 1

            if g_len == t_len and all(generated_digits[i] == int(gt_str[i]) for i in range(t_len)):
                sample_correct += 1

    if hook_handle is not None:
        hook_handle.remove()

    token_acc = token_correct / token_total if token_total else 0.0
    sample_acc = sample_correct / sample_total if sample_total else 0.0
    modified_rate = modified_count / token_total if token_total else 0.0
    tp_correction = tp_corrected / tp_total if tp_total else 0.0
    fp_preservation = fp_preserved / fp_total if fp_total else 0.0
    return token_acc, sample_acc, modified_rate, tp_correction, fp_preservation


def online_eval(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    probe: nn.Module,
    layer: int,
    max_new_tokens: int,
    device: torch.device,
    mode: str = "direct",  # "direct" 或 "prompt"
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> Tuple[float, float, float, float, float]:
    """
    Online evaluation with probe correction.
    
    Args:
        mode: 
            - "direct": 直接用探针预测替换模型输出
            - "prompt": 当探针检测到错误时，追加修正提示让模型重新计算
        norm_weight: RMSNorm 的 weight 参数（可选）
        norm_eps: RMSNorm 的 eps 参数
    """
    if mode == "prompt":
        return online_eval_prompt(
            dataset, tokenizer, model, probe, layer, max_new_tokens, device,
            norm_weight=norm_weight, norm_eps=norm_eps
        )
    
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

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

    probe.eval()
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
            layer_idx = layer
            if layer_idx < 0:
                layer_idx = max_layer
            if layer_idx > max_layer:
                layer_idx = max_layer

            generated_digits: List[int] = []
            original_digits: List[int] = []  # 记录原始模型预测
            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]
                d_pred = select_digit(logits)
                original_digits.append(d_pred)
                hidden_states = outputs.hidden_states

                # 获取 hidden state
                # 如果请求的层是最后一层，使用 pre-norm states（与 generate.py 保存时一致）
                if layer_idx == max_layer and len(captured_prenorm) > 0:
                    h = captured_prenorm[-1][:, -1, :]
                else:
                    h = hidden_states[layer_idx][:, -1, :]

                # 应用 RMSNorm（与模型内部归一化步骤相同）
                if norm_weight is not None:
                    h = rms_norm(h, norm_weight.to(h.device), norm_eps)

                probe_dtype = next(probe.parameters()).dtype
                if h.dtype != probe_dtype:
                    h = h.to(dtype=probe_dtype)
                logits_probe = probe(h)
                corrected_digit = int(torch.argmax(logits_probe, dim=1).item())

                generated_digits.append(corrected_digit)

                next_token_id = digit_id_list[corrected_digit]
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
    tp_correction = tp_corrected / tp_total if tp_total else 0.0
    fp_preservation = fp_preserved / fp_total if fp_total else 0.0
    return token_acc, sample_acc, modified_rate, tp_correction, fp_preservation


def main():
    parser = argparse.ArgumentParser(description="MLP probe for digit prediction.")
    parser.add_argument("--h5", type=Path, default=Path("VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"))
    parser.add_argument("--dataset", type=Path, default=Path("VerticalFlow/num3len10-10000.pkl"), help="Dataset used for generation")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--test-mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--mode", type=str, choices=["direct", "prompt"], default="direct",
                        help="Correction mode: direct (replace with probe output) or prompt (append correction prompt)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 RMSNorm 参数
    if args.model is None:
        raise ValueError("--model is required to load RMSNorm parameters")
    print(f"Loading RMSNorm parameters from {args.model}...")
    norm_weight, norm_eps = get_norm_weight_from_model(args.model, device)
    print(f"RMSNorm eps: {norm_eps}")

    dataset_full = load_dataset(args.dataset)
    positions = load_positions(args.h5)
    flows_all, _, _, gt_digits, pred_digits, sample_ids, pos_ids = build_flat_dataset(
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

    first_error_keep = mask_first_error_positions(pred_digits, gt_digits, sample_ids, pos_ids)
    train_mask = np.logical_and(train_mask, first_error_keep)
    val_mask = np.logical_and(val_mask, first_error_keep)

    flows_train = flows_all[train_mask]
    gt_train = gt_digits[train_mask]
    flows_val = flows_all[val_mask] if val_mask.any() else flows_all
    gt_val = gt_digits[val_mask] if val_mask.any() else gt_digits
    flows_test = flows_all[test_mask] if test_mask.any() else flows_all
    gt_test = gt_digits[test_mask] if test_mask.any() else gt_digits
    pred_test = pred_digits[test_mask] if test_mask.any() else pred_digits
    sample_ids_test = sample_ids[test_mask] if test_mask.any() else sample_ids

    candidate_layers = parse_layer_candidates(num_layers, args.layers, args.layer_start, args.layer_end)

    best_layer = None
    best_val_acc = -1.0
    best_model = None
    for layer in candidate_layers:
        model, val_acc = train_mlp_probe(
            flows_train[:, layer, :],
            gt_train,
            flows_val[:, layer, :],
            gt_val,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_layer = layer
            best_model = model

    if best_model is None or best_layer is None:
        raise RuntimeError("Failed to train MLP probe; no layer selected")

    if args.test_mode == "offline":
        corrected = predict_mlp(best_model, flows_test[:, best_layer, :], device)
        corrected_token_acc, corrected_sample_acc = compute_token_sample_acc(
            corrected, gt_test, sample_ids_test
        )
        modified_rate, tp_correction, fp_preservation = compute_correction_metrics(
            corrected, pred_test, gt_test
        )
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
        dataset_test = [dataset_full[i] for i in sorted(test_ids) if 0 <= i < len(dataset_full)]
        # 从已加载的模型中获取 norm_weight
        norm_module = getattr(lm.model, "norm", None)
        if norm_module is not None:
            online_norm_weight = norm_module.weight.detach().clone()
            online_norm_eps = getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6))
        else:
            online_norm_weight = norm_weight
            online_norm_eps = norm_eps
        corrected_token_acc, corrected_sample_acc, modified_rate, tp_correction, fp_preservation = online_eval(
            dataset_test,
            tokenizer,
            lm,
            best_model,
            best_layer,
            args.max_new_tokens,
            device=device,
            mode=args.mode,
            norm_weight=online_norm_weight,
            norm_eps=online_norm_eps,
        )

    payload = {
        "method": "mlp_probe",
        "test_mode": args.test_mode,
        "mode": args.mode,
        "layer": int(best_layer),
        "val_acc": float(best_val_acc),
        "corrected_token_acc": float(corrected_token_acc),
        "corrected_sample_acc": float(corrected_sample_acc),
        "modified_rate": float(modified_rate),
        "tp_correction": float(tp_correction),
        "fp_preservation": float(fp_preservation),
    }

    print("\n=== MLP Probe Results ===")
    print(f"Best layer: {best_layer}")
    print(f"Mode: {args.mode}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Corrected token accuracy: {corrected_token_acc:.4f}")
    print(f"Corrected sample accuracy: {corrected_sample_acc:.4f}")
    print(f"Modified rate: {modified_rate:.4f}")
    print(f"TP Correction: {tp_correction:.4f}")
    print(f"FP Preservation: {fp_preservation:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
