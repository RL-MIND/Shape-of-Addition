import argparse
import json
from pathlib import Path
from typing import List, Tuple

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


def train_linear_probe(
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

    model = nn.Linear(X_train.shape[1], 10).to(device)
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


def predict_linear(model: nn.Linear, X: np.ndarray, device: torch.device) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.int64)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)


def steer_predict(
    model: nn.Linear,
    X: np.ndarray,
    lambd: float,
) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.int64)
    W = model.weight.detach().cpu().numpy()
    b = model.bias.detach().cpu().numpy() if model.bias is not None else np.zeros(W.shape[0], dtype=W.dtype)
    logits = X @ W.T + b
    pred_class = np.argmax(logits, axis=1)
    v = W[pred_class]
    X_steered = X + lambd * v
    logits_steered = X_steered @ W.T + b
    return np.argmax(logits_steered, axis=1).astype(np.int64)


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


def online_eval(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    probe: nn.Linear,
    layer: int,
    max_new_tokens: int,
    mode: str,
    lambd: float,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    model_dtype = model.dtype if hasattr(model, "dtype") else torch.float32
    W = probe.weight.detach().to(device=device, dtype=model_dtype)
    b = (
        probe.bias.detach().to(device=device, dtype=model_dtype)
        if probe.bias is not None
        else torch.zeros(W.shape[0], device=device, dtype=model_dtype)
    )

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
            # hidden_states[0] = embedding, hidden_states[i] = layer i output
            # flows 保存时 flow[i] 对应 hidden_states[i]，所以不需要 +1
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
                    h = captured_prenorm[-1][:, -1, :].to(W.dtype)
                else:
                    h = hidden_states[layer_idx][:, -1, :].to(W.dtype)

                if mode == "direct":
                    logits_probe = h @ W.t() + b
                    corrected_digit = int(torch.argmax(logits_probe, dim=1).item())
                else:
                    logits_probe = h @ W.t() + b
                    pred_class = int(torch.argmax(logits_probe, dim=1).item())
                    v = W[pred_class]
                    h_steered = h + lambd * v
                    logits_steered = h_steered @ W.t() + b
                    corrected_digit = int(torch.argmax(logits_steered, dim=1).item())

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
    parser = argparse.ArgumentParser(description="Linear probe for digit prediction.")
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["direct", "steer"], default="direct")
    parser.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--test-mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_full = load_dataset(args.dataset)
    positions = load_positions(args.h5)
    flows_all, _, _, gt_digits, pred_digits, sample_ids, pos_ids = build_flat_dataset(
        dataset_full,
        positions,
        positions_filter=args.positions,
    )

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
    sample_ids_val = sample_ids[val_mask] if val_mask.any() else sample_ids

    num_layers = flows_all.shape[1]
    candidate_layers = parse_layer_candidates(num_layers, args.layers, args.layer_start, args.layer_end)

    best_layer = None
    best_val_acc = -1.0
    best_model = None
    for layer in candidate_layers:
        model, val_acc = train_linear_probe(
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
        raise RuntimeError("Failed to train linear probe; no layer selected")

    if args.mode == "direct":
        if args.test_mode == "offline":
            corrected = predict_linear(best_model, flows_test[:, best_layer, :], device)
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
            corrected_token_acc, corrected_sample_acc, modified_rate, tp_correction, fp_preservation = online_eval(
                dataset_test,
                tokenizer,
                lm,
                best_model,
                best_layer,
                args.max_new_tokens,
                mode="direct",
                lambd=0.0,
                device=device,
            )
        payload = {
            "method": "linear_probe",
            "mode": "direct",
            "test_mode": args.test_mode,
            "layer": int(best_layer),
            "val_acc": float(best_val_acc),
            "corrected_token_acc": float(corrected_token_acc),
            "corrected_sample_acc": float(corrected_sample_acc),
            "modified_rate": float(modified_rate),
            "tp_correction": float(tp_correction),
            "fp_preservation": float(fp_preservation),
        }
    else:
        lambdas = [float(x.strip()) for x in args.lambda_grid.split(",") if x.strip()]
        best_lambda = None
        best_val_sample_acc = -1.0
        best_val_token_acc = -1.0
        for lambd in lambdas:
            corrected_val = steer_predict(best_model, flows_val[:, best_layer, :], lambd)
            val_token_acc, val_sample_acc = compute_token_sample_acc(
                corrected_val, gt_val, sample_ids_val
            )
            if val_sample_acc > best_val_sample_acc or (
                val_sample_acc == best_val_sample_acc and val_token_acc > best_val_token_acc
            ):
                best_val_sample_acc = val_sample_acc
                best_val_token_acc = val_token_acc
                best_lambda = lambd

        if best_lambda is None:
            best_lambda = 0.0

        if args.test_mode == "offline":
            corrected = steer_predict(best_model, flows_test[:, best_layer, :], best_lambda)
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
            corrected_token_acc, corrected_sample_acc, modified_rate, tp_correction, fp_preservation = online_eval(
                dataset_test,
                tokenizer,
                lm,
                best_model,
                best_layer,
                args.max_new_tokens,
                mode="steer",
                lambd=best_lambda,
                device=device,
            )
        payload = {
            "method": "linear_probe",
            "mode": "steer",
            "test_mode": args.test_mode,
            "layer": int(best_layer),
            "val_acc": float(best_val_acc),
            "lambda": float(best_lambda),
            "val_sample_acc": float(best_val_sample_acc),
            "val_token_acc": float(best_val_token_acc),
            "corrected_token_acc": float(corrected_token_acc),
            "corrected_sample_acc": float(corrected_sample_acc),
            "modified_rate": float(modified_rate),
            "tp_correction": float(tp_correction),
            "fp_preservation": float(fp_preservation),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
