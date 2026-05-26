import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.metrics import compute_correction_metrics, mask_first_error_positions

from utils.model_utils import (
    analyze_last_layer_normalized,
    apply_chat_template_safe,
    detect_model_type,
    get_norm_module,
    get_norm_weight_from_model,
    rms_norm,
    setup_prenorm_hook,
)
from utils.probe_data import (
    build_flat_dataset,
    compute_token_sample_acc,
    load_h5_sample_dataset,
    load_dataset,
    load_h5_baseline_metrics,
    load_positions,
    split_sample_ids,
)
from utils.probe_utils import (
    get_digit_token_ids,
    online_baseline_eval,
    parse_layer_candidates,
    resolve_selected_layers,
)


def apply_rms_norm_to_flows(
    flows: np.ndarray,
    norm_weight: torch.Tensor,
    layer_idx: int,
    eps: float = 1e-6,
) -> np.ndarray:
    flows_tensor = torch.tensor(flows[:, layer_idx, :], dtype=torch.float32)
    weight = norm_weight.float().cpu()
    normed = rms_norm(flows_tensor, weight, eps)
    flows_out = flows.copy()
    flows_out[:, layer_idx, :] = normed.numpy()
    return flows_out


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

    probe = nn.Linear(X_train.shape[1], 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    best_val = -1.0
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

        probe.eval()
        preds: List[int] = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                preds.extend(torch.argmax(probe(xb), dim=1).cpu().numpy())
        val_acc = float(np.mean(preds == y_val_t.numpy())) if len(y_val_t) else float("nan")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, best_val


def predict_linear(model: nn.Linear, X: np.ndarray, device: torch.device) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.int64)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)


def steer_predict(model: nn.Linear, X: np.ndarray, lambd: float) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.int64)
    weights = model.weight.detach().cpu().numpy()
    bias = model.bias.detach().cpu().numpy() if model.bias is not None else np.zeros(weights.shape[0], dtype=weights.dtype)
    logits = X @ weights.T + bias
    pred_class = np.argmax(logits, axis=1)
    directions = weights[pred_class]
    x_steered = X + lambd * directions
    logits_steered = x_steered @ weights.T + bias
    return np.argmax(logits_steered, axis=1).astype(np.int64)


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
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    model_type: Optional[str] = None,
) -> Tuple[float, float, float, float, float]:
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    model_dtype = model.dtype if hasattr(model, "dtype") else torch.float32
    weights = probe.weight.detach().to(device=device, dtype=model_dtype)
    bias = (
        probe.bias.detach().to(device=device, dtype=model_dtype)
        if probe.bias is not None
        else torch.zeros(weights.shape[0], device=device, dtype=model_dtype)
    )

    captured_prenorm, hook_handle = setup_prenorm_hook(model)

    token_total = 0
    token_correct = 0
    sample_total = 0
    sample_correct = 0
    modified_count = 0
    tp_total = 0
    tp_corrected = 0
    fp_total = 0
    fp_preserved = 0

    probe.eval()
    model.eval()
    with torch.no_grad():
        for operands in dataset:
            gt_val = sum(operands)
            gt_str = str(gt_val)

            expr = " + ".join(str(x) for x in operands)
            messages = [{"role": "user", "content": f"Calculate {expr}. Only output a number."}]
            text = apply_chat_template_safe(tokenizer, messages, model_type)
            text = text + expr + " = "

            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            captured_prenorm.clear()
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            max_layer = len(outputs.hidden_states) - 1
            layer_idx = max_layer if layer < 0 else min(layer, max_layer)

            generated_digits: List[int] = []
            original_digits: List[int] = []
            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]
                d_pred = select_digit(logits)
                original_digits.append(d_pred)
                hidden_states = outputs.hidden_states

                if layer_idx == max_layer and captured_prenorm:
                    h = captured_prenorm[-1][:, -1, :].to(weights.dtype)
                else:
                    h = hidden_states[layer_idx][:, -1, :].to(weights.dtype)

                if norm_weight is not None and layer_idx == max_layer:
                    h = rms_norm(h, norm_weight.to(h.device).to(h.dtype), norm_eps)

                logits_probe = h @ weights.t() + bias
                if mode == "direct":
                    corrected_digit = int(torch.argmax(logits_probe, dim=1).item())
                else:
                    pred_class = int(torch.argmax(logits_probe, dim=1).item())
                    steer_vec = weights[pred_class]
                    h_steered = h + lambd * steer_vec
                    logits_steered = h_steered @ weights.t() + bias
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
            for idx in range(max(g_len, t_len)):
                token_total += 1
                gt_digit = int(gt_str[idx]) if idx < t_len else -1
                orig_digit = original_digits[idx] if idx < o_len else -1
                corr_digit = generated_digits[idx] if idx < g_len else -1

                if idx < t_len and idx < g_len and corr_digit == gt_digit:
                    token_correct += 1
                if idx < o_len and idx < g_len and corr_digit != orig_digit:
                    modified_count += 1
                if idx < t_len and idx < o_len and orig_digit != gt_digit:
                    tp_total += 1
                    if idx < g_len and corr_digit == gt_digit:
                        tp_corrected += 1
                if idx < t_len and idx < o_len and orig_digit == gt_digit:
                    fp_total += 1
                    if idx < g_len and corr_digit == gt_digit:
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


def resolve_output_path(output: Optional[Path], mode: str) -> Path:
    if output is not None:
        return output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = "linear_probe_steer" if mode == "steer" else "linear_probe"
    return Path("results/logs/log_experiments") / f"{base_name}_{timestamp}.json"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Linear probe for digit prediction.")
    parser.add_argument("--h5", type=Path, default=Path("results/activations/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"))
    parser.add_argument("--dataset", type=Path, default=Path("data/num3len10-10000.pkl"))
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=[-1])
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["direct", "steer"], default="direct")
    parser.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--test-mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    args.output = resolve_output_path(args.output, args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model is None:
        raise ValueError("--model is required to load RMSNorm parameters")

    print(f"Loading RMSNorm parameters from {args.model}...")
    norm_weight, norm_eps = get_norm_weight_from_model(args.model, device)
    print(f"RMSNorm eps: {norm_eps}")

    dataset_full = load_dataset(args.dataset)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer")
        original_size = len(dataset_full)
        dataset_full = dataset_full[: args.max_samples]
        print(f"Using dataset subset: {len(dataset_full)}/{original_size} samples")
    h5_metrics, _ = load_h5_baseline_metrics(
        args.h5,
        args.dataset,
        args.positions,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    positions = load_positions(args.h5)
    flows_all, _, _, gt_digits, pred_digits, sample_ids, pos_ids = build_flat_dataset(
        dataset_full,
        positions,
        positions_filter=args.positions,
    )

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
    sample_ids_val = sample_ids[val_mask] if val_mask.any() else sample_ids

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

    report_layer = int(best_layer)

    if args.mode == "steer":
        lambdas = [float(x.strip()) for x in args.lambda_grid.split(",") if x.strip()]
        best_lambda = None
        best_val_sample_acc = -1.0
        best_val_token_acc = -1.0
        for lambd in lambdas:
            corrected_val = steer_predict(best_model, flows_val[:, best_layer, :], lambd)
            val_token_acc, val_sample_acc = compute_token_sample_acc(corrected_val, gt_val, sample_ids_val)
            if val_sample_acc > best_val_sample_acc or (
                val_sample_acc == best_val_sample_acc and val_token_acc > best_val_token_acc
            ):
                best_val_sample_acc = val_sample_acc
                best_val_token_acc = val_token_acc
                best_lambda = lambd
        if best_lambda is None:
            best_lambda = 0.0
    else:
        best_lambda = 0.0
        best_val_sample_acc = float("nan")
        best_val_token_acc = float("nan")

    tokenizer = None
    lm = None
    dataset_test = None
    if args.test_mode == "online":
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype="auto",
            output_hidden_states=True,
        )
        h5_dataset = load_h5_sample_dataset(args.h5)
        if h5_dataset and test_ids:
            dataset_test = [h5_dataset[i] for i in sorted(test_ids) if i in h5_dataset]
        else:
            dataset_test = [dataset_full[i] for i in sorted(test_ids) if 0 <= i < len(dataset_full)] if test_ids else dataset_full
        norm_module = get_norm_module(lm)
        if norm_module is not None:
            online_norm_weight = norm_module.weight.detach().clone()
            online_norm_eps = getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6))
        else:
            online_norm_weight = norm_weight
            online_norm_eps = norm_eps
        model_type = detect_model_type(args.model)
        corrected_token_acc, corrected_sample_acc, modified_rate, tp_correction, fp_preservation = online_eval(
            dataset_test,
            tokenizer,
            lm,
            best_model,
            best_layer,
            args.max_new_tokens,
            mode=args.mode,
            lambd=best_lambda,
            device=device,
            norm_weight=online_norm_weight,
            norm_eps=online_norm_eps,
            model_type=model_type,
        )
        orig_eval_token_acc, orig_eval_sample_acc = online_baseline_eval(
            dataset_test,
            tokenizer,
            lm,
            args.max_new_tokens,
            device,
            model_type=model_type,
        )
    else:
        corrected = (
            predict_linear(best_model, flows_test[:, best_layer, :], device)
            if args.mode == "direct"
            else steer_predict(best_model, flows_test[:, best_layer, :], best_lambda)
        )
        corrected_token_acc, corrected_sample_acc = compute_token_sample_acc(corrected, gt_test, sample_ids_test)
        modified_rate, tp_correction, fp_preservation = compute_correction_metrics(corrected, pred_test, gt_test)
        orig_eval_token_acc, orig_eval_sample_acc = compute_token_sample_acc(pred_test, gt_test, sample_ids_test)

    orig_h5_token_acc, orig_h5_sample_acc = compute_token_sample_acc(pred_test, gt_test, sample_ids_test)

    payload = {
        "method": "linear_probe",
        "mode": args.mode,
        "test_mode": args.test_mode,
        "layer": report_layer,
        "val_acc": float(best_val_acc),
        "lambda": float(best_lambda),
        "val_sample_acc": float(best_val_sample_acc),
        "val_token_acc": float(best_val_token_acc),
        "orig_eval_token_acc": float(orig_eval_token_acc),
        "orig_eval_sample_acc": float(orig_eval_sample_acc),
        "orig_h5_token_acc": float(orig_h5_token_acc),
        "orig_h5_sample_acc": float(orig_h5_sample_acc),
        "orig_token_acc": float(orig_h5_token_acc),
        "orig_sample_acc": float(orig_h5_sample_acc),
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
