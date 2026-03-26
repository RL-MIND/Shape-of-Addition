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

from model_utils import (
    analyze_last_layer_normalized,
    apply_chat_template_safe,
    get_norm_module,
    get_norm_weight_from_model,
    rms_norm,
    setup_prenorm_hook,
)
from probe_data import (
    build_flat_dataset,
    compute_token_sample_acc,
    load_dataset,
    load_h5_baseline_metrics,
    load_positions,
    split_sample_ids,
)
from probe_utils import (
    get_digit_token_ids,
    inspect_teacher_layers,
    load_or_compute_teacher_features,
    online_baseline_eval,
    resolve_teacher_final_norm_local_index,
    resolve_selected_layers,
)


def mask_first_error_positions(
    pred_digits: np.ndarray,
    gt_digits: np.ndarray,
    sample_ids: np.ndarray,
    pos_ids: np.ndarray,
) -> np.ndarray:
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


def get_off_by_one_direction(pred_digit: int, gt_digit: int) -> Optional[str]:
    if not (0 <= pred_digit <= 9 and 0 <= gt_digit <= 9):
        return None
    diff = (pred_digit - gt_digit) % 10
    if diff == 9:
        return "minus_one"
    if diff == 1:
        return "plus_one"
    return None


def analyze_off_by_one_errors(pred_digits: np.ndarray, gt_digits: np.ndarray) -> dict[str, float | int]:
    pred_arr = np.asarray(pred_digits, dtype=np.int64)
    gt_arr = np.asarray(gt_digits, dtype=np.int64)
    orig_error_mask = pred_arr != gt_arr
    orig_error_count = int(np.sum(orig_error_mask))
    off_by_one_count = 0
    for idx in np.flatnonzero(orig_error_mask):
        if get_off_by_one_direction(int(pred_arr[idx]), int(gt_arr[idx])) is not None:
            off_by_one_count += 1
    other_error_count = int(orig_error_count - off_by_one_count)
    denom = orig_error_count if orig_error_count > 0 else 1
    return {
        "orig_error_count": int(orig_error_count),
        "off_by_one_count": int(off_by_one_count),
        "other_error_count": int(other_error_count),
        "off_by_one_ratio": float(off_by_one_count / denom) if orig_error_count > 0 else 0.0,
        "other_error_ratio": float(other_error_count / denom) if orig_error_count > 0 else 0.0,
    }


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

    model = ProbeMLP(X_train.shape[1], num_classes=10).to(device)
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
    n = len(corrected)
    if n == 0:
        return 0.0, 0.0, 0.0

    modified_rate = float(np.sum(corrected != pred_orig)) / n
    orig_errors = pred_orig != gt
    tp_total = np.sum(orig_errors)
    tp_correction = float(np.sum(orig_errors & (corrected == gt)) / tp_total) if tp_total > 0 else float("nan")

    orig_correct = pred_orig == gt
    fp_total = np.sum(orig_correct)
    fp_preservation = float(np.sum(orig_correct & (corrected == gt)) / fp_total) if fp_total > 0 else float("nan")
    return modified_rate, tp_correction, fp_preservation


def load_original_predictions_from_h5(h5_path: Path) -> dict[int, list[int]]:
    import h5py

    original_preds: dict[int, list[int]] = {}
    with h5py.File(h5_path, "r") as hf:
        positions_group = hf.get("all_token_results")
        if positions_group is None:
            return original_preds

        position_data = {}
        for pos_name, pos_group in positions_group.items():
            if pos_name == "extra":
                pos_idx = "extra"
            elif pos_name.startswith("pos_"):
                try:
                    pos_idx = int(pos_name.split("_")[1])
                except Exception:
                    continue
            else:
                continue

            sample_ids_ds = pos_group.get("sample_ids")
            preds_ds = pos_group.get("preds")
            if sample_ids_ds is None or preds_ds is None:
                continue

            sample_ids = np.asarray(sample_ids_ds[:])
            preds = np.asarray(preds_ds[:])
            if preds.dtype.kind in {"S", "O"}:
                preds = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in preds]
            else:
                preds = [str(p) for p in preds]
            position_data[pos_idx] = {"sample_ids": sample_ids, "preds": preds}

        all_sample_ids = set()
        for data in position_data.values():
            all_sample_ids.update(data["sample_ids"])

        numeric_positions = sorted([k for k in position_data.keys() if isinstance(k, int)])
        for sample_id in sorted(all_sample_ids):
            digits = []
            for pos_idx in numeric_positions:
                data = position_data[pos_idx]
                mask = data["sample_ids"] == sample_id
                if mask.any():
                    pred_str = data["preds"][np.where(mask)[0][0]]
                    if pred_str.strip().isdigit():
                        digits.append(int(pred_str.strip()))
            if digits:
                original_preds[int(sample_id)] = digits

    return original_preds


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
    log_file: Optional[Path] = None,
    h5_path: Optional[Path] = None,
    test_ids: Optional[set[int]] = None,
) -> Tuple[float, float, float, float, float]:
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    original_predictions = {}
    if h5_path is not None and h5_path.exists():
        original_predictions = load_original_predictions_from_h5(h5_path)

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

    log_entries = [] if log_file is not None else None
    test_ids_list = sorted(test_ids) if test_ids else list(range(len(dataset)))

    probe.eval()
    model.eval()
    with torch.no_grad():
        for dataset_idx, operands in enumerate(dataset):
            h5_sample_id = test_ids_list[dataset_idx] if dataset_idx < len(test_ids_list) else dataset_idx
            gt_val = sum(operands)
            gt_str = str(gt_val)

            expr = " + ".join(str(x) for x in operands)
            messages = [{"role": "user", "content": f"Calculate {expr}. Only output a number."}]
            prefix = apply_chat_template_safe(tokenizer, messages) + expr + " = "

            model_inputs = tokenizer([prefix], return_tensors="pt").to(device)
            captured_prenorm.clear()
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            max_layer = len(outputs.hidden_states) - 1
            layer_idx = max_layer if layer < 0 else min(layer, max_layer)

            original_digits = original_predictions.get(h5_sample_id, [])
            generated_digits: List[int] = []
            current_output_str = ""

            step = 0
            while step < max_new_tokens:
                logits = outputs.logits[:, -1, :]
                d_pred = select_digit(logits)
                hidden_states = outputs.hidden_states

                if layer_idx == max_layer and captured_prenorm:
                    h = captured_prenorm[-1][:, -1, :]
                else:
                    h = hidden_states[layer_idx][:, -1, :]

                if norm_weight is not None:
                    h = rms_norm(h, norm_weight.to(h.device), norm_eps)

                probe_dtype = next(probe.parameters()).dtype
                if h.dtype != probe_dtype:
                    h = h.to(dtype=probe_dtype)
                logits_probe = probe(h)
                probe_digit = int(torch.argmax(logits_probe, dim=1).item())

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

                if probe_digit != d_pred:
                    modified_count += 1
                    correction_text = f" {d_pred} is incorrect. Let me recalculate: {expr} = " + current_output_str[:-1]
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

                    logits_new = outputs.logits[:, -1, :]
                    corrected_digit = select_digit(logits_new)
                    generated_digits[-1] = corrected_digit
                    current_output_str = current_output_str[:-1] + str(corrected_digit)

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
            for idx in range(max(g_len, t_len)):
                token_total += 1
                gt_digit = int(gt_str[idx]) if idx < t_len else -1
                orig_digit = original_digits[idx] if idx < o_len else -1
                corr_digit = generated_digits[idx] if idx < g_len else -1

                if idx < t_len and idx < g_len and corr_digit == gt_digit:
                    token_correct += 1
                if idx < t_len and idx < o_len and orig_digit != gt_digit:
                    tp_total += 1
                    if idx < g_len and corr_digit == gt_digit:
                        tp_corrected += 1
                if idx < t_len and idx < o_len and orig_digit == gt_digit:
                    fp_total += 1
                    if idx < g_len and corr_digit == gt_digit:
                        fp_preserved += 1

            corrected_ok = g_len == t_len and all(generated_digits[i] == int(gt_str[i]) for i in range(t_len))
            if corrected_ok:
                sample_correct += 1

            if log_entries is not None:
                log_entries.append(
                    {
                        "dataset_idx": int(dataset_idx),
                        "h5_sample_id": int(h5_sample_id),
                        "operands": [int(op) for op in operands],
                        "expression": expr,
                        "ground_truth": gt_str,
                        "original_output": "".join(str(d) for d in original_digits),
                        "corrected_output": "".join(str(d) for d in generated_digits),
                        "corrected_correct": corrected_ok,
                    }
                )

    if hook_handle is not None:
        hook_handle.remove()

    if log_entries:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, ensure_ascii=False, indent=2)

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
    mode: str = "direct",
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    log_file: Optional[Path] = None,
    h5_path: Optional[Path] = None,
    test_ids: Optional[set[int]] = None,
) -> Tuple[float, float, float, float, float]:
    if mode == "prompt":
        return online_eval_prompt(
            dataset,
            tokenizer,
            model,
            probe,
            layer,
            max_new_tokens,
            device,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            log_file=log_file,
            h5_path=h5_path,
            test_ids=test_ids,
        )

    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

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
            text = apply_chat_template_safe(tokenizer, messages)
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
                    h = captured_prenorm[-1][:, -1, :]
                else:
                    h = hidden_states[layer_idx][:, -1, :]

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
    base_name = "mlp_probe_prompt" if mode == "prompt" else "mlp_probe"
    return Path("log/log_experiments") / f"{base_name}_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP probe for digit prediction.")
    parser.add_argument("--h5", type=Path, default=Path("results/plus_num3len10_gemma-3-4b-it/plus_num3len10_gemma-3-4b-it.h5"))
    parser.add_argument("--dataset", type=Path, default=Path("num3len10-10000.pkl"), help="Dataset used for generation")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=[-1])
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--test-mode", type=str, choices=["online", "offline", "teacher"], default="online")
    parser.add_argument("--mode", type=str, choices=["direct", "prompt"], default="direct")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.output = resolve_output_path(args.output, args.mode)

    if args.mode == "prompt" and args.test_mode != "online":
        raise ValueError("prompt mode only supports --test-mode online")

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

    if args.test_mode == "teacher":
        dummy_sample_ids = np.arange(len(dataset_full))
        train_ids_all, val_ids_all, test_ids_all = split_sample_ids(
            dummy_sample_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        tokenizer_tf = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        lm_tf = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype="auto",
            output_hidden_states=True,
            do_sample=False,
        )
        last_layer_normalized = analyze_last_layer_normalized(lm_tf, tokenizer_tf)
        teacher_total_layers, teacher_diag = inspect_teacher_layers(lm_tf, tokenizer_tf)
        teacher_candidate_layers = parse_layer_candidates(
            teacher_total_layers,
            args.layers,
            args.layer_start,
            args.layer_end,
        )
        selected_teacher_layers = resolve_selected_layers(teacher_candidate_layers, teacher_total_layers)
        print(
            "Teacher layer diagnostics: "
            f"config_class={teacher_diag['config_class']} | "
            f"config.num_hidden_layers={teacher_diag['config_num_hidden_layers']} | "
            f"text_config.num_hidden_layers={teacher_diag['text_config_num_hidden_layers']} | "
            f"hidden_states_len={teacher_diag['hidden_states_len']} | "
            f"layer_source={teacher_diag['layer_source']} | "
            f"requested_layers={teacher_candidate_layers} | "
            f"resolved_layers={selected_teacher_layers if selected_teacher_layers is not None else 'all'}"
        )
        if teacher_diag["forward_error"] is not None:
            print(f"Teacher layer forward probe fallback: {teacher_diag['forward_error']}")
        flows_all, _, _, gt_digits, pred_digits, sample_ids, pos_ids = load_or_compute_teacher_features(
            dataset_full,
            dataset_path=args.dataset,
            tokenizer=tokenizer_tf,
            model=lm_tf,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device=device,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            use_prenorm=last_layer_normalized,
            valid_indices=train_ids_all.union(val_ids_all).union(test_ids_all),
            positions=args.positions,
            max_samples=args.max_samples,
            selected_layers=selected_teacher_layers,
        )
        del tokenizer_tf
        del lm_tf
        torch.cuda.empty_cache()
    else:
        teacher_total_layers = None
        selected_teacher_layers = None
        positions = load_positions(args.h5)
        flows_all, _, _, gt_digits, pred_digits, sample_ids, pos_ids = build_flat_dataset(
            dataset_full,
            positions,
            positions_filter=args.positions,
        )

    num_layers = flows_all.shape[1]
    if args.test_mode == "teacher":
        norm_layer_idx = resolve_teacher_final_norm_local_index(selected_teacher_layers, teacher_total_layers)
        if norm_layer_idx is not None:
            print(
                f"Applying RMSNorm to flows (teacher final layer local={norm_layer_idx}, "
                f"global={teacher_total_layers - 1})..."
            )
            flows_all = apply_rms_norm_to_flows(flows_all, norm_weight, norm_layer_idx, norm_eps)
        else:
            print(
                "Skipping RMSNorm on teacher flows because cached layers do not include "
                f"the final layer {teacher_total_layers - 1}."
            )
    else:
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
    error_stats = analyze_off_by_one_errors(pred_test, gt_test)

    if args.test_mode == "teacher" and selected_teacher_layers is not None:
        candidate_layers = list(range(len(selected_teacher_layers)))
    else:
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

    report_layer = (
        int(selected_teacher_layers[best_layer])
        if args.test_mode == "teacher" and selected_teacher_layers is not None
        else int(best_layer)
    )

    tokenizer = None
    lm = None
    dataset_test = None
    if args.test_mode in {"online"}:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype="auto",
            output_hidden_states=True,
            do_sample=False,
        )
        dataset_test = [dataset_full[i] for i in sorted(test_ids) if 0 <= i < len(dataset_full)] if test_ids else dataset_full
        norm_module = get_norm_module(lm)
        if norm_module is not None:
            online_norm_weight = norm_module.weight.detach().clone()
            online_norm_eps = getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6))
        else:
            online_norm_weight = norm_weight
            online_norm_eps = norm_eps

        log_file_path = args.output.parent / f"{args.output.stem}_detail.json" if args.mode == "prompt" else None
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
            log_file=log_file_path,
            h5_path=args.h5,
            test_ids=test_ids,
        )
        orig_eval_token_acc, orig_eval_sample_acc = online_baseline_eval(
            dataset_test,
            tokenizer,
            lm,
            args.max_new_tokens,
            device,
        )
    else:
        corrected = predict_mlp(best_model, flows_test[:, best_layer, :], device)
        corrected_token_acc, corrected_sample_acc = compute_token_sample_acc(corrected, gt_test, sample_ids_test)
        modified_rate, tp_correction, fp_preservation = compute_correction_metrics(corrected, pred_test, gt_test)
        orig_eval_token_acc, orig_eval_sample_acc = compute_token_sample_acc(pred_test, gt_test, sample_ids_test)

    if args.test_mode == "teacher":
        orig_h5_token_acc = float(h5_metrics["orig_h5_token_acc"])
        orig_h5_sample_acc = float(h5_metrics["orig_h5_sample_acc"])
    else:
        orig_h5_token_acc, orig_h5_sample_acc = compute_token_sample_acc(pred_test, gt_test, sample_ids_test)

    payload = {
        "method": "mlp_probe",
        "test_mode": args.test_mode,
        "mode": args.mode,
        "layer": report_layer,
        "val_acc": float(best_val_acc),
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
        "off_by_one_count": int(error_stats["off_by_one_count"]),
        "other_error_count": int(error_stats["other_error_count"]),
        "off_by_one_ratio": float(error_stats["off_by_one_ratio"]),
        "other_error_ratio": float(error_stats["other_error_ratio"]),
    }

    print("\n=== MLP Probe Results ===")
    print(f"Best layer: {report_layer}")
    print(f"Mode: {args.mode}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Orig eval token accuracy: {orig_eval_token_acc:.4f}")
    print(f"Orig eval sample accuracy: {orig_eval_sample_acc:.4f}")
    print(f"Orig H5 token accuracy: {orig_h5_token_acc:.4f}")
    print(f"Orig H5 sample accuracy: {orig_h5_sample_acc:.4f}")
    print(f"Corrected token accuracy: {corrected_token_acc:.4f}")
    print(f"Corrected sample accuracy: {corrected_sample_acc:.4f}")
    print(f"Modified rate: {modified_rate:.4f}")
    print(f"TP Correction: {tp_correction:.4f}")
    print(f"FP Preservation: {fp_preservation:.4f}")
    print(
        "Original error mix: "
        f"off_by_one={error_stats['off_by_one_count']} ({error_stats['off_by_one_ratio']:.4f}) | "
        f"other={error_stats['other_error_count']} ({error_stats['other_error_ratio']:.4f})"
    )

    if args.mode == "prompt" and args.test_mode == "online":
        log_file_path = args.output.parent / f"{args.output.stem}_detail.json"
        if log_file_path.exists():
            print(f"\nDetailed prompt log saved to: {log_file_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
