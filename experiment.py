import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_data import (
    build_flat_dataset,
    compute_token_sample_acc,
    load_dataset,
    load_positions,
    split_sample_ids,
)


def compute_spi(corrected: float, orig: float) -> float:
    eps = 1e-8
    return (corrected - orig) / max(1.0 - orig, eps)


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


def online_baseline_eval(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[float, float]:
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    token_total = 0
    token_correct = 0
    sample_total = 0
    sample_correct = 0

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
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values

            generated_digits: List[int] = []
            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]
                chosen_digit = select_digit(logits)
                generated_digits.append(chosen_digit)

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
            for i in range(max(g_len, t_len)):
                token_total += 1
                if i < t_len and i < g_len and generated_digits[i] == int(gt_str[i]):
                    token_correct += 1
            if g_len == t_len and all(generated_digits[i] == int(gt_str[i]) for i in range(t_len)):
                sample_correct += 1

    token_acc = token_correct / token_total if token_total else 0.0
    sample_acc = sample_correct / sample_total if sample_total else 0.0
    return token_acc, sample_acc


def load_baseline_metrics(
    h5_path: Path,
    dataset_path: Path,
    positions: list[int] | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    compute_orig: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    dataset_full = load_dataset(dataset_path)
    positions_data = load_positions(h5_path)
    flows_all, _, _, gt_digits, pred_digits, sample_ids, _ = build_flat_dataset(
        dataset_full,
        positions_data,
        positions_filter=positions,
    )

    train_ids, val_ids, test_ids = split_sample_ids(
        sample_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    test_mask = np.isin(sample_ids, list(test_ids)) if test_ids else np.zeros_like(sample_ids, dtype=bool)
    gt_test = gt_digits[test_mask] if test_mask.any() else gt_digits
    pred_test = pred_digits[test_mask] if test_mask.any() else pred_digits
    sample_ids_test = sample_ids[test_mask] if test_mask.any() else sample_ids

    if compute_orig:
        token_acc, sample_acc = compute_token_sample_acc(pred_test, gt_test, sample_ids_test)
        orig_token_acc = float(token_acc)
        orig_sample_acc = float(sample_acc)
    else:
        orig_token_acc = float("nan")
        orig_sample_acc = float("nan")
    metrics = {
        "orig_token_acc": orig_token_acc,
        "orig_sample_acc": orig_sample_acc,
        "test_tokens": int(len(gt_test)),
        "test_samples": int(len(np.unique(sample_ids_test))),
    }
    arrays = {
        "gt_test": gt_test,
        "pred_test": pred_test,
        "sample_ids_test": sample_ids_test,
        "test_mask": test_mask,
        "test_ids": test_ids,
    }
    return metrics, arrays


def run_script(cmd: list[str]) -> Dict[str, object]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return {}


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner.")
    parser.add_argument("--h5", type=Path, default=Path("VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"))
    parser.add_argument("--dataset", type=Path, default=Path("VerticalFlow/num3len10-10000.pkl"))
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, choices=["linear", "mlp", "steer", "force", "prompt", "all"], default="all")
    parser.add_argument("--model", type=str, default="/data/Models/Qwen3-4b")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--out-dir", type=Path, default=Path("VerticalFlow/log/log_experiments"))
    parser.add_argument("--force-script", type=Path, default=Path("VerticalFlow/force_probe.py"))
    parser.add_argument("--linear-script", type=Path, default=Path("VerticalFlow/linear_probe.py"))
    parser.add_argument("--mlp-script", type=Path, default=Path("VerticalFlow/mlp_probe.py"))
    parser.add_argument("--skip-orig", action="store_true", help="Skip original acc and SPI computations")
    args = parser.parse_args()

    methods = [args.method] if args.method != "all" else ["linear", "mlp", "steer", "force", "prompt"]

    base_metrics, arrays = load_baseline_metrics(
        args.h5,
        args.dataset,
        args.positions,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        compute_orig=not args.skip_orig,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = None
    tokenizer = None
    lm = None
    need_local_model = not args.skip_orig
    if need_local_model:
        if args.model is None:
            raise ValueError("--model is required for online test mode")
        dataset_full = load_dataset(args.dataset)
        # 保护：若 test_ids 为空，则退化为使用全数据（避免空测试集影响评估）
        test_ids = arrays.get("test_ids")
        if test_ids:
            dataset_test = [dataset_full[i] for i in sorted(test_ids) if 0 <= i < len(dataset_full)]
        else:
            dataset_test = dataset_full
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype="auto",
            output_hidden_states=True,
        )

    if not args.skip_orig:
        base_token_acc, base_sample_acc = online_baseline_eval(
            dataset_test,
            tokenizer,
            lm,
            args.max_new_tokens,
            device,
        )
        base_metrics["orig_token_acc"] = float(base_token_acc)
        base_metrics["orig_sample_acc"] = float(base_sample_acc)
        print("=== Baseline (original model, online) ===")
        print(f"token_acc: {base_metrics['orig_token_acc']:.4f}")
        print(f"sample_acc: {base_metrics['orig_sample_acc']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        print("\n" + "=" * 60)
        print(f"Method: {method}")
        if method == "linear":
            out_path = args.out_dir / "linear_probe.json"
            cmd = [
                sys.executable,
                str(args.linear_script),
                "--h5",
                str(args.h5),
                "--dataset",
                str(args.dataset),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
                "--seed",
                str(args.seed),
                "--output",
                str(out_path),
            ]
            cmd += ["--test-mode", "online", "--model", args.model, "--max-new-tokens", str(args.max_new_tokens)]
            if args.positions:
                cmd += ["--positions", *[str(p) for p in args.positions]]
            if args.layers:
                cmd += ["--layers", *[str(l) for l in args.layers]]
            if args.layer_start is not None:
                cmd += ["--layer-start", str(args.layer_start)]
            if args.layer_end is not None:
                cmd += ["--layer-end", str(args.layer_end)]
            run_script(cmd)
            corrected_metrics = load_json(out_path)
        elif method == "mlp":
            out_path = args.out_dir / "mlp_probe.json"
            cmd = [
                sys.executable,
                str(args.mlp_script),
                "--h5",
                str(args.h5),
                "--dataset",
                str(args.dataset),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
                "--seed",
                str(args.seed),
                "--output",
                str(out_path),
            ]
            cmd += ["--test-mode", "online", "--model", args.model, "--max-new-tokens", str(args.max_new_tokens)]
            if args.positions:
                cmd += ["--positions", *[str(p) for p in args.positions]]
            if args.layers:
                cmd += ["--layers", *[str(l) for l in args.layers]]
            if args.layer_start is not None:
                cmd += ["--layer-start", str(args.layer_start)]
            if args.layer_end is not None:
                cmd += ["--layer-end", str(args.layer_end)]
            run_script(cmd)
            corrected_metrics = load_json(out_path)
        elif method == "steer":
            out_path = args.out_dir / "linear_probe_steer.json"
            cmd = [
                sys.executable,
                str(args.linear_script),
                "--h5",
                str(args.h5),
                "--dataset",
                str(args.dataset),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
                "--seed",
                str(args.seed),
                "--mode",
                "steer",
                "--lambda-grid",
                args.lambda_grid,
                "--output",
                str(out_path),
            ]
            cmd += ["--test-mode", "online", "--model", args.model, "--max-new-tokens", str(args.max_new_tokens)]
            if args.positions:
                cmd += ["--positions", *[str(p) for p in args.positions]]
            if args.layers:
                cmd += ["--layers", *[str(l) for l in args.layers]]
            if args.layer_start is not None:
                cmd += ["--layer-start", str(args.layer_start)]
            if args.layer_end is not None:
                cmd += ["--layer-end", str(args.layer_end)]
            run_script(cmd)
            corrected_metrics = load_json(out_path)
        elif method == "force":
            out_path = args.out_dir / "force_probe.json"
            cmd = [
                sys.executable,
                str(args.force_script),
                "--h5",
                str(args.h5),
                "--dataset",
                str(args.dataset),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
                "--seed",
                str(args.seed),
                "--output",
                str(out_path),
            ]
            cmd += ["--test-mode", "online", "--model", args.model, "--max-new-tokens", str(args.max_new_tokens)]
            if args.positions:
                cmd += ["--positions", *[str(p) for p in args.positions]]
            run_script(cmd)
            corrected_metrics = load_json(out_path)
        elif method == "prompt":
            out_path = args.out_dir / "mlp_probe_prompt.json"
            cmd = [
                sys.executable,
                str(args.mlp_script),
                "--h5",
                str(args.h5),
                "--dataset",
                str(args.dataset),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
                "--seed",
                str(args.seed),
                "--mode",
                "prompt",
                "--output",
                str(out_path),
            ]
            cmd += ["--test-mode", "online", "--model", args.model, "--max-new-tokens", str(args.max_new_tokens)]
            if args.positions:
                cmd += ["--positions", *[str(p) for p in args.positions]]
            if args.layers:
                cmd += ["--layers", *[str(l) for l in args.layers]]
            if args.layer_start is not None:
                cmd += ["--layer-start", str(args.layer_start)]
            if args.layer_end is not None:
                cmd += ["--layer-end", str(args.layer_end)]
            run_script(cmd)
            corrected_metrics = load_json(out_path)
        else:
            raise ValueError(f"Unsupported method: {method}")

        corrected_token = float(corrected_metrics["corrected_token_acc"])
        corrected_sample = float(corrected_metrics["corrected_sample_acc"])
        if args.skip_orig:
            print(f"corrected_token_acc: {corrected_token:.4f}")
            print(f"corrected_sample_acc: {corrected_sample:.4f}")
        else:
            improved_token = corrected_token - base_metrics["orig_token_acc"]
            improved_sample = corrected_sample - base_metrics["orig_sample_acc"]
            token_spi = compute_spi(corrected_token, base_metrics["orig_token_acc"])
            sample_spi = compute_spi(corrected_sample, base_metrics["orig_sample_acc"])

            print(f"orig_token_acc: {base_metrics['orig_token_acc']:.4f}")
            print(f"orig_sample_acc: {base_metrics['orig_sample_acc']:.4f}")
            print(f"corrected_token_acc: {corrected_token:.4f}")
            print(f"corrected_sample_acc: {corrected_sample:.4f}")
            print(f"improved_token_acc: {improved_token:.4f}")
            print(f"improved_sample_acc: {improved_sample:.4f}")
            print(f"token_spi: {token_spi:.4f}")
            print(f"sample_spi: {sample_spi:.4f}")


if __name__ == "__main__":
    main()
