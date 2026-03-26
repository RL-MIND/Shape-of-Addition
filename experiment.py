import argparse
from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from probe_data import load_h5_baseline_metrics


def compute_spi(corrected: float, orig: float) -> float:
    eps = 1e-8
    return (corrected - orig) / max(1.0 - orig, eps)


def run_script(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    if result.stdout:
        print(result.stdout, end="")


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_method(method: str) -> str:
    return "force" if method == "dual" else method


def resolve_methods(method: str, test_mode: str) -> List[str]:
    requested = normalize_method(method)
    if requested == "all":
        methods = ["mlp", "steer", "force", "prompt"]
        if test_mode == "teacher":
            skipped = ["prompt"]
            print(f"Skipping unsupported teacher methods: {', '.join(skipped)}")
            return ["mlp", "steer", "force"]
        return methods

    if test_mode == "teacher" and requested == "prompt":
        raise ValueError("teacher mode is not supported for method=prompt")
    return [requested]


def append_common_args(cmd: list[str], args: argparse.Namespace, out_path: Path) -> list[str]:
    cmd.extend(
        [
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
            "--test-mode",
            args.test_mode,
            "--model",
            args.model,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-samples",
            str(args.max_samples),
            "--output",
            str(out_path),
        ]
    )
    if args.positions:
        cmd += ["--positions", *[str(p) for p in args.positions]]
    if args.layers:
        cmd += ["--layers", *[str(l) for l in args.layers]]
    if args.layer_start is not None:
        cmd += ["--layer-start", str(args.layer_start)]
    if args.layer_end is not None:
        cmd += ["--layer-end", str(args.layer_end)]
    return cmd


def print_metrics(metrics: Dict[str, object], skip_orig: bool) -> None:
    corrected_token = float(metrics["corrected_token_acc"])
    corrected_sample = float(metrics["corrected_sample_acc"])
    if skip_orig:
        print(f"corrected_token_acc: {corrected_token:.4f}")
        print(f"corrected_sample_acc: {corrected_sample:.4f}")
        return

    orig_eval_token = float(metrics["orig_eval_token_acc"])
    orig_eval_sample = float(metrics["orig_eval_sample_acc"])
    orig_h5_token = float(metrics["orig_h5_token_acc"])
    orig_h5_sample = float(metrics["orig_h5_sample_acc"])
    improved_token = corrected_token - orig_eval_token
    improved_sample = corrected_sample - orig_eval_sample
    token_spi = compute_spi(corrected_token, orig_eval_token)
    sample_spi = compute_spi(corrected_sample, orig_eval_sample)

    print(f"orig_eval_token_acc: {orig_eval_token:.4f}")
    print(f"orig_eval_sample_acc: {orig_eval_sample:.4f}")
    print(f"orig_h5_token_acc: {orig_h5_token:.4f}")
    print(f"orig_h5_sample_acc: {orig_h5_sample:.4f}")
    print(f"corrected_token_acc: {corrected_token:.4f}")
    print(f"corrected_sample_acc: {corrected_sample:.4f}")
    print(f"improved_token_acc: {improved_token:.4f}")
    print(f"improved_sample_acc: {improved_sample:.4f}")
    print(f"token_spi: {token_spi:.4f}")
    print(f"sample_spi: {sample_spi:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner.")
    parser.add_argument("--h5", type=Path, default=Path("results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"))
    parser.add_argument("--dataset", type=Path, default=Path("num3len10-10000.pkl"))
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, choices=["mlp", "steer", "force", "dual", "prompt", "all"], default="all")
    parser.add_argument("--model", type=str, default="/data/Models/Qwen3-4b")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--test-mode", type=str, choices=["online", "offline", "teacher"], default="online")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--out-dir", type=Path, default=Path("log/log_experiments"))
    parser.add_argument("--force-script", "--dual-script", dest="force_script", type=Path, default=Path("dualstream_probe.py"))
    parser.add_argument("--linear-script", type=Path, default=Path("linear_probe.py"))
    parser.add_argument("--mlp-script", type=Path, default=Path("mlp_probe.py"))
    parser.add_argument("--skip-orig", action="store_true", help="Skip baseline/improvement/SPI output")
    args = parser.parse_args()

    methods = resolve_methods(args.method, args.test_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_metrics = None
    if not args.skip_orig:
        h5_metrics, _ = load_h5_baseline_metrics(
            args.h5,
            args.dataset,
            args.positions,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )
        print("=== H5 Baseline Reference ===")
        print(f"orig_h5_token_acc: {h5_metrics['orig_h5_token_acc']:.4f}")
        print(f"orig_h5_sample_acc: {h5_metrics['orig_h5_sample_acc']:.4f}")

    for method in methods:
        print("\n" + "=" * 60)
        print(f"Method: {method}")

        if method == "mlp":
            out_path = args.out_dir / f"mlp_probe_{timestamp}.json"
            cmd = append_common_args([sys.executable, str(args.mlp_script)], args, out_path)
        elif method == "steer":
            out_path = args.out_dir / f"linear_probe_steer_{timestamp}.json"
            cmd = append_common_args(
                [sys.executable, str(args.linear_script), "--mode", "steer", "--lambda-grid", args.lambda_grid],
                args,
                out_path,
            )
        elif method == "force":
            out_path = args.out_dir / f"dualstream_probe_{timestamp}.json"
            cmd = append_common_args([sys.executable, str(args.force_script)], args, out_path)
        elif method == "prompt":
            out_path = args.out_dir / f"mlp_probe_prompt_{timestamp}.json"
            cmd = append_common_args([sys.executable, str(args.mlp_script), "--mode", "prompt"], args, out_path)
        else:
            raise ValueError(f"Unsupported method: {method}")

        run_script(cmd)
        corrected_metrics = load_json(out_path)
        if not args.skip_orig and h5_metrics is not None:
            corrected_metrics.setdefault("orig_h5_token_acc", h5_metrics["orig_h5_token_acc"])
            corrected_metrics.setdefault("orig_h5_sample_acc", h5_metrics["orig_h5_sample_acc"])
        print_metrics(corrected_metrics, args.skip_orig)


if __name__ == "__main__":
    main()
