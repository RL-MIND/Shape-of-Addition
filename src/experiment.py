import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.metrics import compute_mean_std_ci  # noqa: E402
from utils.probe_data import load_h5_baseline_metrics  # noqa: E402


COMMON_AGGREGATE_METRIC_KEYS = [
    "orig_eval_token_acc",
    "orig_eval_sample_acc",
    "orig_h5_token_acc",
    "orig_h5_sample_acc",
    "orig_token_acc",
    "orig_sample_acc",
    "corrected_token_acc",
    "corrected_sample_acc",
    "modified_rate",
    "tp_correction",
    "fp_preservation",
]

REPLACEMENT_PROMPT_AGGREGATE_METRIC_KEYS = [
    "val_acc",
    "off_by_one_count",
    "other_error_count",
    "off_by_one_ratio",
    "other_error_ratio",
]

STEERING_AGGREGATE_METRIC_KEYS = [
    "val_acc",
    "val_token_acc",
    "val_sample_acc",
]


def compute_spi(corrected: float, orig: float) -> float:
    if math.isnan(corrected) or math.isnan(orig):
        return float("nan")
    eps = 1e-8
    return (corrected - orig) / max(1.0 - orig, eps)


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_methods(method: str, test_mode: str) -> List[str]:
    if method == "all":
        methods = ["replacement", "steering", "dual-stream"]
        if test_mode == "online":
            methods.append("prompt")
        return methods

    return [method]


def resolve_aggregate_metric_keys(method: str) -> List[str]:
    if method in {"replacement", "prompt"}:
        return [*COMMON_AGGREGATE_METRIC_KEYS, *REPLACEMENT_PROMPT_AGGREGATE_METRIC_KEYS]
    if method == "steering":
        return [*COMMON_AGGREGATE_METRIC_KEYS, *STEERING_AGGREGATE_METRIC_KEYS]
    return COMMON_AGGREGATE_METRIC_KEYS


def aggregate_seed_payloads(seed_payloads: List[Dict[str, object]], method: str) -> Dict[str, object]:
    metric_keys = resolve_aggregate_metric_keys(method)
    metrics = {
        key: compute_mean_std_ci(
            [float(payload[key]) for payload in seed_payloads if key in payload and payload[key] is not None]
        )
        for key in metric_keys
    }
    return {
        "num_runs": int(len(seed_payloads)),
        "metrics": metrics,
    }


def append_common_args(
    cmd: list[str],
    args: argparse.Namespace,
    out_path: Path,
    seed: Optional[int] = None,
) -> list[str]:
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
            str(args.seed if seed is None else seed),
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


def append_dual_stream_args(
    cmd: list[str],
    args: argparse.Namespace,
    out_path: Path,
    seed: Optional[int] = None,
) -> list[str]:
    cmd = append_common_args(cmd, args, out_path, seed=seed)
    if args.layers and len(args.layers) == 1:
        layer_arg = str(args.layers[0])
        try:
            idx = cmd.index("--layers")
        except ValueError:
            return cmd
        cmd[idx:idx + 2] = ["--layers", layer_arg, layer_arg]
    cmd += ["--inertia-delta", str(args.inertia_delta)]
    return cmd


def resolve_output_path(method: str, out_dir: Path, timestamp: str) -> Path:
    if method == "replacement":
        return out_dir / f"replacement_{timestamp}.json"
    if method == "steering":
        return out_dir / f"steering_{timestamp}.json"
    if method == "dual-stream":
        return out_dir / f"dual_stream_{timestamp}.json"
    if method == "prompt":
        return out_dir / f"prompt_{timestamp}.json"
    raise ValueError(f"Unsupported method: {method}")


def resolve_seed_output_path(out_path: Path, seed: int) -> Path:
    return out_path.with_name(f"{out_path.stem}_seed{seed}{out_path.suffix}")


def build_method_argv(
    method: str,
    args: argparse.Namespace,
    out_path: Path,
    seed: Optional[int] = None,
) -> list[str]:
    if method == "replacement":
        return append_common_args([], args, out_path, seed=seed)
    if method == "steering":
        return append_common_args(
            ["--mode", "steer", "--lambda-grid", args.lambda_grid],
            args,
            out_path,
            seed=seed,
        )
    if method == "dual-stream":
        return append_dual_stream_args([], args, out_path, seed=seed)
    if method == "prompt":
        return append_common_args(["--mode", "prompt"], args, out_path, seed=seed)
    raise ValueError(f"Unsupported method: {method}")


def run_method(method: str, argv: list[str]) -> None:
    if method in {"replacement", "prompt"}:
        from src import mlp_probe

        mlp_probe.main(argv)
        return
    if method == "steering":
        from src import linear_probe

        linear_probe.main(argv)
        return
    if method == "dual-stream":
        from src import dualstream

        dualstream.main(argv)
        return
    raise ValueError(f"Unsupported method: {method}")


def is_aggregate_payload(metrics: Dict[str, object]) -> bool:
    return isinstance(metrics.get("aggregate"), dict)


def get_metric_value(metrics: Dict[str, object], key: str) -> float:
    if is_aggregate_payload(metrics):
        aggregate = metrics.get("aggregate", {})
        metric_stats = aggregate.get("metrics", {}).get(key, {})
        value = metric_stats.get("mean") if isinstance(metric_stats, dict) else None
    else:
        value = metrics.get(key)

    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_metric(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def print_metrics(metrics: Dict[str, object], skip_orig: bool) -> None:
    if is_aggregate_payload(metrics):
        print(f"num_runs: {int(metrics['aggregate'].get('num_runs', 0))}")

    corrected_token = get_metric_value(metrics, "corrected_token_acc")
    corrected_sample = get_metric_value(metrics, "corrected_sample_acc")
    if skip_orig:
        print(f"corrected_token_acc: {format_metric(corrected_token)}")
        print(f"corrected_sample_acc: {format_metric(corrected_sample)}")
        return

    orig_eval_token = get_metric_value(metrics, "orig_eval_token_acc")
    orig_eval_sample = get_metric_value(metrics, "orig_eval_sample_acc")
    orig_h5_token = get_metric_value(metrics, "orig_h5_token_acc")
    orig_h5_sample = get_metric_value(metrics, "orig_h5_sample_acc")
    improved_token = corrected_token - orig_eval_token
    improved_sample = corrected_sample - orig_eval_sample
    token_spi = compute_spi(corrected_token, orig_eval_token)
    sample_spi = compute_spi(corrected_sample, orig_eval_sample)

    print(f"orig_eval_token_acc: {format_metric(orig_eval_token)}")
    print(f"orig_eval_sample_acc: {format_metric(orig_eval_sample)}")
    print(f"orig_h5_token_acc: {format_metric(orig_h5_token)}")
    print(f"orig_h5_sample_acc: {format_metric(orig_h5_sample)}")
    print(f"corrected_token_acc: {format_metric(corrected_token)}")
    print(f"corrected_sample_acc: {format_metric(corrected_sample)}")
    print(f"improved_token_acc: {format_metric(improved_token)}")
    print(f"improved_sample_acc: {format_metric(improved_sample)}")
    print(f"token_spi: {format_metric(token_spi)}")
    print(f"sample_spi: {format_metric(sample_spi)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner.")
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("results/activations/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/num3len10-10000.pkl"))
    parser.add_argument("--positions", type=int, nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument(
        "--method",
        type=str,
        choices=["replacement", "steering", "dual-stream", "prompt", "all"],
        default="replacement",
        help="Correction method to run.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--test-mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--inertia-delta", type=float, default=0.0, help="Dual-stream carry tolerance window.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/logs/log_experiments"))
    parser.add_argument("--skip-orig", action="store_true", help="Skip baseline/improvement/SPI output")
    args = parser.parse_args(argv)

    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be a positive integer")

    methods = resolve_methods(args.method, args.test_mode)
    offline_multi_seed_methods = [method for method in methods if method != "dual-stream"]
    if args.num_seeds > 1 and args.test_mode != "online" and offline_multi_seed_methods:
        raise ValueError(
            "Multi-seed for non-dual-stream methods only supports --test-mode online; "
            f"got {args.test_mode} for methods: {', '.join(offline_multi_seed_methods)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_metrics = None
    if not args.skip_orig and args.num_seeds == 1:
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

        out_path = resolve_output_path(method, args.out_dir, timestamp)
        if method == "dual-stream":
            method_argv = build_method_argv(method, args, out_path, seed=args.seed)
            if args.num_seeds > 1:
                method_argv += ["--num-seeds", str(args.num_seeds)]
            run_method(method, method_argv)
            corrected_metrics = load_json(out_path)
        elif args.num_seeds == 1:
            method_argv = build_method_argv(method, args, out_path, seed=args.seed)
            run_method(method, method_argv)
            corrected_metrics = load_json(out_path)
        else:
            seed_runs: List[Dict[str, object]] = []
            for seed_offset in range(args.num_seeds):
                current_seed = args.seed + seed_offset
                print(f"Running seed {current_seed}...")
                seed_out_path = resolve_seed_output_path(out_path, current_seed)
                method_argv = build_method_argv(method, args, seed_out_path, seed=current_seed)
                run_method(method, method_argv)
                seed_metrics = load_json(seed_out_path)
                seed_metrics["seed"] = int(current_seed)
                seed_runs.append(seed_metrics)

            corrected_metrics = {
                "seed_start": int(args.seed),
                "num_seeds": int(args.num_seeds),
                "test_mode": args.test_mode,
                "seed_runs": seed_runs,
                "aggregate": aggregate_seed_payloads(seed_runs, method),
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(corrected_metrics, f, ensure_ascii=False, indent=2)
            print(f"Saved aggregated metrics to {out_path}")

        if not args.skip_orig and h5_metrics is not None and not is_aggregate_payload(corrected_metrics):
            corrected_metrics.setdefault("orig_h5_token_acc", h5_metrics["orig_h5_token_acc"])
            corrected_metrics.setdefault("orig_h5_sample_acc", h5_metrics["orig_h5_sample_acc"])
        print_metrics(corrected_metrics, args.skip_orig)


if __name__ == "__main__":
    main()
