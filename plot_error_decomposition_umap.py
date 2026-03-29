import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "results" / "error_decomposition" / "probe_error_decomposition_pos4.csv"
DEFAULT_JSON = REPO_ROOT / "results" / "error_decomposition" / "probe_error_decomposition_pos4.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "error_decomposition" / "probe_error_decomposition_pos4_trajectory_umap.pdf"
DEFAULT_N_COMPONENTS = 2
DEFAULT_N_NEIGHBORS = 300
DEFAULT_MIN_DIST = 0.3
DEFAULT_METRIC = "cosine"
DEFAULT_SEED = 42
DEFAULT_USE_SEED = False
DEFAULT_N_JOBS = 8
DEFAULT_SAMPLING_MODE = "adaptive_per_group"
DEFAULT_KEEP_ALL_THRESHOLD = 100
DEFAULT_ERROR_SAMPLE_DIVISOR = 2
DEFAULT_CORRECT_SAMPLE_DIVISOR = 4
DEFAULT_MAX_POINTS_PER_GROUP = 200

UMAP_SCRIPT_CANDIDATES = [
    Path(r"Y:\vertical-flow\umap_plot_script.py"),
    Path("/home/wenliuyuan/vertical-flow/umap_plot_script.py"),
]
FONT_PATH_CANDIDATES = [
    REPO_ROOT / "times.ttf",
    Path("/home/wenliuyuan/vertical-flow/times.ttf"),
]

FIGURE_SIZE_2D = (6, 4)
MARKER_FONTSIZE = 5
ANCHOR_FONTSIZE = 11

CSV_BUCKET_ORDER = [
    "both_correct",
    "raw_correct_carry_wrong",
    "raw_wrong_carry_correct",
    "both_wrong",
]

WRONG_RAW_CORRECT_BUCKETS = {"both_correct", "raw_correct_carry_wrong"}
WRONG_RAW_WRONG_BUCKETS = {"raw_wrong_carry_correct", "both_wrong"}

PLOT_GROUP_ORDER = ["correct", "wrong_raw_correct", "wrong_raw_wrong"]

PLOT_GROUP_LABELS = {
    "correct": r"Correct $\mathcal{T}_{\mathrm{gt}}$",
    "wrong_raw_correct": r"Wrong: raw correct $\mathcal{T}_{\mathrm{gt}}$",
    "wrong_raw_wrong": r"Wrong: raw wrong $\mathcal{T}_{\mathrm{p}}(\mathcal{T}_{\mathrm{gt}})$",
}

PLOT_GROUP_COLORS = {
    "correct": "#1f77b4",
    "wrong_raw_correct": "green",
    "wrong_raw_wrong": "#D32F2F",
}

PLOT_GROUP_ALPHAS = {
    "correct": 0.50,
    "wrong_raw_correct": 0.36,
    "wrong_raw_wrong": 0.95,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a trajectory-labeled UMAP with three raw-trajectory groups and anchors.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the error decomposition CSV.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Path to the error decomposition JSON summary.")
    parser.add_argument("--h5", type=Path, default=None, help="Optional H5 path override. Defaults to target_h5 in the JSON.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional dataset pickle fallback when H5 lacks usable samples metadata.")
    parser.add_argument("--model", type=str, default=None, help="Model path for extracting digit anchors. Defaults to umap_plot_script.py MODEL_PATH.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    parser.add_argument("--layer", type=int, default=None, help="Layer index for extracting flow vectors. Defaults to JSON layer.")
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST, help="UMAP min_dist.")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC, help="UMAP metric.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used when --use-seed is set.")
    parser.add_argument(
        "--use-seed",
        action="store_true",
        default=DEFAULT_USE_SEED,
        help="Enable deterministic UMAP with --seed. If not set, random_state=None.",
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS, help="UMAP n_jobs when --use-seed is not set.")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["adaptive_per_group", "balanced_per_group", "none"],
        default=DEFAULT_SAMPLING_MODE,
        help="Sampling mode applied after bucket merge and before UMAP.",
    )
    parser.add_argument(
        "--max-points-per-group",
        type=int,
        default=DEFAULT_MAX_POINTS_PER_GROUP,
        help="Maximum sampled points per plot group when --sampling-mode=balanced_per_group.",
    )
    parser.add_argument(
        "--keep-all-threshold",
        type=int,
        default=DEFAULT_KEEP_ALL_THRESHOLD,
        help="Keep all rows in a plot group when its size is at or below this threshold for adaptive sampling.",
    )
    parser.add_argument(
        "--error-sample-divisor",
        type=int,
        default=DEFAULT_ERROR_SAMPLE_DIVISOR,
        help="For adaptive sampling, error groups larger than --keep-all-threshold are downsampled to size // divisor.",
    )
    parser.add_argument(
        "--correct-sample-divisor",
        type=int,
        default=DEFAULT_CORRECT_SAMPLE_DIVISOR,
        help="For adaptive sampling, the correct group larger than --keep-all-threshold is downsampled to size // divisor.",
    )
    parser.add_argument(
        "--disable-first-error",
        action="store_true",
        help="Disable the first-error filter. By default it matches error_decomposition.py behavior.",
    )
    return parser


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_to_str_list(values) -> List[str]:
    if values is None:
        return []
    return [decode_text(item) for item in values]


def parse_digit(token: str) -> int:
    token = str(token).strip()
    return int(token) if len(token) == 1 and token.isdigit() else -1


def compute_raw_sum(question: str, sum_pos: int) -> int:
    try:
        operands = []
        for part in question.split("+"):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))

        if not operands:
            return -1

        result = sum(operands)
        result_len = len(str(result))
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


def compute_trajectory_digit(question: str, target_pos: int) -> int:
    raw_sum = compute_raw_sum(question, target_pos)
    if raw_sum < 0:
        return -1
    return int(raw_sum % 10)


def configure_matplotlib() -> None:
    serif_fonts = ["Times New Roman"]
    for font_path in FONT_PATH_CANDIDATES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            serif_fonts = [font_name, *serif_fonts]
            break
    serif_fonts.append("DejaVu Serif")

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": serif_fonts,
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.top": False,
            "ytick.right": False,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "legend.loc": "upper right",
            "legend.borderpad": 0.2,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "mathtext.fontset": "stix",
        }
    )


def resolve_repo_path(path_value: Path | str | None) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_metadata(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_h5_path(cli_h5: Optional[Path], metadata: Dict) -> Path:
    if cli_h5 is not None:
        return resolve_repo_path(cli_h5)

    target_h5 = metadata.get("target_h5")
    if not target_h5:
        raise ValueError("JSON summary is missing target_h5, and --h5 was not provided.")
    return resolve_repo_path(target_h5)


def infer_dataset_path(metadata: Dict, h5_path: Path) -> Optional[Path]:
    if metadata.get("sample_reference_source") != "dataset_pickle":
        return None

    search_texts = [str(metadata.get("target_h5", "")), h5_path.name, h5_path.stem]
    for text in search_texts:
        match = re.search(r"(num\d+len\d+)", text)
        if not match:
            continue
        prefix = match.group(1)
        candidates = sorted(REPO_ROOT.glob(f"{prefix}-*.pkl"))
        if candidates:
            return candidates[0]
    return None


def resolve_dataset_path(cli_dataset: Optional[Path], metadata: Dict, h5_path: Path) -> Optional[Path]:
    if cli_dataset is not None:
        return resolve_repo_path(cli_dataset)
    return infer_dataset_path(metadata, h5_path)


def resolve_default_model_path() -> Optional[str]:
    pattern = re.compile(r'^\s*MODEL_PATH\s*=\s*["\']([^"\']+)["\']')
    for script_path in UMAP_SCRIPT_CANDIDATES:
        if not script_path.exists():
            continue
        for line in script_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = pattern.match(stripped)
            if match:
                return match.group(1)
    return None


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"sample_id", "bucket", "raw_gt_mod10", "raw_hat_mod10"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")
    if df["sample_id"].duplicated().any():
        duplicated = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"CSV contains duplicated sample_id values: {duplicated[:10]}")
    invalid_buckets = sorted(set(df["bucket"]) - set(CSV_BUCKET_ORDER))
    if invalid_buckets:
        raise ValueError(f"CSV contains unexpected bucket values: {invalid_buckets}")
    return df


def load_dataset(dataset_path: Path) -> List[List[int]]:
    with dataset_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"Dataset pickle has unexpected type: {type(data).__name__}")
    return data


def load_sample_reference(h5_path: Path, dataset_path: Optional[Path]) -> Tuple[Dict[int, Dict[str, str]], str]:
    with h5py.File(h5_path, "r") as h5f:
        if "samples" in h5f:
            samples_group = h5f["samples"]
            required = {"question", "gt", "sample_idx"}
            if required.issubset(samples_group.keys()):
                questions = decode_to_str_list(samples_group["question"][:])
                gts = decode_to_str_list(samples_group["gt"][:])
                sample_ids = np.asarray(samples_group["sample_idx"][:], dtype=np.int64)
                meta = {
                    int(sample_id): {"question": question, "gt": gt}
                    for sample_id, question, gt in zip(sample_ids.tolist(), questions, gts)
                }
                return meta, "h5_samples"

    if dataset_path is None:
        raise RuntimeError("H5 does not contain usable samples metadata, and no --dataset fallback was provided.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset fallback not found: {dataset_path}")

    dataset = load_dataset(dataset_path)
    meta = {}
    for sample_id, operands in enumerate(dataset):
        question = " + ".join(str(x) for x in operands)
        meta[sample_id] = {"question": question, "gt": str(sum(operands))}
    return meta, "dataset_pickle"


def load_position_sample_ids(pos_group: h5py.Group, row_count: int) -> np.ndarray:
    sample_ds = pos_group.get("sample_indices")
    if sample_ds is None:
        sample_ds = pos_group.get("sample_ids")
    if sample_ds is None:
        return np.arange(row_count, dtype=np.int64)
    return np.asarray(sample_ds[:], dtype=np.int64)


def collect_first_error_positions(
    h5_path: Path,
    sample_meta: Dict[int, Dict[str, str]],
) -> Dict[int, int]:
    first_error_pos: Dict[int, int] = {}
    with h5py.File(h5_path, "r") as h5f:
        all_results = h5f.get("all_token_results")
        if all_results is None:
            raise RuntimeError("H5 is missing all_token_results.")

        numeric_positions: List[Tuple[int, str]] = []
        for pos_name in all_results.keys():
            if not pos_name.startswith("pos_"):
                continue
            suffix = pos_name[4:]
            if suffix.lstrip("-").isdigit():
                numeric_positions.append((int(suffix), pos_name))

        for pos, pos_name in sorted(numeric_positions, key=lambda item: item[0]):
            pos_group = all_results[pos_name]
            preds_ds = pos_group.get("preds")
            gt_chars_ds = pos_group.get("gt_chars")
            if preds_ds is None or gt_chars_ds is None:
                continue

            pred_tokens = decode_to_str_list(preds_ds[:])
            gt_chars_h5 = decode_to_str_list(gt_chars_ds[:])
            sample_ids = load_position_sample_ids(pos_group, len(pred_tokens))
            max_rows = min(len(pred_tokens), len(gt_chars_h5), len(sample_ids))

            for row_idx in range(max_rows):
                sample_id = int(sample_ids[row_idx])
                meta = sample_meta.get(sample_id)
                if meta is None:
                    continue

                gt = meta["gt"]
                if pos >= len(gt):
                    continue

                expected_gt_char = gt[pos]
                if gt_chars_h5[row_idx] != expected_gt_char:
                    continue

                pred_digit = parse_digit(pred_tokens[row_idx])
                gt_digit = parse_digit(expected_gt_char)
                if pred_digit == gt_digit:
                    continue

                previous_pos = first_error_pos.get(sample_id)
                if previous_pos is None or pos < previous_pos:
                    first_error_pos[sample_id] = pos

    return first_error_pos


def pick_target_result_len(result_lens: np.ndarray) -> int:
    values, counts = np.unique(result_lens.astype(np.int64), return_counts=True)
    return int(values[np.argmax(counts)])


def load_position_rows(
    h5_path: Path,
    target_pos: int,
    sample_meta: Dict[int, Dict[str, str]],
) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
    pos_key = f"pos_{target_pos}"
    with h5py.File(h5_path, "r") as h5f:
        all_results = h5f["all_token_results"]
        if pos_key not in all_results:
            raise KeyError(f"H5 missing group all_token_results/{pos_key}")

        pos_group = all_results[pos_key]
        flows_ds = pos_group["flows"]
        flows_shape = tuple(flows_ds.shape)
        labels = np.asarray(pos_group["labels"][:], dtype=np.bool_)
        pred_tokens = decode_to_str_list(pos_group["preds"][:])
        gt_chars_h5 = decode_to_str_list(pos_group["gt_chars"][:])
        sample_ids = load_position_sample_ids(pos_group, len(labels))

    max_rows = min(len(labels), len(pred_tokens), len(gt_chars_h5), len(sample_ids))
    rows = []
    for row_idx in range(max_rows):
        sample_id = int(sample_ids[row_idx])
        sample_record = sample_meta.get(sample_id)
        if sample_record is None:
            continue

        gt = sample_record["gt"]
        if target_pos >= len(gt):
            continue

        expected_gt_char = gt[target_pos]
        pred_digit = parse_digit(pred_tokens[row_idx])
        gt_digit = parse_digit(expected_gt_char)
        is_correct = bool(labels[row_idx])
        if pred_digit >= 0 and gt_digit >= 0 and is_correct != (pred_digit == gt_digit):
            raise ValueError(
                f"H5 correctness label mismatch at sample_id={sample_id}: "
                f"pred={pred_digit}, gt={gt_digit}, label={is_correct}"
            )

        rows.append(
            {
                "row_index": row_idx,
                "sample_id": sample_id,
                "question": sample_record["question"],
                "gt": gt,
                "result_len": len(gt),
                "gt_digit": gt_digit,
                "pred_digit": pred_digit,
                "pred_token": pred_tokens[row_idx],
                "gt_char_h5": gt_chars_h5[row_idx],
                "gt_char_match": gt_chars_h5[row_idx] == expected_gt_char,
                "is_correct": is_correct,
            }
        )

    if not rows:
        raise RuntimeError("No valid rows were loaded from the target position.")

    return pd.DataFrame(rows), flows_shape


def filter_rows_like_error_decomposition(
    position_df: pd.DataFrame,
    h5_path: Path,
    target_pos: int,
    sample_meta: Dict[int, Dict[str, str]],
    disable_first_error: bool,
) -> Tuple[pd.DataFrame, int]:
    target_result_len = pick_target_result_len(position_df["result_len"].to_numpy(dtype=np.int64))
    filtered = position_df.loc[position_df["result_len"] == target_result_len].reset_index(drop=True)

    if filtered.empty:
        raise RuntimeError("No rows remain after target result-length filtering.")
    if (~filtered["gt_char_match"]).any():
        bad = filtered.loc[~filtered["gt_char_match"], "sample_id"].tolist()[:10]
        raise ValueError(f"Filtered rows contain gt_char mismatches: {bad}")

    if disable_first_error:
        return filtered, target_result_len

    first_error_pos = collect_first_error_positions(h5_path, sample_meta)
    keep_mask = np.ones(len(filtered), dtype=np.bool_)
    wrong_mask = ~filtered["is_correct"].to_numpy(dtype=np.bool_)
    wrong_sample_ids = filtered.loc[wrong_mask, "sample_id"].astype(np.int64).to_numpy()
    keep_mask[wrong_mask] = np.asarray(
        [first_error_pos.get(int(sample_id)) == target_pos for sample_id in wrong_sample_ids],
        dtype=np.bool_,
    )

    filtered = filtered.loc[keep_mask].reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError("No rows remain after first-error filtering.")
    return filtered, target_result_len


def validate_against_summary(filtered_df: pd.DataFrame, metadata: Dict) -> None:
    splits_all = metadata.get("splits", {}).get("all")
    if not splits_all:
        return

    expected_rows = splits_all.get("rows")
    dist = splits_all.get("label_correct_distribution", {})
    expected_correct = int(dist.get("1", 0))
    expected_incorrect = int(dist.get("0", 0))

    actual_rows = int(len(filtered_df))
    actual_correct = int(filtered_df["is_correct"].sum())
    actual_incorrect = int((~filtered_df["is_correct"]).sum())

    if expected_rows is not None and actual_rows != int(expected_rows):
        raise ValueError(f"Filtered row count mismatch: actual={actual_rows}, expected={expected_rows}")
    if dist and (actual_correct != expected_correct or actual_incorrect != expected_incorrect):
        raise ValueError(
            "Filtered correctness counts mismatch: "
            f"actual(correct={actual_correct}, incorrect={actual_incorrect}) vs "
            f"expected(correct={expected_correct}, incorrect={expected_incorrect})"
        )


def attach_trajectory_groups(filtered_df: pd.DataFrame, csv_df: pd.DataFrame, metadata: Dict, target_pos: int) -> pd.DataFrame:
    retained_incorrect_ids = set(filtered_df.loc[~filtered_df["is_correct"], "sample_id"].astype(int).tolist())
    csv_ids = set(csv_df["sample_id"].astype(int).tolist())

    missing_in_csv = sorted(retained_incorrect_ids - csv_ids)
    extra_in_csv = sorted(csv_ids - retained_incorrect_ids)
    if missing_in_csv:
        raise ValueError(f"Retained incorrect samples missing bucket labels in CSV: {missing_in_csv[:10]}")
    if extra_in_csv:
        raise ValueError(f"CSV contains sample_id values not present in filtered incorrect pool: {extra_in_csv[:10]}")

    merged = filtered_df.merge(csv_df[["sample_id", "bucket", "raw_gt_mod10", "raw_hat_mod10"]], on="sample_id", how="left")
    if merged.loc[merged["is_correct"], "bucket"].notna().any():
        bad = merged.loc[merged["is_correct"] & merged["bucket"].notna(), "sample_id"].tolist()[:10]
        raise ValueError(f"Correct samples unexpectedly received bucket labels: {bad}")
    if merged.loc[~merged["is_correct"], "bucket"].isna().any():
        bad = merged.loc[(~merged["is_correct"]) & merged["bucket"].isna(), "sample_id"].tolist()[:10]
        raise ValueError(f"Incorrect samples are missing bucket labels after merge: {bad}")

    merged["gt_trajectory_digit"] = merged["question"].apply(lambda question: compute_trajectory_digit(str(question), target_pos))
    if (merged["gt_trajectory_digit"] < 0).any():
        bad = merged.loc[merged["gt_trajectory_digit"] < 0, "sample_id"].tolist()[:10]
        raise ValueError(f"Failed to compute trajectory digits for sample_id values: {bad}")

    expected_bucket_counts = metadata.get("error_decomposition", {}).get("bucket_counts", {})
    if expected_bucket_counts:
        actual_bucket_counts = merged.loc[~merged["is_correct"], "bucket"].value_counts().to_dict()
        for bucket in CSV_BUCKET_ORDER:
            actual = int(actual_bucket_counts.get(bucket, 0))
            expected = int(expected_bucket_counts.get(bucket, 0))
            if actual != expected:
                raise ValueError(f"Bucket count mismatch for {bucket}: actual={actual}, expected={expected}")

    incorrect_mask = ~merged["is_correct"]
    if not np.array_equal(
        merged.loc[incorrect_mask, "gt_trajectory_digit"].to_numpy(dtype=np.int64),
        merged.loc[incorrect_mask, "raw_gt_mod10"].to_numpy(dtype=np.int64),
    ):
        bad = merged.loc[
            incorrect_mask & (merged["gt_trajectory_digit"].astype(np.int64) != merged["raw_gt_mod10"].astype(np.int64)),
            ["sample_id", "gt_trajectory_digit", "raw_gt_mod10"],
        ].head(10)
        raise ValueError(f"Incorrect samples have gt trajectory mismatches with CSV:\n{bad.to_string(index=False)}")

    merged["probe_trajectory_digit"] = merged["gt_trajectory_digit"].astype(np.int64)
    merged.loc[incorrect_mask, "probe_trajectory_digit"] = merged.loc[incorrect_mask, "raw_hat_mod10"].astype(np.int64)
    if ((merged["probe_trajectory_digit"] < 0) | (merged["probe_trajectory_digit"] > 9)).any():
        bad = merged.loc[
            (merged["probe_trajectory_digit"] < 0) | (merged["probe_trajectory_digit"] > 9),
            ["sample_id", "probe_trajectory_digit"],
        ].head(10)
        raise ValueError(f"Probe trajectory digits are outside [0, 9]:\n{bad.to_string(index=False)}")

    merged["plot_group"] = "wrong_raw_wrong"
    merged.loc[merged["is_correct"], "plot_group"] = "correct"
    merged.loc[incorrect_mask & merged["bucket"].isin(sorted(WRONG_RAW_CORRECT_BUCKETS)), "plot_group"] = "wrong_raw_correct"
    merged.loc[incorrect_mask & merged["bucket"].isin(sorted(WRONG_RAW_WRONG_BUCKETS)), "plot_group"] = "wrong_raw_wrong"

    merged["trajectory_text"] = merged["gt_trajectory_digit"].astype(int).astype(str)
    wrong_raw_wrong_mask = merged["plot_group"] == "wrong_raw_wrong"
    merged.loc[wrong_raw_wrong_mask, "trajectory_text"] = (
        merged.loc[wrong_raw_wrong_mask, "probe_trajectory_digit"].astype(int).astype(str)
        + "("
        + merged.loc[wrong_raw_wrong_mask, "gt_trajectory_digit"].astype(int).astype(str)
        + ")"
    )

    return merged


def sample_plot_groups(
    plot_df: pd.DataFrame,
    sampling_mode: str,
    max_points_per_group: int,
    keep_all_threshold: int,
    error_sample_divisor: int,
    correct_sample_divisor: int,
    seed: int,
) -> pd.DataFrame:
    if sampling_mode == "none":
        return plot_df.reset_index(drop=True)
    if sampling_mode not in {"adaptive_per_group", "balanced_per_group"}:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    group_values = plot_df["plot_group"].to_numpy()
    rng = np.random.default_rng(seed)
    selected_positions: List[np.ndarray] = []

    for group in PLOT_GROUP_ORDER:
        group_positions = np.flatnonzero(group_values == group)
        if len(group_positions) == 0:
            continue

        target_size = len(group_positions)
        if sampling_mode == "balanced_per_group":
            if max_points_per_group <= 0:
                raise ValueError(f"--max-points-per-group must be positive, got {max_points_per_group}")
            if len(group_positions) > max_points_per_group:
                target_size = max_points_per_group
        else:
            if keep_all_threshold < 0:
                raise ValueError(f"--keep-all-threshold must be non-negative, got {keep_all_threshold}")
            if error_sample_divisor <= 0:
                raise ValueError(f"--error-sample-divisor must be positive, got {error_sample_divisor}")
            if correct_sample_divisor <= 0:
                raise ValueError(f"--correct-sample-divisor must be positive, got {correct_sample_divisor}")

            if len(group_positions) > keep_all_threshold:
                divisor = correct_sample_divisor if group == "correct" else error_sample_divisor
                target_size = max(1, len(group_positions) // divisor)

        if target_size < len(group_positions):
            group_positions = np.sort(rng.choice(group_positions, size=target_size, replace=False))
        selected_positions.append(group_positions)

    if not selected_positions:
        raise RuntimeError("No rows remain after sampling.")

    sampled_positions = np.sort(np.concatenate(selected_positions))
    return plot_df.iloc[sampled_positions].reset_index(drop=True)


def read_layer_features(
    h5_path: Path,
    target_pos: int,
    row_indices: np.ndarray,
    layer: int,
) -> np.ndarray:
    pos_key = f"pos_{target_pos}"
    row_indices = np.asarray(row_indices, dtype=np.int64)
    order = np.argsort(row_indices)
    sorted_indices = row_indices[order]

    with h5py.File(h5_path, "r") as h5f:
        flows_ds = h5f["all_token_results"][pos_key]["flows"]
        if flows_ds.ndim != 3:
            raise ValueError(f"Expected flows to have shape (N, num_layers, hidden_dim), got {flows_ds.shape}")
        if not (0 <= layer < flows_ds.shape[1]):
            raise ValueError(f"Layer {layer} is out of range for flows with {flows_ds.shape[1]} layers")
        try:
            sorted_features = np.asarray(flows_ds[sorted_indices, layer, :], dtype=np.float32)
        except TypeError:
            sorted_features = np.asarray(flows_ds[:, layer, :], dtype=np.float32)[sorted_indices]

    inverse_order = np.argsort(order)
    feature_matrix = sorted_features[inverse_order]
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature matrix contains NaN or Inf values.")
    return feature_matrix


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return values / norms


def extract_digit_anchors(model_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("transformers is required to extract anchors.") from exc

    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
    )
    model.eval()

    try:
        if not hasattr(model, "lm_head"):
            raise RuntimeError("Loaded model does not expose lm_head for anchor extraction.")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        target_ids = []
        valid_digits = []
        for digit in range(10):
            token_ids = tokenizer.encode(str(digit), add_special_tokens=False)
            if len(token_ids) >= 1:
                target_ids.append(token_ids[0])
                valid_digits.append(digit)

        if not target_ids:
            raise RuntimeError("No valid single-token digits were found for anchor extraction.")

        anchor_weights = model.lm_head.weight[target_ids].detach().float().cpu().numpy()
        anchor_labels = np.asarray(valid_digits, dtype=np.int64)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return anchor_weights.astype(np.float32), anchor_labels


def prepare_umap_inputs(
    feature_matrix: np.ndarray,
    anchor_data: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[int]]:
    if anchor_data is None:
        return np.asarray(feature_matrix, dtype=np.float32), None

    centered_hiddens = np.asarray(feature_matrix, dtype=np.float32) - np.mean(feature_matrix, axis=0, keepdims=True)
    centered_anchors = np.asarray(anchor_data, dtype=np.float32) - np.mean(anchor_data, axis=0, keepdims=True)
    centered_hiddens = l2_normalize_rows(centered_hiddens)
    centered_anchors = l2_normalize_rows(centered_anchors)
    combined = np.vstack([centered_hiddens, centered_anchors]).astype(np.float32)
    return combined, int(feature_matrix.shape[0])


def compute_umap_embedding(
    feature_matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
    use_seed: bool,
    n_jobs: int,
) -> np.ndarray:
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "umap-learn is required to compute the embedding. Please run this script in an environment with umap-learn installed."
        ) from exc

    random_state = seed if use_seed else None
    effective_n_jobs = 1 if random_state is not None else n_jobs

    reducer = umap.UMAP(
        n_components=DEFAULT_N_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=effective_n_jobs,
    )
    embedding = reducer.fit_transform(feature_matrix)
    if embedding.shape != (feature_matrix.shape[0], DEFAULT_N_COMPONENTS):
        raise ValueError(f"Unexpected UMAP embedding shape: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("UMAP embedding contains NaN or Inf values.")
    return embedding


def choose_legend_anchor(embedding: np.ndarray, legend_items: int) -> Tuple[float, float]:
    if embedding.size == 0:
        return (0.78, 0.16)

    x_vals = embedding[:, 0]
    y_vals = embedding[:, 1]
    x_span = float(np.max(x_vals) - np.min(x_vals))
    y_span = float(np.max(y_vals) - np.min(y_vals))
    if x_span <= 0 or y_span <= 0:
        return (0.78, 0.16)

    x_norm = (x_vals - np.min(x_vals)) / x_span
    y_norm = (y_vals - np.min(y_vals)) / y_span

    candidates = [
        (0.80, 0.16),
        (0.82, 0.82),
        (0.55, 0.74),
        (0.24, 0.18),
        (0.80, 0.48),
        (0.54, 0.18),
        (0.30, 0.72),
    ]

    box_w = 0.42
    box_h = 0.08 + 0.03 * legend_items

    best = candidates[0]
    best_score = float("inf")
    sample_n = float(len(x_norm))

    for cx, cy in candidates:
        x0 = max(0.0, cx - box_w / 2.0)
        x1 = min(1.0, cx + box_w / 2.0)
        y0 = max(0.0, cy - box_h / 2.0)
        y1 = min(1.0, cy + box_h / 2.0)

        local_count = np.sum((x_norm >= x0) & (x_norm <= x1) & (y_norm >= y0) & (y_norm <= y1))
        score = float(local_count)

        if cy <= 0.22:
            score -= max(1.0, 0.002 * sample_n)
        elif cx >= 0.72 and cy >= 0.72:
            score -= max(0.5, 0.001 * sample_n)

        if score < best_score:
            best_score = score
            best = (cx, cy)

    return best


def build_legend_elements(plot_groups_present: List[str], include_anchor: bool) -> List[Line2D]:
    elements: List[Line2D] = []
    if include_anchor:
        elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Anchor",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
            )
        )

    for group in PLOT_GROUP_ORDER:
        if group not in plot_groups_present:
            continue
        elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=PLOT_GROUP_LABELS[group],
                markerfacecolor=PLOT_GROUP_COLORS[group],
                markeredgecolor=PLOT_GROUP_COLORS[group],
                markersize=6,
                alpha=PLOT_GROUP_ALPHAS[group],
            )
        )
    return elements


def plot_trajectory_umap(
    plot_df: pd.DataFrame,
    sample_embedding: np.ndarray,
    output_path: Path,
    target_pos: int,
    anchor_embedding: Optional[np.ndarray],
    anchor_labels: Optional[np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)
    fig.subplots_adjust(top=0.94, bottom=0.16)
    ax.scatter(sample_embedding[:, 0], sample_embedding[:, 1], s=0, alpha=0)

    for group in PLOT_GROUP_ORDER:
        mask = plot_df["plot_group"].to_numpy() == group
        if not np.any(mask):
            continue

        group_embedding = sample_embedding[mask]
        group_labels = plot_df.loc[mask, "trajectory_text"].astype(str).to_numpy()
        color = PLOT_GROUP_COLORS[group]
        alpha = PLOT_GROUP_ALPHAS[group]

        for (x_coord, y_coord), text in zip(group_embedding, group_labels):
            ax.text(
                x_coord,
                y_coord,
                text,
                color=color,
                fontsize=MARKER_FONTSIZE,
                alpha=alpha,
                ha="center",
                va="center",
                fontweight="bold",
                clip_on=True,
            )

    include_anchor = anchor_embedding is not None and anchor_labels is not None
    if include_anchor:
        for idx, label in enumerate(anchor_labels.tolist()):
            ax.text(
                anchor_embedding[idx, 0],
                anchor_embedding[idx, 1],
                str(label),
                color="black",
                fontsize=ANCHOR_FONTSIZE,
                fontweight="bold",
                ha="center",
                va="center",
                clip_on=True,
            )

    present_groups = [group for group in PLOT_GROUP_ORDER if (plot_df["plot_group"] == group).any()]

    ax.legend(
        handles=build_legend_elements(present_groups, include_anchor=include_anchor),
        loc="upper left",
        bbox_to_anchor=(0.015, 0.82),
        ncol=1,
        labelspacing=0.15,
        handletextpad=0.25,
        borderaxespad=0.0,
        frameon=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.015,
        0.98,
        "10-digit Addition (3 terms)\n"
        f"position $p = {target_pos}$",
        fontsize=12,
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox={
            "boxstyle": "square,pad=0.2",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.6,
            "linewidth": 1.0,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path = resolve_repo_path(args.csv)
    json_path = resolve_repo_path(args.json)
    output_path = resolve_repo_path(args.out)

    metadata = load_metadata(json_path)
    h5_path = resolve_h5_path(args.h5, metadata)
    dataset_path = resolve_dataset_path(args.dataset, metadata, h5_path)
    model_path = args.model or resolve_default_model_path()
    if model_path is None:
        raise RuntimeError("Could not determine a default model path for anchors. Please pass --model.")

    target_pos = int(metadata.get("target_pos", 4))
    layer = int(metadata.get("layer", 36) if args.layer is None else args.layer)

    configure_matplotlib()

    csv_df = load_csv(csv_path)
    sample_meta, sample_meta_source = load_sample_reference(h5_path, dataset_path)
    position_df, flows_shape = load_position_rows(h5_path, target_pos, sample_meta)
    filtered_df, target_result_len = filter_rows_like_error_decomposition(
        position_df=position_df,
        h5_path=h5_path,
        target_pos=target_pos,
        sample_meta=sample_meta,
        disable_first_error=args.disable_first_error,
    )
    if not args.disable_first_error:
        validate_against_summary(filtered_df, metadata)

    plot_df = attach_trajectory_groups(
        filtered_df=filtered_df,
        csv_df=csv_df,
        metadata=metadata,
        target_pos=target_pos,
    )
    sampled_plot_df = sample_plot_groups(
        plot_df=plot_df,
        sampling_mode=args.sampling_mode,
        max_points_per_group=args.max_points_per_group,
        keep_all_threshold=args.keep_all_threshold,
        error_sample_divisor=args.error_sample_divisor,
        correct_sample_divisor=args.correct_sample_divisor,
        seed=args.seed,
    )
    row_indices = sampled_plot_df["row_index"].to_numpy(dtype=np.int64)

    if len(flows_shape) != 3:
        raise ValueError(f"Expected H5 flows to have 3 dimensions, got {flows_shape}")
    if not (0 <= layer < flows_shape[1]):
        raise ValueError(f"Layer {layer} is out of range for flows with {flows_shape[1]} layers")

    feature_matrix = read_layer_features(
        h5_path=h5_path,
        target_pos=target_pos,
        row_indices=row_indices,
        layer=layer,
    )

    anchor_data, anchor_labels = extract_digit_anchors(model_path)
    umap_input, sample_count = prepare_umap_inputs(feature_matrix, anchor_data)
    embedding = compute_umap_embedding(
        feature_matrix=umap_input,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
        use_seed=args.use_seed,
        n_jobs=args.n_jobs,
    )

    if sample_count is None:
        sample_embedding = embedding
        anchor_embedding = None
    else:
        sample_embedding = embedding[:sample_count]
        anchor_embedding = embedding[sample_count:]

    print(f"Resolved H5 path: {h5_path}")
    print(f"Sample metadata source: {sample_meta_source}")
    if dataset_path is not None:
        print(f"Dataset fallback: {dataset_path}")
    print(f"Using model for anchors: {model_path}")
    print(f"Using target position {target_pos} and layer {layer}")
    print(f"H5 pos_{target_pos} flows shape: {flows_shape}")
    print(f"Target result length: {target_result_len}")
    print(f"First-error enabled: {not args.disable_first_error}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(
        f"Filtered correctness counts: correct={int(filtered_df['is_correct'].sum())}, "
        f"incorrect={int((~filtered_df['is_correct']).sum())}"
    )
    print(
        "Trajectory validation: "
        f"incorrect_csv_matches={int((~plot_df['is_correct']).sum())}, "
        f"correct_probe_equals_gt={int(plot_df.loc[plot_df['is_correct'], 'probe_trajectory_digit'].eq(plot_df.loc[plot_df['is_correct'], 'gt_trajectory_digit']).sum())}"
    )
    print(f"Anchor count: {len(anchor_labels)}")
    print("Plot group counts before sampling:")
    print(plot_df["plot_group"].value_counts().reindex(PLOT_GROUP_ORDER, fill_value=0).to_string())
    if args.sampling_mode == "adaptive_per_group":
        print(
            "Sampling: "
            f"mode={args.sampling_mode}, keep_all_threshold={args.keep_all_threshold}, "
            f"error_sample_divisor={args.error_sample_divisor}, "
            f"correct_sample_divisor={args.correct_sample_divisor}, "
            f"sampling_seed={args.seed}"
        )
    elif args.sampling_mode == "balanced_per_group":
        print(
            f"Sampling: mode={args.sampling_mode}, max_points_per_group={args.max_points_per_group}, "
            f"sampling_seed={args.seed}"
        )
    else:
        print(f"Sampling: mode={args.sampling_mode}, sampling_seed={args.seed}")
    print("Plot group counts after sampling:")
    print(sampled_plot_df["plot_group"].value_counts().reindex(PLOT_GROUP_ORDER, fill_value=0).to_string())
    print(f"Sampled rows for plotting: {len(sampled_plot_df)}")
    print(f"Aligned feature matrix shape: {feature_matrix.shape}")
    print(
        f"UMAP params: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, "
        f"metric={args.metric}, use_seed={args.use_seed}, "
        f"seed={args.seed}, n_jobs={1 if args.use_seed else args.n_jobs}"
    )
    print(f"UMAP embedding shape: {embedding.shape}")

    plot_trajectory_umap(
        plot_df=sampled_plot_df,
        sample_embedding=sample_embedding,
        output_path=output_path,
        target_pos=target_pos,
        anchor_embedding=anchor_embedding,
        anchor_labels=anchor_labels,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
