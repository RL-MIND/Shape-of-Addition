"""Shared evaluation and summary metrics."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

OFF_BY_ONE_ROWS = ("minus_one", "plus_one")
OFF_BY_ONE_COLS = ("fixed_to_gt", "still_off_by_one", "other_wrong")
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def mask_first_error_positions(
    pred_digits: np.ndarray,
    gt_digits: np.ndarray,
    sample_ids: np.ndarray,
    pos_ids: np.ndarray,
) -> np.ndarray:
    """Keep only the first incorrect position for each sample."""
    keep = np.ones_like(pred_digits, dtype=bool)
    incorrect = pred_digits != gt_digits
    if not np.any(incorrect):
        return keep
    for sample_id in np.unique(sample_ids[incorrect]):
        sample_mask = (sample_ids == sample_id) & incorrect
        if np.sum(sample_mask) <= 1:
            continue
        min_pos = pos_ids[sample_mask].min()
        keep[sample_mask & (pos_ids != min_pos)] = False
    return keep


def get_off_by_one_direction(pred_digit: int, gt_digit: int) -> Optional[str]:
    """Return whether pred_digit is one below or above gt_digit modulo 10."""
    if not (0 <= pred_digit <= 9 and 0 <= gt_digit <= 9):
        return None
    diff = (pred_digit - gt_digit) % 10
    if diff == 9:
        return "minus_one"
    if diff == 1:
        return "plus_one"
    return None


def make_empty_off_by_one_confusion() -> Dict[str, Dict[str, int]]:
    """Create an empty off-by-one confusion table."""
    return {row: {col: 0 for col in OFF_BY_ONE_COLS} for row in OFF_BY_ONE_ROWS}


def analyze_off_by_one_errors(
    pred_digits: np.ndarray,
    gt_digits: np.ndarray,
    corrected_digits: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Summarize off-by-one and other digit errors."""
    pred_arr = np.asarray(pred_digits, dtype=np.int64)
    gt_arr = np.asarray(gt_digits, dtype=np.int64)
    corrected_arr = None if corrected_digits is None else np.asarray(corrected_digits, dtype=np.int64)

    orig_error_mask = pred_arr != gt_arr
    orig_error_count = int(np.sum(orig_error_mask))
    confusion = make_empty_off_by_one_confusion()
    off_by_one_count = 0

    for idx in np.flatnonzero(orig_error_mask):
        direction = get_off_by_one_direction(int(pred_arr[idx]), int(gt_arr[idx]))
        if direction is None:
            continue
        off_by_one_count += 1
        if corrected_arr is None:
            continue
        corrected_digit = int(corrected_arr[idx])
        if corrected_digit == int(gt_arr[idx]):
            outcome = "fixed_to_gt"
        elif get_off_by_one_direction(corrected_digit, int(gt_arr[idx])) is not None:
            outcome = "still_off_by_one"
        else:
            outcome = "other_wrong"
        confusion[direction][outcome] += 1

    other_error_count = int(orig_error_count - off_by_one_count)
    denom = orig_error_count if orig_error_count > 0 else 1
    return {
        "orig_error_count": orig_error_count,
        "off_by_one_count": int(off_by_one_count),
        "other_error_count": int(other_error_count),
        "off_by_one_ratio": float(off_by_one_count / denom) if orig_error_count > 0 else 0.0,
        "other_error_ratio": float(other_error_count / denom) if orig_error_count > 0 else 0.0,
        "off_by_one_confusion": confusion,
    }


def compute_mean_std_ci(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute mean, sample std, and a t-based 95% confidence interval."""
    clean_values = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    if not clean_values:
        return {"count": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    arr = np.asarray(clean_values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size == 1:
        return {"count": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(arr, ddof=1))
    t_critical = T_CRITICAL_95.get(arr.size - 1, 1.96)
    margin = float(t_critical * std / math.sqrt(arr.size))
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def compute_correction_metrics(
    corrected: np.ndarray,
    pred_orig: np.ndarray,
    gt: np.ndarray,
) -> tuple[float, float, float]:
    """Compute modification rate, true-positive correction, and false-positive preservation."""
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
