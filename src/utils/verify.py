"""Utilities for residual-stream steering directions.

This module supports the optional steering baseline used by ``dualstream.py``.
It computes digit/carry-conditioned centroids from an
HDF5 activation file and derives carry-transition directions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


def _decode_array(values) -> List[str]:
    result: List[str] = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _position_sort_key(name: str):
    if name.startswith("pos"):
        suffix = name[3:]
        if suffix.lstrip("-").isdigit():
            return (0, int(suffix))
    return (1, name)


def _read_first_available(group: h5py.Group, names: Iterable[str]):
    for name in names:
        if name in group:
            return group[name][()]
    raise KeyError(f"None of the datasets are present in {group.name}: {list(names)}")


def load_position_arrays(h5_path: str | Path):
    """Load labels, ground-truth digits, incoming carries, and last-layer flows by position.

    Returns four dictionaries keyed by integer position. Each value is a NumPy
    array aligned row-wise within the position.
    """
    labels_by_pos: Dict[int, np.ndarray] = {}
    gt_by_pos: Dict[int, np.ndarray] = {}
    true_carry_by_pos: Dict[int, np.ndarray] = {}
    last_layer_by_pos: Dict[int, np.ndarray] = {}

    with h5py.File(h5_path, "r") as h5f:
        if "all_token_results" not in h5f:
            raise RuntimeError(f"{h5_path} does not contain all_token_results")
        root = h5f["all_token_results"]
        for name in sorted(root.keys(), key=_position_sort_key):
            if not name.startswith("pos"):
                continue
            try:
                pos = int(name[3:])
            except ValueError:
                continue
            group = root[name]
            flows = _read_first_available(group, ["flows", "flows_post_ffn", "post_ffn"])
            if flows.ndim != 3:
                raise RuntimeError(f"{group.name}/flows must be 3D, got shape {flows.shape}")
            labels = np.asarray(_read_first_available(group, ["labels"])).astype(bool)
            gt_chars = _decode_array(_read_first_available(group, ["gt_chars", "gts", "gt"]))
            incoming = np.asarray(_read_first_available(group, ["incoming_carries", "true_carries", "in_carries"]))
            gt_digits = np.asarray([int(ch) if str(ch).isdigit() else -1 for ch in gt_chars], dtype=np.int64)
            valid = (gt_digits >= 0) & (incoming >= 0)
            labels_by_pos[pos] = labels[valid]
            gt_by_pos[pos] = gt_digits[valid]
            true_carry_by_pos[pos] = incoming[valid].astype(np.int64)
            last_layer_by_pos[pos] = np.asarray(flows[valid, -1, :], dtype=np.float32)
    return labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos


def collect_records(
    labels_by_pos,
    gt_by_pos,
    true_carry_by_pos,
    last_layer_by_pos,
    positions: Optional[set[int]] = None,
):
    """Collect centroid records from loaded position dictionaries."""
    records = []
    for pos, hidden in last_layer_by_pos.items():
        if positions is not None and pos not in positions:
            continue
        labels = labels_by_pos[pos]
        gt_digits = gt_by_pos[pos]
        carries = true_carry_by_pos[pos]
        for i in range(len(hidden)):
            records.append(
                {
                    "pos": int(pos),
                    "correct": bool(labels[i]),
                    "gt_digit": int(gt_digits[i]),
                    "carry": int(carries[i]),
                    "hidden": hidden[i],
                }
            )
    return records


def compute_means(records):
    """Compute mean hidden state for each (digit, carry) bucket."""
    buckets: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for record in records:
        digit = int(record["gt_digit"])
        carry = int(record["carry"])
        if 0 <= digit <= 9 and 0 <= carry <= 2:
            buckets[(digit, carry)].append(np.asarray(record["hidden"], dtype=np.float32))

    means: Dict[Tuple[int, int], np.ndarray] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for key, values in buckets.items():
        stacked = np.stack(values, axis=0)
        means[key] = stacked.mean(axis=0)
        counts[key] = int(stacked.shape[0])
    return means, counts


def build_dirs_cross_digit(means):
    """Build carry-transition directions used by vector steering.

    ``dir01[d]`` maps carry 0 at digit d toward carry 1 at digit d+1, and
    ``dir12[d]`` maps carry 1 at digit d toward carry 2 at digit d+1 when both
    centroids are available. Missing transitions are stored as ``None``.
    """
    dir01 = {}
    dir12 = {}
    for digit in range(10):
        src01 = means.get((digit, 0))
        dst01 = means.get(((digit + 1) % 10, 1))
        src12 = means.get((digit, 1))
        dst12 = means.get(((digit + 1) % 10, 2))
        dir01[digit] = None if src01 is None or dst01 is None else (dst01 - src01)
        dir12[digit] = None if src12 is None or dst12 is None else (dst12 - src12)
    return dir01, dir12
