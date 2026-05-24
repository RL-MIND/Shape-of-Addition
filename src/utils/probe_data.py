import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def compute_raw_sum(question: str, sum_pos: int) -> int:
    """Compute the positional digit sum without applying modulo."""
    try:
        operands = []
        for part in question.split("+"):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))

        if not operands:
            return -1

        result = sum(operands)
        result_str = str(result)
        result_len = len(result_str)
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


def compute_c_potential(question: str, current_pos: int) -> float:
    """Compute the C_potential (Potential of Truth) score."""
    try:
        operands = []
        for part in question.split("+"):
            part = part.strip()
            if part.isdigit():
                operands.append(int(part))
        if not operands:
            return 0.0

        result_len = len(str(sum(operands)))
        c_potential = 0.0
        for k in range(current_pos + 1, result_len + 5):
            raw_sum_k = compute_raw_sum(question, k)
            if raw_sum_k == -1:
                break

            exponent = k - current_pos
            term = raw_sum_k / (10 ** exponent)
            c_potential += term

            if term < 1e-9:
                break

        return c_potential
    except Exception:
        return 0.0


def load_dataset(path: Path) -> List[List[int]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [list(item) for item in data]


def _to_str_list(arr) -> List[str]:
    if arr is None:
        return []
    out: List[str] = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def parse_question_operands(question: str) -> List[int]:
    operands: List[int] = []
    for part in question.split("+"):
        part = part.strip()
        if part.isdigit():
            operands.append(int(part))
    return operands


def load_h5_sample_metadata(h5_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    questions: Dict[int, str] = {}
    gts: Dict[int, str] = {}
    with h5py.File(h5_path, "r") as hf:
        samples_group = hf.get("samples")
        if samples_group is None:
            return questions, gts
        sample_idx_ds = samples_group.get("sample_idx")
        question_ds = samples_group.get("question")
        gt_ds = samples_group.get("gt")
        if sample_idx_ds is None or question_ds is None:
            return questions, gts

        sample_ids = np.asarray(sample_idx_ds, dtype=np.int64)
        question_values = _to_str_list(question_ds)
        gt_values = _to_str_list(gt_ds) if gt_ds is not None else []
        for row, sample_id in enumerate(sample_ids.tolist()):
            if row < len(question_values):
                questions[int(sample_id)] = question_values[row]
            if row < len(gt_values):
                gts[int(sample_id)] = gt_values[row]
    return questions, gts


def load_h5_sample_dataset(h5_path: Path) -> Dict[int, List[int]]:
    questions, _ = load_h5_sample_metadata(h5_path)
    return {
        sample_id: operands
        for sample_id, question in questions.items()
        if (operands := parse_question_operands(question))
    }


def load_positions(h5_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    positions: Dict[int, Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as hf:
        sample_questions, sample_gts = load_h5_sample_metadata(h5_path)
        positions_group = hf["all_token_results"]
        for pos_name, pos_group in positions_group.items():
            if not pos_name.startswith("pos_"):
                continue
            try:
                pos_idx = int(pos_name.split("_", 1)[1])
            except Exception:
                continue

            flows_ds = pos_group.get("flows")
            if flows_ds is None:
                flows_ds = pos_group.get("flows_post_ffn")
            flows = np.asarray(flows_ds)
            labels = np.asarray(pos_group.get("labels"), dtype=np.bool_)
            preds = _to_str_list(pos_group.get("preds"))
            gt_chars = _to_str_list(pos_group.get("gt_chars"))
            true_carry = pos_group.get("true_in_carry")
            pred_carry = pos_group.get("pred_in_carry")
            sample_ids_ds = pos_group.get("sample_ids")
            if sample_ids_ds is None:
                sample_ids_ds = pos_group.get("sample_indices")
            sample_ids = np.asarray(sample_ids_ds) if sample_ids_ds is not None else None
            positions[pos_idx] = {
                "flows": flows,
                "labels": labels,
                "preds": preds,
                "gt_chars": gt_chars,
                "true_carry": np.asarray(true_carry) if true_carry is not None else None,
                "pred_carry": np.asarray(pred_carry) if pred_carry is not None else None,
                "sample_ids": sample_ids,
                "sample_questions": sample_questions,
                "sample_gts": sample_gts,
            }
    return positions


def build_flat_dataset(
    dataset: List[List[int]],
    positions: Dict[int, Dict[str, np.ndarray]],
    positions_filter: List[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align per-position flows with the underlying dataset to derive labels."""
    flows_list: List[np.ndarray] = []
    raw_labels: List[int] = []
    carry_labels: List[float] = []
    gt_digits: List[int] = []
    pred_digits: List[int] = []
    sample_ids: List[int] = []
    pos_ids: List[int] = []

    positions_allow = set(positions_filter) if positions_filter else None

    for pos_idx, data in positions.items():
        if positions_allow is not None and pos_idx not in positions_allow:
            continue
        flows = np.asarray(data["flows"])
        preds = data["preds"]
        gt_chars = data["gt_chars"]
        ids = data.get("sample_ids")
        sample_questions = data.get("sample_questions", {})
        sample_gts = data.get("sample_gts", {})

        if ids is not None:
            max_rows = min(len(flows), len(preds), len(gt_chars), len(ids))
            for row in range(max_rows):
                sid = int(ids[row])
                if sid in sample_questions:
                    question = sample_questions[sid]
                    gt_str = str(sample_gts.get(sid, ""))
                    if not gt_str:
                        operands = parse_question_operands(question)
                        if not operands:
                            continue
                        gt_str = str(sum(operands))
                else:
                    if not (0 <= sid < len(dataset)):
                        continue
                    operands = dataset[sid]
                    question = " + ".join(str(x) for x in operands)
                    gt_value = sum(operands)
                    gt_str = str(gt_value)
                if pos_idx >= len(gt_str):
                    continue
                if gt_chars[row] != gt_str[pos_idx]:
                    continue

                raw_sum_val = compute_raw_sum(question, pos_idx)
                if raw_sum_val < 0:
                    continue
                raw_mod = raw_sum_val % 10
                c_potential = compute_c_potential(question, pos_idx)

                pred_token = preds[row]
                pred_digit = int(pred_token) if str(pred_token).isdigit() else -1

                flows_list.append(flows[row])
                raw_labels.append(raw_mod)
                carry_labels.append(c_potential)
                gt_digits.append(int(gt_str[pos_idx]))
                pred_digits.append(pred_digit)
                sample_ids.append(sid)
                pos_ids.append(pos_idx)
        else:
            for flow_idx, flow in enumerate(flows):
                if flow_idx >= len(dataset):
                    break
                operands = dataset[flow_idx]
                question = " + ".join(str(x) for x in operands)
                gt_value = sum(operands)
                gt_str = str(gt_value)
                if pos_idx >= len(gt_str):
                    continue
                raw_sum_val = compute_raw_sum(question, pos_idx)
                if raw_sum_val < 0:
                    continue
                raw_mod = raw_sum_val % 10
                c_potential = compute_c_potential(question, pos_idx)

                flows_list.append(flow)
                raw_labels.append(raw_mod)
                carry_labels.append(c_potential)
                gt_digits.append(int(gt_str[pos_idx]))
                pred_digits.append(-1)
                sample_ids.append(flow_idx)
                pos_ids.append(pos_idx)

    return (
        np.asarray(flows_list, dtype=np.float32),
        np.asarray(raw_labels, dtype=np.int64),
        np.asarray(carry_labels, dtype=np.float32),
        np.asarray(gt_digits, dtype=np.int64),
        np.asarray(pred_digits, dtype=np.int64),
        np.asarray(sample_ids, dtype=np.int64),
        np.asarray(pos_ids, dtype=np.int64),
    )


def select_test_subset(
    gt_digits: np.ndarray,
    pred_digits: np.ndarray,
    sample_ids: np.ndarray,
    test_ids: set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_mask = np.isin(sample_ids, list(test_ids)) if test_ids else np.zeros_like(sample_ids, dtype=bool)
    gt_test = gt_digits[test_mask] if test_mask.any() else gt_digits
    pred_test = pred_digits[test_mask] if test_mask.any() else pred_digits
    sample_ids_test = sample_ids[test_mask] if test_mask.any() else sample_ids
    return gt_test, pred_test, sample_ids_test, test_mask


def split_sample_ids(
    sample_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set[int], set[int], set[int]]:
    unique_samples = np.unique(sample_ids)
    total_samples = len(unique_samples)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train/val/test ratio sum must be > 0")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_samples)

    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count
    if total_samples >= 3:
        if train_count < 1:
            train_count = 1
        if val_count < 1:
            val_count = 1
        test_count = total_samples - train_count - val_count
        if test_count < 1:
            test_count = 1
            if train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

    train_ids = set(unique_samples[:train_count])
    val_ids = set(unique_samples[train_count:train_count + val_count])
    test_ids = set(unique_samples[train_count + val_count:train_count + val_count + test_count])
    return train_ids, val_ids, test_ids


def compute_token_sample_acc(
    pred_digits: np.ndarray,
    gt_digits: np.ndarray,
    sample_ids: np.ndarray,
) -> Tuple[float, float]:
    token_acc = float(np.mean(pred_digits == gt_digits)) if len(gt_digits) else 0.0
    unique_samples = np.unique(sample_ids)
    correct_samples = 0
    for sid in unique_samples:
        mask = sample_ids == sid
        if np.all(pred_digits[mask] == gt_digits[mask]):
            correct_samples += 1
    sample_acc = correct_samples / len(unique_samples) if len(unique_samples) else 0.0
    return token_acc, sample_acc


def load_h5_baseline_metrics(
    h5_path: Path,
    dataset_path: Path,
    positions: List[int] | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    dataset_full = load_dataset(dataset_path)
    positions_data = load_positions(h5_path)
    _, _, _, gt_digits, pred_digits, sample_ids, _ = build_flat_dataset(
        dataset_full,
        positions_data,
        positions_filter=positions,
    )

    _, _, test_ids = split_sample_ids(
        sample_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    gt_test, pred_test, sample_ids_test, test_mask = select_test_subset(
        gt_digits,
        pred_digits,
        sample_ids,
        test_ids,
    )
    token_acc, sample_acc = compute_token_sample_acc(pred_test, gt_test, sample_ids_test)

    metrics = {
        "orig_h5_token_acc": float(token_acc),
        "orig_h5_sample_acc": float(sample_acc),
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
