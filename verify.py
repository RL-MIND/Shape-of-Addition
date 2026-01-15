import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_H5_PATH = Path("VerticalFlow/results/plus_num3len10_Qwen3-4b.h5")
DEFAULT_MODEL_PATH = "/data/Models/Qwen3-4b"


def load_position_arrays(h5_path: Path):
    """Load labels, gt_chars, true_in_carry, and last-layer flows per position."""
    with h5py.File(h5_path, "r") as hf:
        positions_group = hf["all_token_results"]
        labels_by_pos = {}
        gt_by_pos = {}
        true_carry_by_pos = {}
        last_layer_by_pos = {}
        for pos_name, pos_group in positions_group.items():
            if not pos_name.startswith("pos_"):
                continue
            try:
                pos_idx = int(pos_name.split("_", 1)[1])
            except Exception:
                continue
            labels_ds = pos_group.get("labels")
            gt_ds = pos_group.get("gt_chars")
            true_carry_ds = pos_group.get("true_in_carry")
            flows_ds = pos_group.get("flows")
            if labels_ds is None or gt_ds is None or true_carry_ds is None or flows_ds is None:
                continue
            labels_by_pos[pos_idx] = np.asarray(labels_ds[:], dtype=bool)
            gt_by_pos[pos_idx] = np.asarray(gt_ds[:]).astype(str)
            true_carry_by_pos[pos_idx] = np.asarray(true_carry_ds[:], dtype=float)
            flows_np = np.asarray(flows_ds)
            if flows_np.ndim != 3:
                raise ValueError(f"Unexpected flows ndim={flows_np.ndim} for {pos_name}")
            last_layer_by_pos[pos_idx] = flows_np[:, -1, :]
    return labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos


def collect_records(labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos, positions=None):
    """Flatten per-position arrays into records with vector, digit, carry, and label."""
    records = []
    for p, labels in labels_by_pos.items():
        if positions is not None and p not in positions:
            continue
        gt_arr = gt_by_pos.get(p)
        carry_arr = true_carry_by_pos.get(p)
        vec_arr = last_layer_by_pos.get(p)
        if gt_arr is None or carry_arr is None or vec_arr is None:
            continue
        n = min(len(labels), len(gt_arr), len(carry_arr), len(vec_arr))
        for i in range(n):
            ch = gt_arr[i]
            if len(ch) != 1 or not ch.isdigit():
                continue
            carry_val = carry_arr[i]
            if not np.isfinite(carry_val):
                continue
            c_int = int(carry_val)
            if c_int not in (0, 1, 2):
                continue
            records.append({
                "pos": p,
                "digit": int(ch),
                "carry": c_int,
                "label": bool(labels[i]),
                "vec": vec_arr[i],
            })
    return records


def compute_means(records: List[dict]):
    """Compute per-digit means for carries 0/1/2 using only correct records."""
    buckets = {d: {0: [], 1: [], 2: []} for d in range(10)}
    for r in records:
        if not r["label"]:
            continue
        buckets[r["digit"]][r["carry"]].append(r["vec"])

    means: Dict[int, Dict[int, Optional[np.ndarray]]] = {d: {0: None, 1: None, 2: None} for d in range(10)}
    counts: Dict[int, Dict[int, int]] = {d: {0: 0, 1: 0, 2: 0} for d in range(10)}
    for d in range(10):
        for c in (0, 1, 2):
            vecs = buckets[d][c]
            counts[d][c] = len(vecs)
            if vecs:
                means[d][c] = np.mean(np.stack(vecs, axis=0), axis=0)
    return means, counts


def build_dirs_cross_digit(means: Dict[int, Dict[int, Optional[np.ndarray]]]):
    """Cross-digit directions as in analyze_carry_similarity.py.

    dir01[d] = mean(d+1, carry1) - mean(d, carry0)
    dir12[d] = mean(d+1, carry2) - mean(d, carry1)
    (digit index wraps mod 10)
    """
    dir01 = {}
    dir12 = {}
    for d in range(10):
        next_d = (d + 1) % 10
        m0 = means[d].get(0)
        m1 = means[d].get(1)
        m1_next = means[next_d].get(1)
        m2_next = means[next_d].get(2)
        dir01[d] = m1_next - m0 if m0 is not None and m1_next is not None else None
        dir12[d] = m2_next - m1 if m1 is not None and m2_next is not None else None
    return dir01, dir12


def build_combined_dir(dir01: Dict[int, Optional[np.ndarray]], dir12: Dict[int, Optional[np.ndarray]]):
    """Average available dir01/dir12 per digit to get a single carry direction."""
    combined = {}
    for d in range(10):
        parts = []
        if dir01.get(d) is not None:
            parts.append(dir01[d])
        if dir12.get(d) is not None:
            parts.append(dir12[d])
        if parts:
            combined[d] = np.mean(np.stack(parts, axis=0), axis=0)
        else:
            combined[d] = None
    return combined


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_lm_head_and_tokenizer(model_path: str, device: torch.device):
    logging.info("Loading model (for lm_head) from %s ...", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype="auto")
    lm_head = model.lm_head.to(device)
    lm_head.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    del model
    return lm_head, tokenizer


def digit_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    ids = []
    for d in range(10):
        tokens = tokenizer.encode(str(d), add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Digit {d} is not a single token for this tokenizer: {tokens}")
        ids.append(tokens[0])
    return ids


def predict_digits(vectors: List[np.ndarray], lm_head, digit_ids: List[int], device: torch.device, batch_size: int = 256) -> List[int]:
    if not vectors:
        return []
    head_dtype = next(lm_head.parameters()).dtype
    preds: List[int] = []
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start + batch_size]
            batch_tensor = torch.tensor(np.stack(batch), dtype=head_dtype, device=device)
            logits = lm_head(batch_tensor)
            digit_logits = logits[:, digit_ids].float()
            pred_idx = torch.argmax(digit_logits, dim=-1).tolist()
            preds.extend(pred_idx)
    return preds


def evaluate(records: List[dict], mode: str, dir01, dir12, combined_dir, lm_head, digit_ids, device: torch.device, batch_size: int, use_combined: bool):
    """
    Evaluate accuracy under different modes.
    mode: baseline | add | sub
    add: carry in {0,1}; direction = combined[d] if use_combined else dir12[d]; target digit = (d+1)%10
    sub: carry in {1,2}; direction = combined[d-1] if use_combined else dir01[d-1]; target digit = (d+9)%10
    baseline: always evaluate vs true digit
    Returns (accuracy, used_count)
    """
    xs = []
    targets = []
    metas = []  # keep (digit, carry) for failure reporting
    for r in records:
        d = r["digit"]
        c = r["carry"]
        if mode == "baseline":
            xs.append(r["vec"])
            targets.append(d)
        elif mode == "add":
            if c not in (0, 1):
                continue
            dv = (combined_dir.get(d) if use_combined else dir12.get(d))
            if dv is None:
                continue
            xs.append(r["vec"] + dv)
            targets.append((d + 1) % 10)
            metas.append((d, c))
        elif mode == "sub":
            if c not in (1, 2):
                continue
            prev_d = (d + 9) % 10
            dv = (combined_dir.get(prev_d) if use_combined else dir01.get(prev_d))
            if dv is None:
                continue
            xs.append(r["vec"] - dv)
            targets.append((d + 9) % 10)
            metas.append((d, c))
        else:
            raise ValueError(f"Unknown mode {mode}")
    preds = predict_digits(xs, lm_head, digit_ids, device, batch_size)
    if not preds:
        return float("nan"), 0, []
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    acc = float(np.mean(np.equal(preds_arr, targets_arr)))
    failures = []
    if metas:
        for p, t, meta in zip(preds_arr, targets_arr, metas):
            if p != t:
                failures.append(meta)
    return acc, len(targets), failures


def sample_records(records: List[dict], count: int, rng: np.random.Generator):
    if len(records) <= count:
        return list(records)
    idx = rng.choice(len(records), size=count, replace=False)
    return [records[i] for i in idx]


def main():
    parser = argparse.ArgumentParser(description="Verify carry directions by nudging last-layer vectors and unembedding.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH, help="Path to HDF5 results file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model path for loading lm_head/tokenizer")
    parser.add_argument("--positions", type=int, nargs="*", help="Optional position filter (e.g., 0 1 2)")
    parser.add_argument("--correct-count", type=int, default=1000, help="Number of correct tokens to sample")
    parser.add_argument("--incorrect-count", type=int, default=1000, help="Number of incorrect tokens to sample")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for lm_head forward")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device for lm_head inference (auto|cuda|cpu)")
    parser.add_argument("--use-combined", action="store_true", help="Use per-digit averaged carry direction (avg of dir01/dir12)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos = load_position_arrays(args.h5)
    pos_filter = set(args.positions) if args.positions else None
    records = collect_records(labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos, positions=pos_filter)
    logging.info("Loaded %d usable records (positions filter: %s)", len(records), pos_filter if pos_filter else "all")

    correct_records = [r for r in records if r["label"]]
    wrong_records = [r for r in records if not r["label"]]
    logging.info("Correct records: %d | Incorrect records: %d", len(correct_records), len(wrong_records))

    means, counts = compute_means(records)
    dir01, dir12 = build_dirs_cross_digit(means)
    combined_dir = build_combined_dir(dir01, dir12)
    for d in range(10):
        logging.info(
            "Digit %d counts c0/c1/c2 = (%d, %d, %d)",
            d, counts[d][0], counts[d][1], counts[d][2],
        )

    lm_head, tokenizer = load_lm_head_and_tokenizer(args.model, device)
    digit_ids = digit_token_ids(tokenizer)
    logging.info("Digit token ids: %s", digit_ids)

    rng = np.random.default_rng(args.seed)
    sampled_correct = sample_records(correct_records, args.correct_count, rng)
    sampled_wrong = sample_records(wrong_records, args.incorrect_count, rng)
    combined = sampled_correct + sampled_wrong

    use_combined = bool(args.use_combined)

    results = {}
    results["correct_add"] = evaluate(sampled_correct, "add", dir01, dir12, combined_dir, lm_head, digit_ids, device, args.batch_size, use_combined)
    results["correct_sub"] = evaluate(sampled_correct, "sub", dir01, dir12, combined_dir, lm_head, digit_ids, device, args.batch_size, use_combined)

    results["wrong_add"] = evaluate(sampled_wrong, "add", dir01, dir12, combined_dir, lm_head, digit_ids, device, args.batch_size, use_combined)
    results["wrong_sub"] = evaluate(sampled_wrong, "sub", dir01, dir12, combined_dir, lm_head, digit_ids, device, args.batch_size, use_combined)
    

    logging.info("\n=== Verification (accuracy, count used) ===")
    logging.info("Correct :   %.4f (%d)", (results["correct_add"][0] * results["correct_add"][1]) / results["correct_add"][1], results["correct_add"][1]+results["correct_sub"][1])
    logging.info("  Correct +:         %.4f (%d)", results["correct_add"][0], results["correct_add"][1])
    logging.info("  Correct -:         %.4f (%d)", results["correct_sub"][0], results["correct_sub"][1])
    logging.info("Wrong :     %.4f (%d)", (results["wrong_add"][0] * results["wrong_add"][1]) / results["wrong_add"][1], results["wrong_add"][1]+results["wrong_sub"][1])
    logging.info("  Wrong +:           %.4f (%d)", results["wrong_add"][0], results["wrong_add"][1])
    logging.info("  Wrong -:           %.4f (%d)", results["wrong_sub"][0], results["wrong_sub"][1])
    logging.info("All :       %.4f (%d)", (results["correct_add"][0] * results["correct_add"][1] + results["wrong_add"][0] * results["wrong_add"][1]) / (results["correct_add"][1]+results["correct_sub"][1]+results["wrong_add"][1]+results["wrong_sub"][1]), results["correct_add"][1]+results["correct_sub"][1]+results["wrong_add"][1]+results["wrong_sub"][1])

    # Failure breakdown for wrong add/sub
    def log_failures(name, failures):
        if not failures:
            logging.info("%s failures: none", name)
            return
        counts = {}
        for d, c in failures:
            key = (d, c)
            counts[key] = counts.get(key, 0) + 1
        parts = [f"(digit={k[0]},carry={k[1]}):{v}" for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
        logging.info("%s failures count=%d detail=%s", name, len(failures), "; ".join(parts))

    log_failures("Wrong +", results["wrong_add"][2])
    log_failures("Wrong -", results["wrong_sub"][2])


if __name__ == "__main__":
    main()
