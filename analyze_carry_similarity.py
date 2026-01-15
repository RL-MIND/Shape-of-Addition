import argparse
import logging
from pathlib import Path
import numpy as np
import h5py


DEFAULT_H5_PATH = Path("VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity; returns NaN if vectors are empty or zero-norm."""
    if vec_a.size == 0 or vec_b.size == 0:
        return float("nan")
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def sample_internal_cosines(vectors, sample_pairs=50):
    """Sample cosine similarities within a list of vectors and bucket counts."""
    if len(vectors) < 2:
        return None
    rng = np.random.default_rng()
    n = len(vectors)
    sims = []
    for _ in range(min(sample_pairs, n * (n - 1) // 2)):
        i, j = rng.integers(0, n, size=2)
        while j == i:
            j = rng.integers(0, n)
        sims.append(cosine_similarity(vectors[i], vectors[j]))
    sims = [s for s in sims if np.isfinite(s)]
    bins = {"0.9-1": 0, "0.8-0.9": 0, "0.7-0.8": 0, "<0.7": 0}
    for s in sims:
        if s >= 0.9:
            bins["0.9-1"] += 1
        elif s >= 0.8:
            bins["0.8-0.9"] += 1
        elif s >= 0.7:
            bins["0.7-0.8"] += 1
        else:
            bins["<0.7"] += 1
    return {"total": len(sims), "bins": bins}


def load_position_arrays(h5_path: Path):
    """Load per-position arrays: labels, gt_chars, true_in_carry, and flows (last-layer vectors)."""
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
            # gt_chars as strings
            gt_by_pos[pos_idx] = np.asarray(gt_ds[:]).astype(str)
            true_carry_by_pos[pos_idx] = np.asarray(true_carry_ds[:], dtype=float)
            # take last layer vectors: flows shape (N, L, D) -> (N, D)
            flows_np = np.asarray(flows_ds)
            if flows_np.ndim != 3:
                raise ValueError(f"Unexpected flows ndim={flows_np.ndim} for {pos_name}")
            last_layer_by_pos[pos_idx] = flows_np[:, -1, :]
    return labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos


def build_sample_mask_whole_correct(labels_by_pos):
    """Whole-sample-correct mask without per-position alignment.

    For sample index s, it is considered whole-correct if for all positions p
    where s < len(labels_by_pos[p]), labels_by_pos[p][s] is True.
    The mask length is max length over positions.
    """
    if not labels_by_pos:
        return np.array([], dtype=bool)
    max_len = max(len(arr) for arr in labels_by_pos.values())
    mask = np.ones(max_len, dtype=bool)
    for arr in labels_by_pos.values():
        # Expand per-position to max_len notionally by ignoring indices beyond its length
        effective = np.ones(max_len, dtype=bool)
        effective[: len(arr)] = np.asarray(arr, dtype=bool)
        # For indices beyond len(arr), sample didn't have that position -> no constraint
        mask &= effective
    return mask


def filter_carries(carry_array: np.ndarray, mask: np.ndarray, min_len: int) -> np.ndarray:
    """Apply mask and drop NaNs."""
    if carry_array.size == 0 or min_len == 0:
        return np.array([], dtype=float)
    trimmed = np.asarray(carry_array[:min_len], dtype=float)
    filtered = trimmed[mask]
    if filtered.size == 0:
        return filtered
    return filtered[~np.isnan(filtered)]


def collect_vectors_by_digit_and_carry(labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos, positions=None):
    """Return dict[digit][carry] -> list of vectors, only from whole-correct samples and token-level correct.

    - Whole-sample correctness is computed across all positions via build_sample_mask_whole_correct.
    - Token-level correctness uses labels_by_pos[p][s].
    - Only use finite true_in_carry values.
    - Only use gt_chars that are single digits 0-9.
    - Vector is last layer hidden state for that token.
    """
    sample_mask = build_sample_mask_whole_correct(labels_by_pos)
    if sample_mask.size == 0:
        return {}

    buckets = {d: {0: [], 1: [], 2: []} for d in range(10)}
    max_len = sample_mask.size

    for p, labels in labels_by_pos.items():
        if positions is not None and len(positions) > 0 and p not in positions:
            continue
        n = len(labels)
        # Prepare aligned arrays for this position up to max_len notionally
        gt = gt_by_pos.get(p)
        tc = true_carry_by_pos.get(p)
        vecs = last_layer_by_pos.get(p)
        if gt is None or tc is None or vecs is None:
            continue
        # Iterate only over indices where this position exists
        upto = min(n, max_len)
        for s in range(upto):
            if not sample_mask[s]:
                continue
            if not bool(labels[s]):
                continue
            ch = gt[s]
            if len(ch) != 1 or not ch.isdigit():
                continue
            carry_val = tc[s]
            if not np.isfinite(carry_val):
                continue
            c_int = int(carry_val)
            if c_int not in (0, 1, 2):
                continue
            d_int = int(ch)
            buckets[d_int][c_int].append(vecs[s])

    return buckets


def analyze(h5_path: Path, use_pca: bool = False, pca_components: int = 3, positions=None):
    labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos = load_position_arrays(h5_path)
    pos_filter = set(positions) if positions else None
    buckets = collect_vectors_by_digit_and_carry(labels_by_pos, gt_by_pos, true_carry_by_pos, last_layer_by_pos, positions=pos_filter)

    results = []
    dir_c0_by_digit = [None] * 10  # direction for carry 0 -> carry 1 of next digit
    dir_c1_by_digit = [None] * 10  # direction for carry 1 -> carry 2 of next digit
    
    from sklearn.decomposition import PCA

    for d in range(10):
        groups = buckets.get(d, {0: [], 1: [], 2: []})
        means = {}
        counts = {}
        
        # Gather all vectors for this digit to fit PCA if requested
        all_vecs_d = []
        if use_pca:
            for c in (0, 1, 2):
                if groups.get(c):
                    all_vecs_d.extend(groups[c])
            
            pca_model = None
            if len(all_vecs_d) > pca_components:
                pca_model = PCA(n_components=pca_components)
                pca_model.fit(np.stack(all_vecs_d))
                explained = pca_model.explained_variance_ratio_
                logging.info("Digit %d PCA explained variance: %s (Total: %.4f)", d, explained, sum(explained))
        
        for c in (0, 1, 2):
            vecs = groups.get(c, [])
            counts[c] = len(vecs)
            # Internal similarity stats before averaging (on raw vectors)
            stats = sample_internal_cosines(vecs)
            if stats:
                logging.info("Digit %d carry %d internal cos: total=%d bins=%s", d, c, stats["total"], stats["bins"])
            if len(vecs) > 0:
                data = np.stack(vecs, axis=0)
                if use_pca and pca_model:
                     data = pca_model.transform(data)
                means[c] = np.mean(data, axis=0)
            else:
                means[c] = None

        cos_sim = float("nan")
        # direction for i=0: digit d carry0 -> digit (d+1)%10 carry1
        # direction for i=1: digit d carry1 -> digit (d+1)%10 carry2
        next_d = (d + 1) % 10
        # For the "next" digit, we need its means; will compute after we have all digits.
        results.append({
            "digit": d,
            "count_c0": counts[0],
            "count_c1": counts[1],
            "count_c2": counts[2],
            "means": means,
        })

    # After collecting means, compute directions that depend on digit pairs
    for d in range(10):
        next_d = (d + 1) % 10
        means_d = results[d]["means"]
        means_next = results[next_d]["means"] if next_d < len(results) else None
        dir0 = None
        dir1 = None
        if means_d and means_next:
            if means_d.get(0) is not None and means_next.get(1) is not None:
                dir0 = means_next[1] - means_d[0]
            if means_d.get(1) is not None and means_next.get(2) is not None:
                dir1 = means_next[2] - means_d[1]
        dir_c0_by_digit[d] = dir0
        dir_c1_by_digit[d] = dir1

    # Compute per-digit cosine between dir0 and dir1
    per_digit_cos = []
    for d in range(10):
        cos_val = float("nan")
        if dir_c0_by_digit[d] is not None and dir_c1_by_digit[d] is not None:
            cos_val = cosine_similarity(dir_c0_by_digit[d], dir_c1_by_digit[d])
        per_digit_cos.append(cos_val)
        counts = (results[d]["count_c0"], results[d]["count_c1"], results[d]["count_c2"])
        logging.info("Digit %d: count(c0,c1,c2)=(%d,%d,%d) cos(dir0,dir1)=%.6f", d, counts[0], counts[1], counts[2], cos_val)

    # Build cosine similarity matrices across digits
    def build_matrix(vectors, num=10):
        mat = np.full((num, num), np.nan, dtype=float)
        for i in range(num):
            vi = vectors[i]
            if vi is None:
                continue
            for j in range(num):
                vj = vectors[j]
                if vj is None:
                    continue
                mat[i, j] = cosine_similarity(vi, vj)
        return mat

    mat_dir0 = build_matrix(dir_c0_by_digit)
    mat_dir1 = build_matrix(dir_c1_by_digit)

    # Pretty print matrices
    def print_matrix(name, mat):
        logging.info("%s cosine similarity matrix (digits 0-9 rows/cols):", name)
        with np.printoptions(precision=4, suppress=True, linewidth=200):
            logging.info("\n%s\n", np.array2string(mat, formatter={"float_kind":lambda x: f"{x:.4f}" if np.isfinite(x) else "nan"}))

    print_matrix("dir0 (carry0@d -> carry1@d+1)", mat_dir0)
    print_matrix("dir1 (carry1@d -> carry2@d+1)", mat_dir1)

    return {
        "per_digit_cos": per_digit_cos,
        "matrix_dir0": mat_dir0,
        "matrix_dir1": mat_dir1,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze carry vectors similarity for correct predictions.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH, help="Path to HDF5 results file")
    parser.add_argument("--pca", action="store_true", help="Apply PCA to carry vectors before averaging")
    parser.add_argument("--pca-components", type=int, default=3, help="Number of PCA components to keep (only used when --pca)")
    parser.add_argument("--positions", type=int, nargs="*", help="Only use specified digit positions (e.g., 0 1 2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analyze(args.h5, use_pca=args.pca, pca_components=args.pca_components, positions=args.positions)


if __name__ == "__main__":
    main()
