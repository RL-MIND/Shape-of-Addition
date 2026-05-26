import argparse
import copy
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models import CircularProbe, ProbeMLP

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required to load HDF5 result files. Install via `pip install h5py`." ) from exc

# Default configuration
DEFAULT_DATA_PATH = Path(
    "results/activations/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5"
)
DEFAULT_PROBE_TYPE = "mlp"  # choices: linear, mlp, circular
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 20
DEFAULT_MAX_EPOCHS = 200
DEFAULT_TEST_SIZE = 0.1
DEFAULT_SEED = 42

LOG_DIR = Path("results/logs/log_incarry")
SAVE_DIR = Path("results/checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Model factory
# ----------------------------


def build_model(probe_type: str, input_dim: int, num_classes: int) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(input_dim, num_classes).to(DEVICE)
    if probe_type == "mlp":
        return ProbeMLP(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    if probe_type == "circular":
        return CircularProbe(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    raise ValueError(f"Unsupported probe type: {probe_type}")


# ----------------------------
# Data loading
# ----------------------------

def _extract_positions(positions_group) -> Tuple[List[int], List[str]]:
    numeric_positions = list(positions_group.attrs.get("numeric_positions", []))
    string_positions = list(positions_group.attrs.get("string_positions", []))
    numeric_positions = [int(p) for p in numeric_positions]
    string_positions = [str(p) for p in string_positions]
    if not numeric_positions and not string_positions:
        for key in positions_group.keys():
            if key.startswith("pos_"):
                suffix = key[4:]
                if suffix.lstrip("-").isdigit():
                    numeric_positions.append(int(suffix))
                else:
                    string_positions.append(str(suffix))
            else:
                string_positions.append(str(key))
    return numeric_positions, string_positions


def load_carry_data(path: Path, include_extra: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load flows and carry labels from the HDF5 file."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    flows_list: List[np.ndarray] = []
    true_carry_list: List[np.ndarray] = []
    pred_carry_list: List[np.ndarray] = []

    with h5py.File(path, "r") as hf:
        positions_group = hf["all_token_results"]
        numeric_positions, string_positions = _extract_positions(positions_group)
        positions_to_load: List[str] = []

        for pos in numeric_positions:
            positions_to_load.append(f"pos_{pos}")
        for pos in string_positions:
            if pos == "extra" and not include_extra:
                continue
            positions_to_load.append(str(pos) if pos.startswith("pos_") else pos)

        print(f"Found positions: numeric={numeric_positions}, string={string_positions}")
        print(f"Loading positions: {positions_to_load}")

        for pos_name in positions_to_load:
            if pos_name not in positions_group:
                continue
            pos_group = positions_group[pos_name]
            flows_ds = pos_group.get("flows")
            if flows_ds is None:
                flows_ds = pos_group.get("flows_post_ffn")
            if flows_ds is None:
                continue
            flows = flows_ds[:].astype(np.float32)
            true_carry = pos_group.get("true_in_carry")
            if true_carry is None:
                true_carry = pos_group.get("incoming_carries")
            pred_carry = pos_group.get("pred_in_carry")
            if pred_carry is None:
                pred_carry = pos_group.get("outgoing_carries")
            if true_carry is None or pred_carry is None:
                continue
            true_carry = np.asarray(true_carry[:], dtype=np.float32)
            pred_carry = np.asarray(pred_carry[:], dtype=np.float32)
            if not (len(flows) == len(true_carry) == len(pred_carry)):
                print(f"Skip position {pos_name}: mismatched lengths")
                continue
            flows_list.append(flows)
            true_carry_list.append(true_carry)
            pred_carry_list.append(pred_carry)
            print(f"  Loaded {pos_name}: {len(flows)} samples, flow shape {flows.shape[1:]}" )

    if not flows_list:
        raise RuntimeError("No usable data loaded. Check that true_in_carry and pred_in_carry exist.")

    flows_all = np.concatenate(flows_list, axis=0)
    true_all = np.concatenate(true_carry_list, axis=0)
    pred_all = np.concatenate(pred_carry_list, axis=0)
    return flows_all, true_all, pred_all


def load_carry_data_with_sample_ids(
    path: Path, include_extra: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load flows, carries, and optional sample_ids if present in H5.

    Returns (flows_all, true_all, pred_all, sample_ids_all or None).
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    flows_list: List[np.ndarray] = []
    true_carry_list: List[np.ndarray] = []
    pred_carry_list: List[np.ndarray] = []
    sample_ids_list: List[np.ndarray] = []
    all_have_ids = True

    with h5py.File(path, "r") as hf:
        positions_group = hf["all_token_results"]
        numeric_positions, string_positions = _extract_positions(positions_group)
        positions_to_load: List[str] = []

        for pos in numeric_positions:
            positions_to_load.append(f"pos_{pos}")
        for pos in string_positions:
            if pos == "extra" and not include_extra:
                continue
            positions_to_load.append(str(pos) if pos.startswith("pos_") else pos)

        for pos_name in positions_to_load:
            if pos_name not in positions_group:
                continue
            pos_group = positions_group[pos_name]
            flows_ds = pos_group.get("flows")
            if flows_ds is None:
                flows_ds = pos_group.get("flows_post_ffn")
            if flows_ds is None:
                continue
            flows = flows_ds[:].astype(np.float32)
            true_carry = pos_group.get("true_in_carry")
            if true_carry is None:
                true_carry = pos_group.get("incoming_carries")
            pred_carry = pos_group.get("pred_in_carry")
            if pred_carry is None:
                pred_carry = pos_group.get("outgoing_carries")
            sample_ids_ds = pos_group.get("sample_ids")
            if sample_ids_ds is None:
                sample_ids_ds = pos_group.get("sample_indices")
            if true_carry is None or pred_carry is None:
                continue
            true_carry = np.asarray(true_carry[:], dtype=np.float32)
            pred_carry = np.asarray(pred_carry[:], dtype=np.float32)
            if not (len(flows) == len(true_carry) == len(pred_carry)):
                print(f"Skip position {pos_name}: mismatched lengths")
                all_have_ids = False
                continue
            sample_ids_pos: Optional[np.ndarray] = None
            if sample_ids_ds is not None:
                sample_ids_pos = np.asarray(sample_ids_ds[:], dtype=np.int64)
                if sample_ids_pos is not None and len(sample_ids_pos) != len(flows):
                    print(f"Skip sample_ids for {pos_name}: length mismatch")
                    sample_ids_pos = None
            else:
                all_have_ids = False

            flows_list.append(flows)
            true_carry_list.append(true_carry)
            pred_carry_list.append(pred_carry)
            if sample_ids_pos is not None:
                sample_ids_list.append(sample_ids_pos)
            else:
                all_have_ids = False
            print(
                f"  Loaded {pos_name}: {len(flows)} samples, flow shape {flows.shape[1:]}, "
                f"sample_ids={'yes' if sample_ids_pos is not None else 'no'}"
            )

    if not flows_list:
        raise RuntimeError("No usable data loaded. Check that true_in_carry and pred_in_carry exist.")

    flows_all = np.concatenate(flows_list, axis=0)
    true_all = np.concatenate(true_carry_list, axis=0)
    pred_all = np.concatenate(pred_carry_list, axis=0)
    sample_ids_all: Optional[np.ndarray]
    if all_have_ids and sample_ids_list:
        sample_ids_all = np.concatenate(sample_ids_list, axis=0)
    else:
        sample_ids_all = None
    return flows_all, true_all, pred_all, sample_ids_all


# ----------------------------
# Training helpers
# ----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    loss_avg = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds) if all_labels else float("nan")
    try:
        unique_labels = np.unique(all_labels)
        if len(unique_labels) >= 2:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")  # type: ignore[arg-type]
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    return {"loss": loss_avg, "acc": acc, "auc": auc}


def train_single_layer(
    X_layer: torch.Tensor,
    y: torch.Tensor,
    probe_type: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    max_epochs: int,
    test_size: float,
) -> Tuple[nn.Module, float, float, int]:
    """Train one layer and return model plus best validation accuracy, AUC, and epoch."""
    num_classes = int(torch.unique(y).numel())
    if num_classes < 2:
        raise ValueError("Need at least two classes for training.")

    indices = np.arange(len(X_layer))
    strat_labels = y.numpy()
    stratify_param = strat_labels if len(np.unique(strat_labels)) > 1 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=DEFAULT_SEED,
        stratify=stratify_param,
    )

    X_train, X_val = X_layer[train_idx], X_layer[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_layer.shape[1]
    model = build_model(probe_type, input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = float("-inf")
    best_acc = 0.0
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate(model, val_loader, criterion)
        monitor_metric = val_metrics["auc"]
        if math.isnan(monitor_metric):
            monitor_metric = val_metrics["acc"]

        if monitor_metric > best_auc:
            best_auc = monitor_metric
            best_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:03d} | TrainLoss {running_loss/max(len(train_loader),1):.4f} "
            f"| ValLoss {val_metrics['loss']:.4f} | ValAcc {val_metrics['acc']:.4f} "
            f"| ValAUC {val_metrics['auc']:.4f} | Patience {no_improve}/{patience}"
        )

        if no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_acc, best_auc, best_epoch


# ----------------------------
# Main
# ----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-layer carry probes for true/pred carries.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_DATA_PATH, help="Path to HDF5 results file")
    parser.add_argument("--probe-type", type=str, choices=["linear", "mlp", "circular"], default=DEFAULT_PROBE_TYPE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-extra", action="store_true", help="Include extra position if present")
    parser.add_argument("--layer", type=int, default=None, help="Layer index for head-specific probe on true_in_carry")
    parser.add_argument("--head-index", type=int, default=None, help="Head index to slice (requires num-heads)")
    parser.add_argument("--num-heads", type=int, default=None, help="Total number of heads for slicing features evenly")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    log_dir = args.log_dir
    save_dir = args.save_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  data_path: {args.h5}")
    print(f"  probe_type: {args.probe_type}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  patience: {args.patience}")
    print(f"  max_epochs: {args.epochs}")
    print(f"  test_size: {args.test_size}")
    print(f"  device: {DEVICE}")

    flows, true_carry, pred_carry = load_carry_data(args.h5, include_extra=args.include_extra)
    seq_len, feature_dim = flows.shape[1], flows.shape[2]
    print(f"Loaded total samples: {len(flows)}, seq_len={seq_len}, feature_dim={feature_dim}")

    X_all = torch.tensor(flows, dtype=torch.float32)
    true_labels = torch.tensor(true_carry, dtype=torch.float32)
    pred_labels = torch.tensor(pred_carry, dtype=torch.float32)

    targets = {
        "true_in_carry": true_labels,
        "pred_in_carry": pred_labels,
    }

    results: Dict[str, List[Dict[str, object]]] = {"true_in_carry": [], "pred_in_carry": []}

    for target_name, labels_tensor in targets.items():
        mask = torch.isfinite(labels_tensor)
        labels_filtered = labels_tensor[mask]
        X_filtered = X_all[mask]
        labels_int = labels_filtered.long()
        unique_vals = torch.unique(labels_int)
        print(f"\nTraining for {target_name}: {len(labels_int)} samples, classes={unique_vals.tolist()}")
        if len(labels_int) < 2 or unique_vals.numel() < 2:
            print(f"  Skip {target_name}: not enough valid samples or classes")
            continue

        X_layers = X_filtered  # (N, L, D)
        for layer_idx in range(seq_len):
            X_layer = X_layers[:, layer_idx, :]
            try:
                model, best_acc, best_auc, best_epoch = train_single_layer(
                    X_layer,
                    labels_int,
                    probe_type=args.probe_type,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    max_epochs=args.epochs,
                    test_size=args.test_size,
                )
                results[target_name].append({
                    "layer": layer_idx,
                    "acc": best_acc,
                    "auc": best_auc,
                    "epoch": best_epoch,
                    "model": model,
                })
                print(f"Layer {layer_idx:02d} | {target_name} | BestAcc {best_acc:.4f} | BestAUC {best_auc:.4f} | Epoch {best_epoch}")
            except ValueError as err:
                print(f"Layer {layer_idx:02d} | {target_name} skipped: {err}")

        # Optional head-specific training for true_in_carry
        if target_name == "true_in_carry" and args.head_index is not None:
            if args.layer is None or args.num_heads is None:
                print("Head-specific training skipped: both --layer and --num-heads are required when --head-index is set.")
            elif not (0 <= args.layer < seq_len):
                print("Head-specific training skipped: head_layer out of range.")
            else:
                head_dim = feature_dim // args.num_heads
                if feature_dim % args.num_heads != 0:
                    print("Head-specific training skipped: feature_dim not divisible by num_heads.")
                else:
                    start = head_dim * args.head_index
                    end = start + head_dim
                    if end > feature_dim:
                        print("Head-specific training skipped: head_index exceeds available heads.")
                    else:
                        X_head = X_layers[:, args.layer, start:end]
                        try:
                            model, best_acc, best_auc, best_epoch = train_single_layer(
                                X_head,
                                labels_int,
                                probe_type=args.probe_type,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                patience=args.patience,
                                max_epochs=args.epochs,
                                test_size=args.test_size,
                            )
                            results.setdefault("true_in_carry_head", []).append({
                                "layer": args.layer,
                                "head_index": args.head_index,
                                "num_heads": args.num_heads,
                                "acc": best_acc,
                                "auc": best_auc,
                                "epoch": best_epoch,
                                "model": model,
                            })
                            print(
                                f"Head slice | Layer {args.layer:02d} Head {args.head_index} | true_in_carry | "
                                f"BestAcc {best_acc:.4f} | BestAUC {best_auc:.4f} | Epoch {best_epoch}"
                            )
                        except ValueError as err:
                            print(f"Head slice training skipped: {err}")

    print("\nSummary (validation best accuracy per layer):")
    best_log_lines: List[str] = []
    for target_name, layer_results in results.items():
        if not layer_results:
            print(f"  {target_name}: no results")
            continue
        print(f"  {target_name}:")
        for item in layer_results:
            print(f"    Layer {item['layer']:02d}: Acc={item['acc']:.4f}, AUC={item['auc']:.4f}")
        best_item = max(layer_results, key=lambda x: (x["acc"], x["auc"]))
        line = (
            f"{target_name} best: Layer {best_item['layer']} "
            f"BestAcc: {best_item['acc']:.4f} | BestAUC: {best_item['auc']:.4f} | Epoch {best_item['epoch']}"
        )
        best_log_lines.append(line)
        print(f"  -> {line}")

        # Save best probe model
        model_to_save = best_item.get("model")
        if model_to_save is not None:
            save_path = save_dir / f"incarry_{target_name}_layer{best_item['layer']}.pt"
            torch.save(
                {
                    "model_state": model_to_save.state_dict(),
                    "probe_type": args.probe_type,
                    "target": target_name,
                    "layer": best_item["layer"],
                    "feature_dim": feature_dim,
                    "seq_len": seq_len,
                },
                save_path,
            )
            print(f"  Saved best {target_name} probe to {save_path}")

    # Save head-specific best if available
    if "true_in_carry_head" in results and results["true_in_carry_head"]:
        head_best = max(results["true_in_carry_head"], key=lambda x: (x["acc"], x["auc"]))
        line = (
            f"true_in_carry_head best: Layer {head_best['layer']} Head {head_best['head_index']} "
            f"BestAcc: {head_best['acc']:.4f} | BestAUC: {head_best['auc']:.4f} | Epoch {head_best['epoch']}"
        )
        best_log_lines.append(line)
        print(f"  -> {line}")
        model_to_save = head_best.get("model")
        if model_to_save is not None:
            save_path = save_dir / (
                f"incarry_true_head{head_best['head_index']}_layer{head_best['layer']}_h{head_best['num_heads']}.pt"
            )
            torch.save(
                {
                    "model_state": model_to_save.state_dict(),
                    "probe_type": args.probe_type,
                    "target": "true_in_carry_head",
                    "layer": head_best["layer"],
                    "head_index": head_best["head_index"],
                    "num_heads": head_best["num_heads"],
                    "feature_dim": feature_dim // head_best["num_heads"],
                },
                save_path,
            )
            print(f"  Saved head-specific true_in_carry probe to {save_path}")

    # Write best summaries to log file
    if best_log_lines:
        log_path = log_dir / "incarry_best.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(best_log_lines) + "\n")
        print(f"Best summaries appended to {log_path}")


if __name__ == "__main__":
    main()

