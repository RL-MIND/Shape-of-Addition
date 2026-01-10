import os
import sys
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure current dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mera import LinearProbe, MLPProbe, SteeringController, MERAHook, SimpleSteeringController, SimpleHook
from generate_mera import MathGenerator

def load_training_data(h5_path, layer_idx, position=None):
    """
    Load flows (activations), labels, and preds from HDF5 for a specific layer.
    
    Args:
        h5_path: Path to HDF5 file
        layer_idx: Layer index to extract
        position: If specified, only load data from this position (e.g., 0 for pos_0)
                  If None, load all positions (original behavior)
    """
    pos_filter = f"pos_{position}" if position is not None else None
    print(f"Loading training data from {h5_path}, Layer {layer_idx}, Position filter: {pos_filter or 'ALL'}...")
    
    with h5py.File(h5_path, 'r') as f:
        flows_list = []
        labels_list = []
        preds_list = []
        
        if 'all_token_results' not in f:
            raise ValueError("No 'all_token_results' in file. Run generation first.")
            
        group = f['all_token_results']
        for pos_key in group.keys():
            # Filter by position if specified
            if pos_filter is not None and pos_key != pos_filter:
                continue
                
            pos_group = group[pos_key]
            if 'flows' in pos_group and 'labels' in pos_group:
                fl = pos_group['flows'][:]
                lb = pos_group['labels'][:]
                
                # Load preds if available
                pr = None
                if 'preds' in pos_group:
                    pr = pos_group['preds'][:]
                    # Decode bytes to str if needed
                    if pr.dtype.kind == 'S' or pr.dtype.kind == 'O':
                        pr = np.array([p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in pr])
                
                # fl shape: (N_samples, n_layers, hidden_dim) or similar
                if len(fl.shape) == 3:
                    if layer_idx < fl.shape[1]:
                        flows_list.append(fl[:, layer_idx, :])
                        labels_list.append(lb)
                        if pr is not None:
                            preds_list.append(pr)
                    else:
                        print(f"Warning: Layer {layer_idx} out of bounds {fl.shape}")
        
    if not flows_list:
        raise ValueError(f"No valid flow data found for position filter: {pos_filter}")
        
    X = np.concatenate(flows_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    preds = np.concatenate(preds_list, axis=0) if preds_list else None
    
    print(f"Loaded {X.shape[0]} samples")
    
    # y is 'correctness' (True/False). 
    # MERA ErrorProbe predicts ERROR probability.
    # So Target = 1 (Error) if y is False (Incorrect)
    # Target = 0 (No Error) if y is True (Correct)
    y_target = (~y).astype(float)
    
    return X, y_target, preds


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    好处：
    1. 降低易分类样本的权重：当模型对某个样本预测自信（p_t接近1）时，
       (1-p_t)^gamma 接近0，损失被大幅降低
    2. 关注难分类样本：当模型预测不自信时，损失保持较大
    3. 自动处理类别不平衡：不需要手动设置类别权重
    
    Args:
        alpha: 类别权重因子，用于平衡正负样本（默认0.25）
        gamma: 聚焦参数，gamma越大越关注难分类样本（默认2.0）
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        # logits: (batch, 1), targets: (batch, 1)
        probs = torch.sigmoid(logits)
        
        # p_t = p if y=1 else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t = alpha if y=1 else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE loss (without reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


def train_probe_model(probe, X_train, y_train, epochs=40, batch_size=64, lr=1e-3, device='cuda'):
    probe.to(device)
    probe.train()
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    
    # 计算类别分布
    n_pos = (y_train == 1).sum()  # Error samples
    n_neg = (y_train == 0).sum()  # Correct samples
    total = n_pos + n_neg
    
    print(f"Original class distribution: Positive(Error)={n_pos} ({n_pos/total*100:.1f}%), Negative(Correct)={n_neg} ({n_neg/total*100:.1f}%)")
    
    # === 数据集平衡：下采样多数类 ===
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    
    min_samples = min(len(pos_indices), len(neg_indices))
    
    # 随机采样使两类数量相等
    np.random.seed(42)
    if len(pos_indices) > min_samples:
        pos_indices = np.random.choice(pos_indices, min_samples, replace=False)
    if len(neg_indices) > min_samples:
        neg_indices = np.random.choice(neg_indices, min_samples, replace=False)
    
    balanced_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_train[balanced_indices]
    y_balanced = y_train[balanced_indices]
    
    print(f"Balanced dataset: {len(X_balanced)} samples (50% Error, 50% Correct)")
    
    X_tensor = torch.FloatTensor(X_balanced).to(device)
    y_tensor = torch.FloatTensor(y_balanced).unsqueeze(1).to(device)
    
    # 平衡后使用标准 BCE Loss（或 Focal Loss with alpha=0.5）
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training Probe with BCE Loss on balanced data...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = probe(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return probe

def main():
    layer=34
    train_limit=5000
    eval_limit = 500
    parser = argparse.ArgumentParser(description="MERA Task Runner")
    parser.add_argument('--data_path', type=str, default="MCOT/Vertical_Flow/num2len5-10000.pkl")
    parser.add_argument('--h5_results', type=str, default=f"MCOT/Vertical_Flow/results/mul_num2len5_Qwen3-8B/baseline_train_{train_limit}.h5")
    parser.add_argument('--steered_results_path', type=str, default="MCOT/Vertical_Flow/results/mul_num2len5_Qwen3-8B/steered_eval.h5")
    parser.add_argument('--layer', type=int, default=layer)
    parser.add_argument('--probe_type', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--alpha_steer', type=float, default=0.4)
    parser.add_argument('--alpha_backtrack', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train', action='store_true', help="Whether to train probe (requires existing h5 results)")
    parser.add_argument('--gen_baseline', action='store_true', help="Run baseline generation to create traning data")
    parser.add_argument('--eval_steered', action='store_true', help="Run steered evaluation")
    parser.add_argument('--probe_path', type=str, default=f"MCOT/Vertical_Flow/saved_models/mera_probe_Layer{layer}.pt")
    parser.add_argument('--simple', action='store_true', help="Use simple steering (replace with nearest avg vector)")
    parser.add_argument('--avg_vectors_path', type=str, default=f"MCOT/Vertical_Flow/saved_models/avg_vectors_Layer{layer}.pt")
    parser.add_argument('--position', type=int, default=None, help="Train/eval on specific position only (e.g., 0 for pos_0). If not set, uses all positions.")
    parser.add_argument('--test_probe_on_position', type=int, default=None, help="Test an existing probe (trained on all positions) on a specific position")
    parser.add_argument('--train_avg_vectors_only', action='store_true', help="Only compute position-specific avg_vectors using existing probe (no probe training)")
    
    args = parser.parse_args()
    
    # Shared Generator Instance
    gen = None
    
    # 1. Generate Baseline Data (if needed)
    if args.gen_baseline:
        print("Running Baseline Generation...")
        gen = MathGenerator(device=args.device)
        # Process a subset or full for training
        gen.process_dataset(args.data_path, args.h5_results, limit=train_limit) # Limit for demo speedi
    
    # 2. Train Probe
    probe = None
    if args.train:
        
        if not os.path.exists(args.h5_results):
            print(f"Error: Training data {args.h5_results} not found. Run --gen_baseline first.")
            return
        
        # Update paths if position is specified
        if args.position is not None:
            pos_suffix = f"_Pos{args.position}"
            if pos_suffix not in args.probe_path:
                args.probe_path = args.probe_path.replace('.pt', f'{pos_suffix}.pt')
            if pos_suffix not in args.avg_vectors_path:
                args.avg_vectors_path = args.avg_vectors_path.replace('.pt', f'{pos_suffix}.pt')
            print(f"Position-specific training: Position {args.position}")
            print(f"  Probe path: {args.probe_path}")
            print(f"  Avg vectors path: {args.avg_vectors_path}")

        train_layer_idx = args.layer + 1
        print(f"Loading training data for Hook Layer {args.layer} (Data Index {train_layer_idx})...")
        
        X, y, preds = load_training_data(args.h5_results, train_layer_idx, position=args.position)
        print(f"Training Data: {X.shape}, {y.shape}")
        if preds is not None:
            print(f"Preds shape: {preds.shape}")
        
        # Split (include preds for simple steering)
        if preds is not None:
            X_train, X_test, y_train, y_test, preds_train, preds_test = train_test_split(
                X, y, preds, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            preds_train, preds_test = None, None
        
        input_dim = X.shape[1]
        probe = LinearProbe(input_dim) if args.probe_type == 'linear' else MLPProbe(input_dim)
        
        probe = train_probe_model(probe, X_train, y_train, device=args.device)
        
        # Validate Probe
        probe.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(args.device)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(args.device)
            logits = probe(X_test_tensor)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Avoid BCEWithLogitsLoss on eval for metrics, just Acc
            acc = (preds == y_test_tensor).float().mean().item()
            
            # Calculating F1/Precision/Recall using sklearn
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            y_true_np = y_test
            y_prob_np = torch.sigmoid(logits).cpu().numpy()
            y_pred_np = (y_prob_np > 0.5).astype(int)
            
            sk_acc = accuracy_score(y_true_np, y_pred_np)
            sk_f1 = f1_score(y_true_np, y_pred_np)
            sk_auc = roc_auc_score(y_true_np, y_prob_np)
            
            print(f"\nProbe Validation Results:")
            print(f"  Accuracy: {sk_acc:.4f}")
            print(f"  F1 Score: {sk_f1:.4f}")
            print(f"  AUC-ROC:  {sk_auc:.4f}")
            
            if sk_auc < 0.6:
                print("WARNING: Probe performance is poor (AUC < 0.6). Steering may be ineffective.")

        # Save Probe
        torch.save(probe, args.probe_path)
        print(f"Probe saved to {args.probe_path}")
        
        # Compute and save average vectors for simple steering
        if preds_train is not None:
            print("\nComputing average vectors for simple steering...")
            avg_vectors = SimpleSteeringController.compute_avg_vectors(
                X_train, y_train, preds_train, device=args.device
            )
            torch.save(avg_vectors, args.avg_vectors_path)
            print(f"Average vectors saved to {args.avg_vectors_path}")
    
    # === NEW: Test combined probe on specific position ===
    if args.test_probe_on_position is not None:
        print(f"\n=== Testing Combined Probe on Position {args.test_probe_on_position} ===")
        
        # Load probe (trained on all positions)
        if probe is None:
            if os.path.exists(args.probe_path):
                print(f"Loading combined probe from {args.probe_path}")
                probe = torch.load(args.probe_path)
                probe.to(args.device)
            else:
                print(f"Error: No probe found at {args.probe_path}. Train one first.")
                return
        
        # Load position-specific data
        train_layer_idx = args.layer + 1
        X_pos, y_pos, preds_pos = load_training_data(args.h5_results, train_layer_idx, position=args.test_probe_on_position)
        print(f"Position {args.test_probe_on_position} Data: {X_pos.shape}")
        
        # Evaluate on position-specific data
        probe.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_pos).to(args.device)
            logits = probe(X_tensor)
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            y_prob_np = torch.sigmoid(logits).cpu().numpy()
            y_pred_np = (y_prob_np > 0.5).astype(int)
            
            acc = accuracy_score(y_pos, y_pred_np)
            f1 = f1_score(y_pos, y_pred_np, zero_division=0)
            
            try:
                auc = roc_auc_score(y_pos, y_prob_np)
            except:
                auc = 0.5  # If only one class present
            
            print(f"\nProbe Performance on Position {args.test_probe_on_position}:")
            print(f"  Samples: {len(y_pos)}")
            print(f"  Class distribution: Error={y_pos.sum():.0f} ({y_pos.sum()/len(y_pos)*100:.1f}%), Correct={(1-y_pos).sum():.0f} ({(1-y_pos).sum()/len(y_pos)*100:.1f}%)")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  AUC-ROC:  {auc:.4f}")
    
    # === NEW: Train avg_vectors only (using existing probe) ===
    if args.train_avg_vectors_only:
        print(f"\n=== Computing Position-Specific Avg Vectors ===")
        
        if args.position is None:
            print("Error: --train_avg_vectors_only requires --position to be set")
            return
        
        # Load position-specific data
        train_layer_idx = args.layer + 1
        X_pos, y_pos, preds_pos = load_training_data(args.h5_results, train_layer_idx, position=args.position)
        
        if preds_pos is None:
            print("Error: No preds data available for computing avg_vectors")
            return
        
        # Compute avg_vectors for this position
        avg_vectors = SimpleSteeringController.compute_avg_vectors(
            X_pos, y_pos, preds_pos, device=args.device
        )
        
        # Save with position suffix
        pos_suffix = f"_Pos{args.position}"
        avg_path = args.avg_vectors_path.replace('.pt', f'{pos_suffix}.pt')
        torch.save(avg_vectors, avg_path)
        print(f"Position {args.position} avg_vectors saved to {avg_path}")
    
    # 3. MERA Steered Evaluation
    if args.eval_steered:
        # Update paths if position is specified (for loading correct probe/avg_vectors)
        if args.position is not None:
            pos_suffix = f"_Pos{args.position}"
            if pos_suffix not in args.probe_path:
                args.probe_path = args.probe_path.replace('.pt', f'{pos_suffix}.pt')
            if pos_suffix not in args.avg_vectors_path:
                args.avg_vectors_path = args.avg_vectors_path.replace('.pt', f'{pos_suffix}.pt')
            print(f"Position-specific evaluation: Position {args.position}")
        
        if probe is None:
            if os.path.exists(args.probe_path):
                print(f"Loading probe from {args.probe_path}")
                probe = torch.load(args.probe_path)
                probe.to(args.device)
            else:
                print(f"Error: No probe found at {args.probe_path}. Train one first.")
                return

        print("\n=== Running MERA Steered Evaluation ===")
        
        # Setup Controller based on mode
        if args.simple:
            # Simple Steering Mode
            print("Mode: Simple (Replace with nearest avg vector)")
            
            # Load average vectors
            if os.path.exists(args.avg_vectors_path):
                print(f"Loading average vectors from {args.avg_vectors_path}")
                avg_vectors = torch.load(args.avg_vectors_path)
            else:
                print(f"Error: Average vectors not found at {args.avg_vectors_path}. Train first with --train.")
                return
            
            controller = SimpleSteeringController(probe, avg_vectors, device=args.device)
            steer_hook = SimpleHook(controller, args.layer)
            mode_name = "Simple"
        else:
            # MERA Steering Mode
            print(f"Mode: MERA (Alpha={args.alpha_steer}, Backtrack={args.alpha_backtrack})")
            
            controller = SteeringController(
                probe, 
                alpha_steer=args.alpha_steer, 
                alpha_backtrack=args.alpha_backtrack,
            )
            steer_hook = MERAHook(controller, args.layer)
            mode_name = "MERA"
        
        # Run Generation with Hook
        if gen is None:
            gen = MathGenerator(device=args.device)
        
        print(f"\nEvaluating Baseline (No Steering) on first {eval_limit} samples...")
        base_res = gen.process_dataset(args.data_path, "MCOT/Vertical_Flow/results/baseline_eval.h5", hooks=[], limit=eval_limit)
        
        print(f"\nEvaluating Steered ({mode_name}) on first {eval_limit} samples...")
        steer_res = gen.process_dataset(args.data_path, args.steered_results_path, hooks=[steer_hook], limit=eval_limit)
        
        # Calculate Metrics
        base_acc = base_res['token_acc']
        steer_acc = steer_res['token_acc']
        
        improved_acc = steer_acc - base_acc
        
        # SPI (Steering Performance Index) = Reduction in Error Rate
        # SPI = (Acc_steered - Acc_baseline) / (1 - Acc_baseline)
        if base_acc < 1.0:
            if steer_acc > base_acc:
                spi = (steer_acc - base_acc) / (1.0 - base_acc)
            else:
                spi = (steer_acc - base_acc) / base_acc
        else:
            spi = 0.0 # Already perfect
            
        print("\n" + "="*50)
        print("FINAL COMPARISON RESULTS")
        print("="*50)
        print(f"Baseline Accuracy: {base_acc:.4f}")
        print(f"Steered Accuracy:  {steer_acc:.4f}")
        print(f"Improved Acc:      {improved_acc:+.4f}")
        print(f"SPI (Score):       {spi:.4f}")
        print("="*50 + "\n")
        
        # Analysis could be done by comparing baseline_train.h5 and steered_eval.h5
        # but process_dataset logs accuracy, so valid enough for checking.

if __name__ == "__main__":
    main()