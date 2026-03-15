import logging
import sys
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from probe_data import compute_c_potential, load_dataset
except ImportError:
    raise ImportError("请确保 probe_data.py 在当前目录下")

# 配置
DEFAULT_H5_PATH = Path("VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5")
DEFAULT_DATASET_PATH = Path("VerticalFlow/num3len10-10000.pkl")
DEFAULT_MODEL_PATH = "/data/Models/Qwen3-4b"

logging.basicConfig(level=logging.INFO, format="%(message)s")

def analyze_alignment(h5_path, dataset_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 1. 加载模型 Head 和 Norm
    logging.info("Loading model components...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype="auto")
    lm_head = model.lm_head.to(device).eval()
    
    norm = None
    if hasattr(model, "model") and hasattr(model.model, "norm"): norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"): norm = model.transformer.ln_f
    else: norm = getattr(model, "norm", None)
    norm = norm.to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 获取数字 token ID (0-9)
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    # 获取对应数字的 Embedding 权重 (Unembedding Matrix Rows) W[d]
    # W 的形状是 (Vocab, Dim)
    W = lm_head.weight.detach() # (Vocab, Dim)
    
    # 2. 加载数据并分组
    logging.info("Loading data...")
    dataset = load_dataset(dataset_path)
    
    # 用于计算 Steering Vector
    vec_buckets = {d: {1: [], 2: []} for d in range(10)}
    # 用于测试的样本
    meta_samples = []
    stable_samples = []
    
    with h5py.File(h5_path, "r") as hf:
        positions_group = hf["all_token_results"]
        for pos_name, pos_group in positions_group.items():
            if not pos_name.startswith("pos_"): continue
            try: pos_idx = int(pos_name.split("_", 1)[1])
            except: continue
            
            labels = np.asarray(pos_group["labels"][:], dtype=bool)
            gt_chars = np.asarray(pos_group["gt_chars"][:]).astype(str)
            true_carry_arr = np.asarray(pos_group["true_in_carry"][:], dtype=float)
            sample_ids = np.asarray(pos_group["sample_ids"][:], dtype=int)
            flows = np.asarray(pos_group["flows"], dtype=np.float32)
            if flows.ndim == 3: flows = flows[:, -1, :]
            
            for i in range(len(labels)):
                if not labels[i]: continue
                if not gt_chars[i].isdigit(): continue
                gt_digit = int(gt_chars[i])
                c_int = int(true_carry_arr[i])
                raw_sum = (gt_digit - c_int) % 10
                
                # 计算 potential
                sid = sample_ids[i]
                c_pot = true_carry_arr[i]
                if 0 <= sid < len(dataset):
                    try: c_pot = compute_c_potential(" + ".join(str(x) for x in dataset[sid]), pos_idx)
                    except: pass
                
                vec = flows[i]
                
                # 收集用于计算 Steering 的向量
                if c_int in [1, 2]:
                    vec_buckets[raw_sum][c_int].append(vec)
                
                # 收集测试样本 (Carry=1)
                # 记录: 向量, GT数字(d_curr), 目标数字(d_tgt), 原始潜在值
                if c_int == 1:
                    item = {
                        'vec': vec,
                        'raw_sum': raw_sum,
                        'd_curr': gt_digit,
                        'd_tgt': (gt_digit + 1) % 10,
                        'c_pot': c_pot
                    }
                    if c_pot > 1.9:
                        meta_samples.append(item)
                    elif 1.4 <= c_pot <= 1.6:
                        stable_samples.append(item)

    logging.info(f"Meta Samples: {len(meta_samples)}, Stable Samples: {len(stable_samples)}")

    # 3. 计算 Steering Vectors 和 Readout Vectors
    logging.info("Computing Alignment Statistics...")
    
    alignments_steer = []   # v_steer vs (W_tgt - W_curr)
    
    meta_proj_steer = []    # Meta 在 v_steer 上的投影
    stable_proj_steer = []  # Stable 在 v_steer 上的投影
    
    meta_proj_readout = []   # Meta 在 Readout 上的投影
    stable_proj_readout = [] # Stable 在 Readout 上的投影
    
    meta_logit_diffs = []    # 实际 Logit 差
    stable_logit_diffs = []
    
    # 预计算每种 raw_sum 下的 steering vector
    steering_vecs = {}
    for d in range(10):
        v1 = vec_buckets[d][1]
        v2 = vec_buckets[d][2]
        if v1 and v2:
            # v = Mean(2) - Mean(1)
            steering_vecs[d] = np.mean(v2, axis=0) - np.mean(v1, axis=0)

    # 批处理计算 Logits
    def calc_logit_diffs(samples):
        if not samples: return []
        vecs = [s['vec'] for s in samples]
        # 必须转为 Model 的 dtype
        vecs_t = torch.tensor(np.stack(vecs), device=device, dtype=lm_head.weight.dtype)
        with torch.no_grad():
            normed = norm(vecs_t)
            logits = lm_head(normed)
            
        diffs = []
        for i, s in enumerate(samples):
            l_curr = logits[i, digit_ids[s['d_curr']]].item()
            l_tgt  = logits[i, digit_ids[s['d_tgt']]].item()
            diffs.append(l_tgt - l_curr)
        return diffs

    meta_logit_diffs = calc_logit_diffs(meta_samples)
    stable_logit_diffs = calc_logit_diffs(stable_samples)

    # 逐个样本分析几何关系
    # 注意：这里的投影计算在 Norm 之前的空间进行（Raw Residual Space）
    
    # 为了公平对比，我们需要获取 W_tgt - W_curr。
    # 但 W 作用于 Norm 之后的向量。
    # 我们这里近似认为 Norm 是一个缩放，主要关注方向。
    
    combined_samples = [('Meta', s) for s in meta_samples] + [('Stable', s) for s in stable_samples]
    
    for group_name, s in combined_samples:
        d = s['raw_sum']
        if d not in steering_vecs: continue
        
        v_steer = steering_vecs[d]
        
        # 获取 Readout 方向: W[d_tgt] - W[d_curr]
        # 注意：这里我们忽略了 RMSNorm 的非线性，直接看权重差异
        w_curr = W[digit_ids[s['d_curr']]].float().cpu().numpy()
        w_tgt  = W[digit_ids[s['d_tgt']]].float().cpu().numpy()
        v_readout = w_tgt - w_curr
        
        # 计算 Steering Vector 与 Readout Vector 的余弦相似度 (只需算一次 per digit)
        if group_name == 'Meta' and len(alignments_steer) < 10: # 采样记录
             cos_align = np.dot(v_steer, v_readout) / (np.linalg.norm(v_steer) * np.linalg.norm(v_readout))
             alignments_steer.append(cos_align)
        
        # 投影计算
        # Proj = x . v / |v|
        p_steer = np.dot(s['vec'], v_steer) / np.linalg.norm(v_steer)
        p_readout = np.dot(s['vec'], v_readout) / np.linalg.norm(v_readout)
        
        if group_name == 'Meta':
            meta_proj_steer.append(p_steer)
            meta_proj_readout.append(p_readout)
        else:
            stable_proj_steer.append(p_steer)
            stable_proj_readout.append(p_readout)

    logging.info("\n=== 核心结果 ===")
    logging.info(f"1. Steering Vector vs Readout Vector Alignment (Mean Cosine): {np.mean(alignments_steer):.4f}")
    if np.mean(alignments_steer) < 0.1:
        logging.warning("   -> 警告: Steering 向量与 Readout 方向几乎正交！这解释了为何 Steering 有效但 Logit 初始无差异。")
    
    logging.info("\n2. Projection on Steering Vector (Internal Geometry)")
    logging.info(f"   Meta Mean:   {np.mean(meta_proj_steer):.4f}")
    logging.info(f"   Stable Mean: {np.mean(stable_proj_steer):.4f}")
    logging.info(f"   Diff:        {np.mean(meta_proj_steer) - np.mean(stable_proj_steer):.4f}")
    
    logging.info("\n3. Projection on Readout Vector (Output Logits Geometry - PreNorm)")
    logging.info(f"   Meta Mean:   {np.mean(meta_proj_readout):.4f}")
    logging.info(f"   Stable Mean: {np.mean(stable_proj_readout):.4f}")
    logging.info(f"   Diff:        {np.mean(meta_proj_readout) - np.mean(stable_proj_readout):.4f}")
    
    logging.info("\n4. Actual Logit Difference (Post-Norm + Head)")
    logging.info(f"   Meta Mean:   {np.mean(meta_logit_diffs):.4f}")
    logging.info(f"   Stable Mean: {np.mean(stable_logit_diffs):.4f}")
    logging.info(f"   Diff:        {np.mean(meta_logit_diffs) - np.mean(stable_logit_diffs):.4f}")

if __name__ == "__main__":
    analyze_alignment(DEFAULT_H5_PATH, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)