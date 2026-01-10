
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 尝试导入 h5py
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("提示: 安装 h5py 可以大幅加速大文件加载: pip install h5py")

# ==========================================
# ========== 配置参数区域 ==========
# ==========================================

# 日志配置
LOG_DIR = Path("MCOT/Vertical_Flow/log/log_baseline")
LOGGER = None
ORIGINAL_PRINT = print

# --- 1. 数据配置 ---
DATA_FILE_PATH = 'MCOT/Vertical_Flow/results/mul_num3len10_Qwen3-8B'
MODEL_PATH = 'Models/Qwen3-8B'  # Qwen3-8B 模型路径
BALANCE_DATASET = True           # 是否平衡数据集
BALANCE_BY_POSITION = True       # True: 对每个位置分别平衡, False: 所有位置混合平衡
TEST_SIZE = 0.2                  # 划分验证集比例
SEED = 42                        # 随机种子

# 选择使用哪些位置的数据
POSITION_SELECT = 'all'  # 'all', 'all_no_extra', [0,1,2], 0, 'extra'

# 按位置统计评估指标
EVALUATE_BY_POSITION = False      # 是否按位置分别报告 AUC/Acc/F1
# --- 2. 不确定性度量配置 ---
METRIC_TYPE = 'all'  # 'entropy', 'perplexity', 'lnpe', 'eubhd', 'max_prob', 'top5_entropy', 'self_certainty', 'all'
THRESHOLD_MODE = 'auto'  # 'auto': 自动搜索最佳阈值, 'manual': 手动指定
MANUAL_THRESHOLD = 0.5   # 手动指定的阈值（仅当 THRESHOLD_MODE='manual' 时生效）
THRESHOLD_SEARCH_STEPS = 100  # 阈值搜索步数

# --- 3. 按层评估配置 ---
EVALUATE_EACH_LAYER = True  # 是否对每一层单独评估
SPECIFIC_LAYER_INDEX = None  # None 或具体层索引（如 -1 表示最后一层）

# --- 4. 计算配置 ---
BATCH_SIZE = 256  # 批处理大小（用于加速 lm_head 推理）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ========== 随机种子设置 ==========
# ==========================================
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# ========== 日志工具 ==========
# ==========================================

def setup_logger():
    """初始化全局 LOGGER"""
    global LOGGER
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("classifier_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}.log"
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    LOGGER = logger
    ORIGINAL_PRINT(f"日志已初始化，输出到 {log_path}")
    return logger


def log_print(*args, **kwargs):
    """兼容原 print，额外将信息写入 LOGGER"""
    message = " ".join(str(a) for a in args)
    if LOGGER:
        LOGGER.info(message)
        return
    return ORIGINAL_PRINT(*args, **kwargs)


print = log_print


# ==========================================
# ========== 数据加载（复用 classifier.py）==========
# ==========================================

def load_data_from_pickle(file_path):
    """从 pickle 文件加载数据"""
    print(f"正在从 pickle 加载数据: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['all_token_results']


def load_data_from_hdf5(file_path, position_select='all'):
    """从 HDF5 文件加载数据"""
    print(f"正在从 HDF5 快速加载数据: {file_path}")
    
    all_token_results = {}
    
    with h5py.File(file_path, 'r') as hf:
        positions_group = hf['all_token_results']
        numeric_positions = list(positions_group.attrs.get('numeric_positions', []))
        string_positions = list(positions_group.attrs.get('string_positions', []))
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
        
        # 确定要加载的位置
        if position_select == 'all':
            positions_to_load = numeric_positions + string_positions
        elif position_select == 'all_no_extra':
            positions_to_load = numeric_positions
        elif position_select == 'extra':
            positions_to_load = ['extra'] if 'extra' in string_positions else []
        elif isinstance(position_select, int):
            positions_to_load = [position_select] if position_select in numeric_positions else []
        elif isinstance(position_select, list):
            positions_to_load = [p for p in position_select if p in numeric_positions + string_positions]
        else:
            positions_to_load = numeric_positions + string_positions
        
        print(f"  HDF5 中的位置: 数字={numeric_positions}, 字符串={string_positions}")
        print(f"  将加载的位置: {positions_to_load}")
        
        for pos in positions_to_load:
            if isinstance(pos, np.integer):
                pos = int(pos)
            pos_name = f"pos_{pos}" if isinstance(pos, int) else str(pos)
            
            if pos_name not in positions_group:
                continue
                
            pos_group = positions_group[pos_name]
            pos_data = {}
            
            # 加载 flows
            if 'flows' in pos_group:
                flows_array = pos_group['flows'][:]
                pos_data['flows'] = [flows_array[i] for i in range(len(flows_array))]
            
            # 加载标签
            if 'labels' in pos_group:
                pos_data['labels'] = list(pos_group['labels'][:])
            
            # 加载预测 tokens
            if 'preds' in pos_group:
                preds_array = pos_group['preds'][:]
                pos_data['preds'] = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in preds_array]
            
            all_token_results[pos] = pos_data
            print(f"    位置 {pos}: 加载完成")
    
    return all_token_results


def load_flows_and_labels(file_path, position_select='all'):
    """
    加载 flows 和 labels 数据
    
    Returns:
        flows_all: List[np.ndarray], 每个元素形状 (num_layers, hidden_dim)
        labels_all: List[int], 标签列表 (0 或 1)
        tokens_all: List[str], 预测的 token 字符串
        position_indices: List[int], 每个样本对应的位置索引
        selected_positions: List, 实际使用的位置列表
        num_layers: int, 层数
        hidden_dim: int, 隐藏层维度
    """
    file_path = Path(file_path)
    h5_path = file_path.with_suffix('.h5')
    pickle_path = file_path.with_suffix('.pkl')

    if h5_path.exists() and HAS_H5PY:
        all_token_results = load_data_from_hdf5(h5_path, position_select)
    elif file_path.suffix == '.h5' and HAS_H5PY:
        all_token_results = load_data_from_hdf5(file_path, position_select)
    elif pickle_path.exists():
        all_token_results = load_data_from_pickle(pickle_path)
    else:
        raise FileNotFoundError(f"找不到数据文件: {file_path} 或 {h5_path}")
    
    # 解析位置选择
    available_positions = list(all_token_results.keys())
    numeric_positions = sorted([k for k in available_positions if isinstance(k, int)])
    
    print(f"可用位置: {available_positions}")
    print(f"数字位置: {numeric_positions}")
    if 'extra' in available_positions:
        print(f"Extra位置样本数: {len(all_token_results['extra']['labels'])}")
    
    # 确定要使用的位置
    if position_select == 'all':
        selected_positions = available_positions
    elif position_select == 'all_no_extra':
        selected_positions = numeric_positions
    elif position_select == 'extra':
        selected_positions = ['extra'] if 'extra' in available_positions else []
    elif isinstance(position_select, int):
        selected_positions = [position_select] if position_select in available_positions else []
    elif isinstance(position_select, list):
        selected_positions = [p for p in position_select if p in available_positions]
    else:
        raise ValueError(f"不支持的 position_select 值: {position_select}")
    
    if not selected_positions:
        raise ValueError(f"没有找到符合条件的位置: {position_select}")
    
    print(f"选择的位置: {selected_positions}")
    
    # 收集数据
    flows_all = []
    labels_all = []
    tokens_all = []
    position_indices = []
    pos_to_idx = {pos: i for i, pos in enumerate(selected_positions)}
    
    for pos in selected_positions:
        pos_data = all_token_results[pos]
        flows = pos_data['flows']
        labels = pos_data['labels']
        preds = pos_data.get('preds', ['' for _ in labels])  # 如果没有 preds，用空字符串
        
        
        print(f"  位置 {pos}: {len(labels)} 个样本, 正样本 {sum(labels)}, 负样本 {len(labels) - sum(labels)}")
        
        for flow, label, pred in zip(flows, labels, preds):
            flows_all.append(flow)
            labels_all.append(0 if label else 1) # 注意：原label为是否正确；此处label为是否为幻觉
            tokens_all.append(pred)
            position_indices.append(pos_to_idx[pos])
    
    # 获取维度信息
    sample_flow = flows_all[0]
    num_layers = sample_flow.shape[0]
    hidden_dim = sample_flow.shape[1]
    
    print(f"\n数据加载完成:")
    print(f"  总样本数: {len(flows_all)}, 正样本 {sum(labels_all)}, 负样本 {len(labels_all) - sum(labels_all)}")
    print(f"  Flow 形状: ({num_layers}, {hidden_dim})")
    
    return flows_all, labels_all, tokens_all, position_indices, selected_positions, num_layers, hidden_dim


def balance_dataset(flows, labels, tokens, position_indices, selected_positions, by_position=False):
    """
    平衡数据集，使两个类别数量相等
    
    Args:
        by_position: True - 对每个位置分别平衡, False - 所有位置混合平衡
    """
    labels_array = np.array(labels)
    pos_array = np.array(position_indices)
    
    if not by_position:
        # 原始方法：所有位置混合平衡
        unique, counts = np.unique(labels_array, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        print(f"\n原始数据类别分布（所有位置混合）:")
        for cls, count in sorted(class_counts.items()):
            print(f"  类别 {cls}: {count} 个样本")
        
        min_count = min(class_counts.values())
        
        flows_balanced = []
        labels_balanced = []
        tokens_balanced = []
        pos_balanced = []
        
        for cls in sorted(class_counts.keys()):
            cls_indices = np.where(labels_array == cls)[0]
            if len(cls_indices) > min_count:
                np.random.seed(SEED)
                selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
            else:
                selected_indices = cls_indices
            
            for idx in selected_indices:
                flows_balanced.append(flows[idx])
                labels_balanced.append(labels[idx])
                tokens_balanced.append(tokens[idx])
                pos_balanced.append(position_indices[idx])
            
            print(f"  类别 {cls}: 保留 {len(selected_indices)} 个")
        
        # 打乱顺序
        indices = np.random.permutation(len(flows_balanced))
        flows_balanced = [flows_balanced[i] for i in indices]
        labels_balanced = [labels_balanced[i] for i in indices]
        tokens_balanced = [tokens_balanced[i] for i in indices]
        pos_balanced = [pos_balanced[i] for i in indices]
        
        print(f"\n数据平衡完成: 总样本数 {len(flows_balanced)}")
        
    else:
        # 新方法：对每个位置分别平衡
        print(f"\n按位置平衡数据集:")
        flows_balanced = []
        labels_balanced = []
        tokens_balanced = []
        pos_balanced = []
        
        for pos_idx, pos in enumerate(selected_positions):
            # 获取当前位置的所有样本
            pos_mask = pos_array == pos_idx
            pos_flows = [flows[i] for i in range(len(flows)) if pos_mask[i]]
            pos_labels = labels_array[pos_mask]
            
            if len(pos_flows) == 0:
                continue
            
            # 统计该位置的类别分布
            unique, counts = np.unique(pos_labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            print(f"  位置 {pos}: ", end="")
            for cls, count in sorted(class_counts.items()):
                print(f"类别{cls}={count} ", end="")
            
            # 如果只有一个类别，全部保留
            if len(class_counts) == 1:
                print(f"→ 只有一个类别，全部保留 {len(pos_flows)} 个")
                flows_balanced.extend(pos_flows)
                labels_balanced.extend(pos_labels)
                pos_tokens = [tokens[j] for j in range(len(tokens)) if pos_mask[j]]
                tokens_balanced.extend(pos_tokens)
                pos_balanced.extend([pos_idx] * len(pos_flows))
                continue
            
            # 对该位置进行平衡
            min_count = min(class_counts.values())
            pos_indices = np.where(pos_mask)[0]
            
            for cls in sorted(class_counts.keys()):
                cls_local_indices = np.where(pos_labels == cls)[0]
                cls_global_indices = pos_indices[cls_local_indices]
                
                if len(cls_global_indices) > min_count:
                    np.random.seed(SEED + pos_idx)  # 不同位置使用不同种子
                    selected_global = np.random.choice(cls_global_indices, size=min_count, replace=False)
                else:
                    selected_global = cls_global_indices
                
                for idx in selected_global:
                    flows_balanced.append(flows[idx])
                    labels_balanced.append(labels[idx])
                    tokens_balanced.append(tokens[idx])
                    pos_balanced.append(pos_idx)
            
            print(f"→ 平衡后 {min_count * len(class_counts)} 个")
        
        # 打乱顺序
        indices = np.random.permutation(len(flows_balanced))
        flows_balanced = [flows_balanced[i] for i in indices]
        labels_balanced = [labels_balanced[i] for i in indices]
        tokens_balanced = [tokens_balanced[i] for i in indices]
        pos_balanced = [pos_balanced[i] for i in indices]
        
        print(f"\n数据平衡完成: 总样本数 {len(flows_balanced)}")
    
    return flows_balanced, labels_balanced, tokens_balanced, pos_balanced


# ==========================================
# ========== LM Head 加载 ==========
# ==========================================

def load_lm_head(model_path, device='cuda'):
    """
    加载 Qwen3-8B 的 lm_head
    
    Returns:
        lm_head: nn.Linear
        vocab_size: int
        hidden_dim: int
    """
    print(f"\n正在加载 lm_head from {model_path}...")
    
    # 加载完整模型（只需要 lm_head）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"  # 先加载到 CPU
    )
    
    lm_head = model.lm_head
    vocab_size = lm_head.out_features
    hidden_dim = lm_head.in_features
    
    # 移到指定设备
    lm_head = lm_head.to(device)
    lm_head.eval()  # 设置为评估模式
    
    print(f"lm_head 加载完成:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  device: {device}")
    
    return lm_head, vocab_size, hidden_dim



def load_tokenizer(model_path):
    """
    加载 tokenizer
    
    Returns:
        tokenizer: AutoTokenizer
    """
    print(f"\n正在加载 tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"tokenizer 加载完成")
    return tokenizer


# ==========================================
# ========== 不确定性计算 ==========
# ==========================================




def compute_uncertainty_scores(flows, lm_head, metric_type='entropy', layer_idx=-1, batch_size=256, device='cuda', tokens=None, tokenizer=None):
    """
    计算所有样本的不确定性分数

    Args:
        flows: List[np.ndarray], 每个元素形状 (num_layers, hidden_dim)
        lm_head: nn.Linear
        metric_type: 'entropy', 'perplexity', 'lnpe', 'eubhd', 'max_prob', 'top5_entropy', 'all'
        layer_idx: 使用哪一层的隐藏状态（-1 表示最后一层）
        batch_size: 批处理大小
        device: 计算设备
        tokens: List[str], 真实的 token 字符串（用于 token-level 指标）
        tokenizer: AutoTokenizer（用于 token-level 指标）

    Returns:
        scores: np.ndarray or dict, 
               - 单指标时: 形状 (num_samples,)
               - metric_type='all' 时: dict，键为指标名，值为 (num_samples,) 的分数数组
    """
    # 特殊处理 'all' 模式：计算所有指标并返回字典
    if metric_type == 'all':
        print(f"计算所有不确定性指标 (layer={layer_idx})...")
        all_metrics = ['entropy', 'perplexity', 'lnpe', 'eubhd', 'max_prob', 'top5_entropy', 'self_certainty']
        scores_dict = {}
        for metric in all_metrics:
            print(f"  -> 计算 {metric}...")
            scores_dict[metric] = compute_uncertainty_scores(
                flows, lm_head, metric, layer_idx, batch_size, device, tokens, tokenizer
            )
        print(f"所有指标计算完成！")
        return scores_dict
    
    print(f"计算不确定性分数 (metric={metric_type}, layer={layer_idx})...")

    all_scores = []
    num_samples = len(flows)

    # 批处理
    for i in tqdm(range(0, num_samples, batch_size), desc="计算进度"):
        batch_flows = flows[i:i + batch_size]

        # 堆叠当前 batch 的所有层隐藏状态，既可取指定层也可用于跨层统计
        batch_flows_tensor = torch.tensor(
            np.stack(batch_flows),
            dtype=torch.float16
        ).to(device)

        # 提取指定层的隐藏状态
        batch_hidden = batch_flows_tensor[:, layer_idx]

        with torch.no_grad():
            # 通过 lm_head 得到 logits（转为 float32 提高数值稳定性）
            batch_logits = lm_head(batch_hidden).float()  # (batch, vocab_size)

            # 根据不同指标计算分数
            if metric_type == "entropy":
                # 香农熵: H = -Σ(p * log(p))
                log_probs = F.log_softmax(batch_logits, dim=-1)
                probs = torch.exp(log_probs)
                batch_scores = -(probs * log_probs).sum(dim=-1)

            elif metric_type == "perplexity":
                # 真正的 Token-level Perplexity: PPL = exp(-log P(token_true))
                if tokens is None or tokenizer is None:
                    raise ValueError("perplexity 需要 tokens 和 tokenizer 参数")

                log_probs = F.log_softmax(batch_logits, dim=-1)  # (batch, vocab_size)

                # 获取当前 batch 的真实 token IDs
                batch_tokens = tokens[i:i + len(batch_flows)]
                batch_token_ids = []
                for token_str in batch_tokens:
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    batch_token_ids.append(token_ids[0] if token_ids else 0)

                token_ids_tensor = torch.tensor(batch_token_ids, dtype=torch.long, device=device)

                # 提取真实 token 的对数概率
                true_log_probs = log_probs.gather(1, token_ids_tensor.unsqueeze(1)).squeeze(1)

                # PPL = exp(-log_prob)
                batch_scores = torch.exp(-true_log_probs)

                # 限制范围避免溢出
                batch_scores = torch.clamp(batch_scores, max=1e6)

            elif metric_type == "lnpe":
                log_probs = F.log_softmax(batch_logits, dim=-1)
                probs = torch.exp(log_probs)
                batch_scores = -(probs * log_probs).sum(dim=-1)

            elif metric_type == "eubhd":
                # EUBHD = (1/T) * Σ_t [H_t + λ * Var_l(h_t^l)]，单 token 场景
                log_probs = F.log_softmax(batch_logits, dim=-1)
                probs = torch.exp(log_probs)
                entropy = -(probs * log_probs).sum(dim=-1)

                # 跨层方差：层维度求方差，隐藏维度取均值
                hidden_var = batch_flows_tensor.float().var(dim=1, unbiased=False).mean(dim=1)
                lambda_coef = 0.2
                batch_scores = entropy + lambda_coef * hidden_var

            elif metric_type == "max_prob":
                # 最大概率（取负，使得高不确定性 = 高分数）
                probs = F.softmax(batch_logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                batch_scores = -max_probs

            elif metric_type == "top5_entropy":
                # 只对 top-5 概率计算熵
                log_probs = F.log_softmax(batch_logits, dim=-1)
                probs = torch.exp(log_probs)
                top5_probs, _ = torch.topk(probs, k=5, dim=-1)
                top5_probs = top5_probs / (top5_probs.sum(dim=-1, keepdim=True) + 1e-10)  # 重新归一化
                top5_log_probs = torch.log(top5_probs + 1e-10)
                batch_scores = -(top5_probs * top5_log_probs).sum(dim=-1)

            elif metric_type == "self_certainty":
                # Self-Certainty: C_i = log V - (1/V) * sum(log p_j)
                # 幻觉检测取负: -C_i = (1/V) * sum(log p_j) - log V
                log_probs = F.log_softmax(batch_logits, dim=-1)
                V = batch_logits.size(-1)
                batch_scores = log_probs.mean(dim=-1) - np.log(V)

            else:
                raise ValueError(f"不支持的 metric_type: {metric_type}")

            # 转换为 numpy 并检查 NaN
            batch_scores_np = batch_scores.cpu().numpy()

            # 检测并报告 NaN
            nan_mask = np.isnan(batch_scores_np)
            if nan_mask.any():
                num_nan = nan_mask.sum()
                print(f"  警告: 检测到 {num_nan} 个 NaN 值，已替换为最大有效值")
                # 用最大有效值替换 NaN
                valid_scores = batch_scores_np[~nan_mask]
                if len(valid_scores) > 0:
                    replacement = valid_scores.max()
                else:
                    replacement = 0.0
                batch_scores_np[nan_mask] = replacement

            all_scores.append(batch_scores_np)

    scores = np.concatenate(all_scores)

    print(f"不确定性分数统计:")
    print(f"  最小值: {scores.min():.4f}")
    print(f"  最大值: {scores.max():.4f}")
    print(f"  平均值: {scores.mean():.4f}")
    print(f"  标准差: {scores.std():.4f}")

    return scores

# ==========================================
# ========== 阈值搜索与评估 ==========
# ==========================================

def find_best_threshold(scores, labels, num_steps=100):
    """
    搜索最佳阈值（使 ACC 最大）
    
    Args:
        scores: np.ndarray, 不确定性分数
        labels: np.ndarray, 标签 (0 或 1)
        num_steps: 搜索步数
    
    Returns:
        best_threshold: float
        best_auc: float
    """
    print(f"\n搜索最佳阈值（搜索步数={num_steps}）...")
    
    # 使用 ROC 曲线找最佳阈值
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # Youden's J statistic: J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # 计算该阈值下的 AUC
    best_auc = roc_auc_score(labels, scores)
    
    print(f"最佳阈值: {best_threshold:.4f}")
    print(f"AUC: {best_auc:.4f}")
    print(f"TPR: {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")
    
    return best_threshold, best_auc


def evaluate_with_threshold(scores, labels, threshold):
    """
    使用给定阈值评估
    
    Args:
        scores: np.ndarray, 不确定性分数
        labels: np.ndarray, 标签 (0 或 1)
        threshold: float, 阈值
    
    Returns:
        metrics: dict, 包含 auc, acc, f1
    """
    # 预测: 分数 > 阈值 → 预测为类别 1（错误）
    preds = (scores > threshold).astype(int)
    
    auc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label=1)
    
    return {
        'auc': auc,
        'acc': acc,
        'f1': f1
    }


# ==========================================
# ========== 主程序 ==========
# ==========================================

def main():
    setup_logger()
    
    print("=" * 60)
    print("Baseline Classifier (基于 LM Head 的不确定性度量)")
    print("=" * 60)
    print(f"配置参数:")
    print(f"  数据文件: {DATA_FILE_PATH}")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  位置选择: {POSITION_SELECT}")
    print(f"  指标类型: {METRIC_TYPE}")
    print(f"  阈值模式: {THRESHOLD_MODE}")
    print(f"  数据平衡: {BALANCE_DATASET}")
    print(f"  按位置平衡: {BALANCE_BY_POSITION}")
    print(f"  按位置评估: {EVALUATE_BY_POSITION}")
    print(f"  按层评估: {EVALUATE_EACH_LAYER}")
    print("=" * 60)
    
    # 1. 加载数据
    flows, labels, tokens, position_indices, selected_positions, num_layers, hidden_dim = load_flows_and_labels(
        DATA_FILE_PATH,
        position_select=POSITION_SELECT
    )
    
    # 2. 数据平衡
    if BALANCE_DATASET:
        flows, labels, tokens, position_indices = balance_dataset(flows, labels, tokens, position_indices, selected_positions, BALANCE_BY_POSITION)
    
    # 转为 numpy 数组
    labels = np.array(labels)
    position_indices = np.array(position_indices)
    
    # 3. 数据划分
    indices = np.arange(len(flows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels
    )
    
    flows_train = [flows[i] for i in train_idx]
    flows_val = [flows[i] for i in val_idx]
    tokens_train = [tokens[i] for i in train_idx]
    tokens_val = [tokens[i] for i in val_idx]
    labels_train = labels[train_idx]
    labels_val = labels[val_idx]
    pos_train = position_indices[train_idx]
    pos_val = position_indices[val_idx]
    
    print(f"\n训练集: {len(flows_train)} 样本")
    print(f"验证集: {len(flows_val)} 样本")
    
    # 4. 加载 lm_head 和 tokenizer
    lm_head, vocab_size, lm_hidden_dim = load_lm_head(MODEL_PATH, device=DEVICE)
    tokenizer = load_tokenizer(MODEL_PATH)
    
    # 检查维度匹配
    if lm_hidden_dim != hidden_dim:
        print(f"\n警告: lm_head 的 hidden_dim ({lm_hidden_dim}) 与 flow 的 hidden_dim ({hidden_dim}) 不匹配!")
        print(f"请检查数据文件和模型是否对应。")
        return
    
    if not EVALUATE_EACH_LAYER:
        # 常规评估：使用最后一层
        layer_idx = SPECIFIC_LAYER_INDEX if SPECIFIC_LAYER_INDEX is not None else -1
        
        # 5. 计算训练集不确定性分数
        scores_train = compute_uncertainty_scores(
            flows_train, lm_head, METRIC_TYPE, layer_idx, BATCH_SIZE, DEVICE,
            tokens=tokens_train, tokenizer=tokenizer
        )
        
        # 处理 'all' 模式：对每个指标分别评估
        if METRIC_TYPE == 'all':
            print("" + "=" * 60)
            print("多指标评估模式：所有指标结果")
            print("=" * 60)
            
            # 计算验证集的所有指标
            scores_val = compute_uncertainty_scores(
                flows_val, lm_head, METRIC_TYPE, layer_idx, BATCH_SIZE, DEVICE,
                tokens=tokens_val, tokenizer=tokenizer
            )
            
            # 对每个指标分别评估
            all_results = {}
            for metric_name in scores_train.keys():
                print(f"--- 指标: {metric_name} ---")
                
                # 搜索阈值
                if THRESHOLD_MODE == 'auto':
                    best_threshold, _ = find_best_threshold(
                        scores_train[metric_name], labels_train, THRESHOLD_SEARCH_STEPS
                    )
                else:
                    best_threshold = MANUAL_THRESHOLD
                
                # 评估
                val_metrics = evaluate_with_threshold(scores_val[metric_name], labels_val, best_threshold)
                all_results[metric_name] = val_metrics
                
                print(f"  AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            # 汇总：按 AUC 排序
            print("" + "=" * 60)
            print("多指标评估汇总（按 AUC 降序）:")
            print("=" * 60)
            sorted_metrics = sorted(all_results.items(), key=lambda x: x[1]['auc'], reverse=True)
            for metric_name, metrics in sorted_metrics:
                print(f"  {metric_name:>12}: AUC={metrics['auc']:.4f} | Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f}")
            
            best_metric = sorted_metrics[0]
            print(f"最佳指标: {best_metric[0]}，AUC: {best_metric[1]['auc']:.4f}")
            
        else:
            # 单指标模式
            # 6. 搜索或使用手动阈值
            if THRESHOLD_MODE == 'auto':
                best_threshold, train_auc = find_best_threshold(
                    scores_train, labels_train, THRESHOLD_SEARCH_STEPS
                )
            else:
                best_threshold = MANUAL_THRESHOLD
                train_metrics = evaluate_with_threshold(scores_train, labels_train, best_threshold)
                print(f"使用手动阈值: {best_threshold}")
                print(f"训练集 AUC: {train_metrics['auc']:.4f}")
            
            # 7. 计算验证集不确定性分数
            scores_val = compute_uncertainty_scores(
                flows_val, lm_head, METRIC_TYPE, layer_idx, BATCH_SIZE, DEVICE,
                tokens=tokens_val, tokenizer=tokenizer
            )
            
            # 8. 在验证集上评估
            val_metrics = evaluate_with_threshold(scores_val, labels_val, best_threshold)
            
            print("" + "=" * 60)
            print("验证集评估结果:")
            print(f"  AUC:  {val_metrics['auc']:.4f}")
            print(f"  Acc:  {val_metrics['acc']:.4f}")
            print(f"  F1:   {val_metrics['f1']:.4f}")
            print("=" * 60)
        

        # 9. 按位置统计准确率和 AUC
        if EVALUATE_BY_POSITION:
            print("\n" + "=" * 60)
            print("按位置统计验证集指标:")
            print("=" * 60)
            
            preds_val = (scores_val > best_threshold).astype(int)
            
            for idx, pos in enumerate(selected_positions):
                # 获取该位置的样本
                pos_mask = (pos_val == idx)
                if pos_mask.sum() == 0:
                    print(f"位置 {pos}: 验证样本数为 0")
                    continue
                
                pos_labels = labels_val[pos_mask]
                pos_scores = scores_val[pos_mask]
                pos_preds = preds_val[pos_mask]
                
                # 计算指标
                pos_acc = accuracy_score(pos_labels, pos_preds)
                
                # 计算 AUC（需要至少两个类别）
                if len(np.unique(pos_labels)) > 1:
                    pos_auc = roc_auc_score(pos_labels, pos_scores)
                    pos_f1 = f1_score(pos_labels, pos_preds, pos_label=0)
                    print(f"位置 {pos:>2}: Acc={pos_acc:.4f} | AUC={pos_auc:.4f} | F1={pos_f1:.4f} | 样本数={pos_mask.sum()}")
                else:
                    print(f"位置 {pos:>2}: Acc={pos_acc:.4f} | AUC=N/A (只有一个类别) | 样本数={pos_mask.sum()}")
        else:
            print("\n按位置统计验证准确率:")
            preds_val = (scores_val > best_threshold).astype(int)
            
            per_pos_correct = {i: 0 for i in range(len(selected_positions))}
            per_pos_total = {i: 0 for i in range(len(selected_positions))}
            
            for i, (pred, label, pos_idx) in enumerate(zip(preds_val, labels_val, pos_val)):
                per_pos_total[pos_idx] += 1
                if pred == label:
                    per_pos_correct[pos_idx] += 1
            
            for idx, pos in enumerate(selected_positions):
                total = per_pos_total[idx]
                if total == 0:
                    print(f"  位置 {pos}: 验证样本数为 0")
                else:
                    acc = per_pos_correct[idx] / total
                    print(f"  位置 {pos}: 准确率 {acc:.4f} (样本数 {total})")
        print("\n按位置统计验证准确率:")
        preds_val = (scores_val > best_threshold).astype(int)
        
        per_pos_correct = {i: 0 for i in range(len(selected_positions))}
        per_pos_total = {i: 0 for i in range(len(selected_positions))}
        
        for i, (pred, label, pos_idx) in enumerate(zip(preds_val, labels_val, pos_val)):
            per_pos_total[pos_idx] += 1
            if pred == label:
                per_pos_correct[pos_idx] += 1
        
        for idx, pos in enumerate(selected_positions):
            total = per_pos_total[idx]
            if total == 0:
                print(f"  位置 {pos}: 验证样本数为 0")
            else:
                acc = per_pos_correct[idx] / total
                print(f"  位置 {pos}: 准确率 {acc:.4f} (样本数 {total})")
    
    else:
        # 逐层评估模式
        print(f"\n{'='*60}")
        print("开启按层评估模式：每一层都会独立评估并报告 AUC。")
        print(f"{'='*60}\n")
        
        layer_results = []
        
        for layer_idx in range(num_layers):
            if SPECIFIC_LAYER_INDEX is not None and layer_idx != SPECIFIC_LAYER_INDEX:
                continue
            
            print(f"{'-'*40}")
            print(f"评估第 {layer_idx} 层")
            
            # 计算训练集分数
            scores_train = compute_uncertainty_scores(
            flows_train, lm_head, METRIC_TYPE, layer_idx, BATCH_SIZE, DEVICE,
            tokens=tokens_train, tokenizer=tokenizer
            )
            
            # 处理 'all' 模式
            if METRIC_TYPE == 'all':
                # 计算验证集的所有指标
                scores_val = compute_uncertainty_scores(
                    flows_val, lm_head, METRIC_TYPE, layer_idx, BATCH_SIZE, DEVICE,
                    tokens=tokens_val, tokenizer=tokenizer
                )
                
                # 对每个指标分别评估
                print(f"[Layer {layer_idx}] 多指标评估：")
                layer_metrics = {}
                for metric_name in scores_train.keys():
                    # 搜索阈值
                    if THRESHOLD_MODE == 'auto':
                        best_threshold, _ = find_best_threshold(
                            scores_train[metric_name], labels_train, THRESHOLD_SEARCH_STEPS
                        )
                    else:
                        best_threshold = MANUAL_THRESHOLD
                    
                    # 评估
                    val_metrics = evaluate_with_threshold(scores_val[metric_name], labels_val, best_threshold)
                    layer_metrics[metric_name] = val_metrics
                    print(f"  {metric_name:>12}: AUC={val_metrics['auc']:.4f} | Acc={val_metrics['acc']:.4f} | F1={val_metrics['f1']:.4f}")
                
                # 记录所有指标的结果
                for metric_name, metrics in layer_metrics.items():
                    layer_results.append({
                        'layer': layer_idx,
                        'metric': metric_name,
                        'auc': metrics['auc'],
                        'acc': metrics['acc'],
                        'f1': metrics['f1']
                    })
            else:
                # 单指标模式
                # 搜索阈值
                if THRESHOLD_MODE == 'auto':
                    best_threshold, _ = find_best_threshold(
                        scores_train, labels_train, THRESHOLD_SEARCH_STEPS
                    )
                else:
                    best_threshold = MANUAL_THRESHOLD
                
                # 计算验证集分数
                scores_val = compute_uncertainty_scores(
                flows_val, lm_head, METRIC_TYPE, layer_idx, BATCH_SIZE, DEVICE,
                tokens=tokens_val, tokenizer=tokenizer
                )
                
                # 评估
                val_metrics = evaluate_with_threshold(scores_val, labels_val, best_threshold)
                
                layer_results.append({
                    'layer': layer_idx,
                    'threshold': best_threshold,
                    'auc': val_metrics['auc'],
                    'acc': val_metrics['acc'],
                    'f1': val_metrics['f1']
                })
                
                print(f"[Layer {layer_idx}] Val AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 汇总结果
        print(f"{'='*60}")
        print("按层评估结果汇总（按 AUC 降序）：")
        layer_results_sorted = sorted(layer_results, key=lambda x: x['auc'], reverse=True)
        
        if METRIC_TYPE == 'all':
            # 多指标模式：显示指标名称
            for item in layer_results_sorted[:20]:  # 只显示前20个结果
                metric_name = item.get('metric', 'N/A')
                print(f"  层 {item['layer']:>2} | {metric_name:>12}: AUC={item['auc']:.4f}, Acc={item['acc']:.4f}, F1={item['f1']:.4f}")
            
            best_layer = layer_results_sorted[0]
            print(f"最佳组合: 层 {best_layer['layer']} + {best_layer.get('metric', 'N/A')}，最佳验证 AUC: {best_layer['auc']:.4f}")
        else:
            # 单指标模式
            for item in layer_results_sorted:
                print(f"  层 {item['layer']:>2}: AUC={item['auc']:.4f}, Acc={item['acc']:.4f}, F1={item['f1']:.4f}")
            
            best_layer = layer_results_sorted[0]
            print(f"最佳层: {best_layer['layer']}，最佳验证 AUC: {best_layer['auc']:.4f}")


if __name__ == "__main__":
    main()