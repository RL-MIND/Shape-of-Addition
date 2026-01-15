import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import logging
import os
import builtins
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA
from pathlib import Path
from datetime import datetime

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("提示: 安装 h5py 可以大幅加速大文件加载: pip install h5py")

# CUDA_VISIBLE_DEVICES=2 python classifier.py


# ==========================================
# ========== 配置参数区域 ==========
# ==========================================

# 日志配置
LOG_DIR = Path("VerticalFlow/log/log_classify")
LOGGER = None  # 运行时在 main 中初始化
ORIGINAL_PRINT = print

# --- 1. 数据配置 ---
# DATA_FILE_PATH = 'results/results-Qwen3-0p6B'
DATA_FILE_PATH = 'VerticalFlow/results/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b'
# DATA_FILE_PATH = 'VerticalFlow/results/mul_num2len5_Qwen3-4b/mul_num2len5_Qwen3-4b'
# DATA_FILE_PATH = 'results/mul_num3len3_Qwen3-4B-Instruct-2507'
# DATA_FILE_PATH = 'results/plus_num3len10_Qwen3-4B-Instruct-2507'
# DATA_FILE_PATH = 'results/plus_num5len20_Qwen3-4B-Instruct-2507'

BALANCE_DATASET = False           # 是否平衡数据集（使两个类别数量相等）
STRONG_BALANCE_BY_POSITION = True  # 是否按位置先平衡再合并（每个位置内类别均衡）
TEST_SIZE = 0.2                  # 划分验证集比例

# 选择使用哪些位置的数据进行训练
#   - 'all': 使用所有位置（包括extra）
#   - 'all_no_extra': 使用所有位置（不包括extra）
#   - [0, 1, 2, ...]: 指定位置列表
#   - 0, 1, 2, ...: 单个位置（整数）
#   - 'extra': 只使用extra位置
POSITION_SELECT = 3

# 选择使用哪种特征
FEATURE_TYPE = 'flows'  # 'flows', 'velocities', 'curvatures'

# Pooling配置（对seq_len维度进行池化降维）
POOLING_TYPE = 'None'    # None, 'avg', 'max'

# 按层评估开关
# 打开后会逐层训练/验证，每层单独跑一遍完整训练流程，取验证集 AUC 最高的 epoch 作为该层得分
EVALUATE_EACH_LAYER = True
SPECIFIC_LAYER_INDEX = None    # None, 0, 1, 2, ...

# 按位置评估开关
# 打开后会逐位置单独训练/验证，取该位置验证集 AUC 最高的 epoch 作为该位置得分
EVALUATE_EACH_POSITION = False

# PCA配置
USE_PCA = False                  # 是否使用PCA降维
PCA_DIM = 100                    # PCA降维后的维度

# --- 2. 训练超参数 ---
BATCH_SIZE = 256                 # 批次大小
EARLY_STOP_PATIENCE = 20         # 连续多少个 epoch 验证 AUC 未提升则停止
LEARNING_RATE = 1e-4             # 学习率
WEIGHT_DECAY = 1e-4              # L2正则化系数
SEED = 42                        # 随机种子
CIRCULAR_PROBE_EPOCHS = 300    # CircularProbe训练epoch数（不使用early stopping）

# --- 3. 模型选择 ---
# 可选: 'mlp', 'mlp10', 'transformer', 'logreg', 'ar_transformer', 'lstm', 'circular_probe', 'spiral_probe'
MODEL_TYPE = 'mlp10'

# --- 3.1 模型存储 ---
SAVE_MODEL = False               # 是否在训练结束后保存模型
SAVE_DIR = 'VerticalFlow/saved_models'        # 模型保存目录
SAVE_NAME = DATA_FILE_PATH.split('/')[-1]

# --- 4. MLP模型参数 ---
MLP_HIDDEN_DIM = 512             # MLP隐藏层维度
MLP_DROPOUT = 0.4                # MLP Dropout比率

# --- 5. Transformer模型参数 ---
TRANSFORMER_D_MODEL = 512        # Transformer模型维度
TRANSFORMER_NHEAD = 2            # 多头注意力头数
TRANSFORMER_NUM_LAYERS = 2       # Transformer编码器层数
TRANSFORMER_DIM_FEEDFORWARD = 1024  # 前馈网络隐藏层维度
TRANSFORMER_DROPOUT = 0.1        # Transformer Dropout比率

# (已删除 CNN 和 SVM 相关配置)

# --- 8. Autoregressive Transformer 参数 ---
AR_TRANSFORMER_D_MODEL = 256        # 模型维度
AR_TRANSFORMER_NHEAD = 8            # 注意力头数
AR_TRANSFORMER_NUM_LAYERS = 2       # 层数
AR_TRANSFORMER_DIM_FEEDFORWARD = 1024 # 前馈网络维度
AR_TRANSFORMER_DROPOUT = 0.1        # Dropout

# --- 9. LSTM 参数 ---
LSTM_HIDDEN_DIM = 256               # 隐藏层维度
LSTM_NUM_LAYERS = 2                 # 层数
LSTM_DROPOUT = 0.1                  # Dropout
LSTM_BIDIRECTIONAL = True          # 是否双向

# --- 10. CircularProbe 参数 ---
CIRCULAR_PROBE_NUM_CLASSES = 10     # 数字分类数量（0-9）

# --- 11. SpiralProbe 参数 ---
SPIRAL_PROBE_NUM_CLASSES = 10       # 数字分类数量（0-9）
SPIRAL_PROBE_N_HARMONICS = 4        # 谐波数量 k，螺旋基维度为 1 + 2k
SPIRAL_PROBE_BASE_PERIOD = 10.0     # 基础周期（数字0-9的周期）
SPIRAL_PROBE_HELIX_LOSS_WEIGHT = 0.1  # 螺旋正则化损失权重 λ
SPIRAL_PROBE_EPOCHS = 300           # SpiralProbe 训练 epoch 数
SPIRAL_PROBE_HIDDEN_DIM = 512       # MLP编码器隐藏层维度
SPIRAL_PROBE_DROPOUT = 0.3          # Dropout比率

# ==========================================
# ========== 设备与随机种子设置 ==========
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# ========== 日志工具 ==========
# ==========================================

def setup_logger():
    """
    初始化全局 LOGGER，日志输出到 log_classify 目录，同时保留控制台输出。
    """
    # 当由 parallel_runner 调用时，直接使用 stdout/stderr，不创建单独日志文件
    if os.getenv("PARALLEL_RUNNER") == "1":
        return None
    global LOGGER
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("classifier")
    logger.setLevel(logging.INFO)
    
    # 清理旧的 handler，避免重复添加
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{MODEL_TYPE}.log"
    
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
    """
    兼容原 print，额外将信息写入 LOGGER（如果已初始化）。
    """
    message = " ".join(str(a) for a in args)
    if LOGGER:
        LOGGER.info(message)
        return  # 避免重复打印到控制台
    return ORIGINAL_PRINT(*args, **kwargs)


# 覆盖全局 print，使后续 print 自动写入日志
print = log_print


# ==========================================
# ========== 数据加载与预处理 ==========
# ==========================================

def load_data_from_pickle(file_path):
    """从 pickle 文件加载数据（慢）"""
    print(f"正在从 pickle 加载数据（较慢）: {file_path}")
    print("提示: 运行 python convert_to_hdf5.py 转换为 HDF5 格式可以大幅加速!")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['all_token_results']


def load_data_from_hdf5(file_path, position_select='all'):
    """
    从 HDF5 文件加载数据（快速，支持部分加载）
    
    只读取 flows；velocities、curvatures 在内存中按需计算。
    """
    print(f"正在从 HDF5 快速加载数据: {file_path}")
    
    all_token_results = {}
    
    with h5py.File(file_path, 'r') as hf:
        positions_group = hf['all_token_results']
        # 获取可用位置，优先使用 attrs，若缺失则从 group 名推断
        numeric_positions = list(positions_group.attrs.get('numeric_positions', []))
        string_positions = list(positions_group.attrs.get('string_positions', []))
        # 转成纯 Python 类型，避免 numpy.int64 参与字符串拼接时报错
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
            # 加载全部，让后面的代码处理
            positions_to_load = numeric_positions + string_positions
        
        print(f"  HDF5 中的位置: 数字={numeric_positions}, 字符串={string_positions}")
        print(f"  将加载的位置: {positions_to_load}")
        
        for pos in positions_to_load:
            # 兼容 numpy 类型
            if isinstance(pos, np.integer):
                pos = int(pos)
            pos_name = f"pos_{pos}" if isinstance(pos, int) else str(pos)
            
            if pos_name not in positions_group:
                continue
                
            pos_group = positions_group[pos_name]
            pos_data = {}
            
            # 加载 flows（其他特征在内存计算）
            if 'flows' in pos_group:
                flows_array = pos_group['flows'][:]
                pos_data['flows'] = [flows_array[i] for i in range(len(flows_array))]
            
            # 加载标签
            if 'labels' in pos_group:
                pos_data['labels'] = list(pos_group['labels'][:])
            
            # 加载gt_chars和preds（用于CircularProbe）
            if 'gt_chars' in pos_group:
                pos_data['gt_chars'] = list(pos_group['gt_chars'][:].astype(str))
            if 'preds' in pos_group:
                pos_data['preds'] = list(pos_group['preds'][:].astype(str))
            
            all_token_results[pos] = pos_data
            print(f"    位置 {pos}: 加载完成")
    
    return all_token_results


def compute_feature_from_flow(flow, feature_type):
    """
    根据需要的特征类型从 flow 计算对应特征。
    flow: numpy 数组，形状 (seq_len, feature_dim)
    """
    if feature_type == 'flows':
        return flow
    if feature_type == 'velocities':
        return np.diff(flow, axis=0)
    if feature_type == 'curvatures':
        return np.diff(np.diff(flow, axis=0), axis=0)
    raise ValueError(f"不支持的特征类型: {feature_type}")


def load_and_process_data(file_path, position_select='all', feature_type='flows', 
                          pooling_type=None, use_pca=False, pca_dim=512):
    """
    加载并预处理 generate.py 输出的数据
    
    自动检测文件格式（.pkl 或 .h5），优先使用 HDF5 格式
    
    Args:
        file_path: 数据文件路径（支持 .pkl 和 .h5 格式）
        position_select: 选择使用哪些位置
            - 'all': 使用所有位置（包括extra）
            - 'all_no_extra': 使用所有位置（不包括extra）
            - [0, 1, 2]: 指定位置列表
            - 0, 1, 2: 单个位置（整数）
            - 'extra': 只使用extra位置
        feature_type: 特征类型 ('flows', 'velocities', 'curvatures')
        pooling_type: 池化类型 (None, 'avg', 'max')
            - None: 不使用pooling
            - 'avg': Average Pooling
            - 'max': Max Pooling
        use_pca: 是否使用PCA降维
        pca_dim: PCA降维后的维度
    
    Returns:
        X_all: 特征数据 (Tensor)
        y_all: 标签数据 (Tensor)
        position_indices: 每个样本对应的位置索引 (Tensor)
        selected_positions: 实际使用的位置列表（用于打印）
        seq_len: 序列长度
        feature_dim: 特征维度
    """
    file_path = Path(file_path)
    # 自动检测并使用最佳格式
    h5_path = file_path.with_suffix('.h5')
    pickle_path = file_path.with_suffix('.pkl')

    if h5_path.exists() and HAS_H5PY:
        # 优先使用 HDF5 格式（快速）
        all_token_results = load_data_from_hdf5(h5_path, position_select)
    elif file_path.suffix == '.h5' and HAS_H5PY:
        all_token_results = load_data_from_hdf5(file_path, position_select)
    elif pickle_path.exists():
        # 回退到 pickle 格式（慢）
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
    X_list = []
    y_list = []
    position_idx_list = []  # 记录每个样本来自哪个位置（索引）
    gt_chars_list = []  # 收集gt_chars
    preds_list = []     # 收集preds
    pos_to_idx = {pos: i for i, pos in enumerate(selected_positions)}
    
    for pos in selected_positions:
        pos_data = all_token_results[pos]
        flows = pos_data['flows']  # list of numpy arrays
        labels = pos_data['labels']  # list of bool
        
        gt_chars = pos_data.get('gt_chars', [None] * len(labels))  # 获取gt_chars
        preds = pos_data.get('preds', [None] * len(labels))       # 获取preds
        print(f"  位置 {pos}: {len(labels)} 个样本, 正样本 {sum(labels)}, 负样本 {len(labels) - sum(labels)}")
        
        for flow, label, gt_char, pred in zip(flows, labels, gt_chars, preds):
            feat = compute_feature_from_flow(flow, feature_type)
            # feat shape: (seq_len, feature_dim) 或衍生后的 (seq_len-1, feature_dim) / (seq_len-2, feature_dim)
            
            # 应用 pooling
            if pooling_type == 'avg':
                # Average Pooling: 对 seq_len 维度取平均
                feat_processed = feat.mean(axis=0)  # shape: (feature_dim,)
            elif pooling_type == 'max':
                # Max Pooling: 对 seq_len 维度取最大值
                feat_processed = feat.max(axis=0)  # shape: (feature_dim,)
            else:
                # 不使用 pooling，展平
                feat_processed = feat.flatten()  # shape: (seq_len * feature_dim,)
            
            X_list.append(feat_processed)
            y_list.append(0 if label else 1) #将原标签反转，测试幻觉率
            position_idx_list.append(pos_to_idx[pos])
            gt_chars_list.append(int(gt_char) if gt_char is not None and gt_char.isdigit() else -1)
            preds_list.append(int(pred) if pred is not None and pred.isdigit() else -1)
    
    # 转换为 Tensor
    X_all = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y_all = torch.tensor(y_list, dtype=torch.long)
    position_indices = torch.tensor(position_idx_list, dtype=torch.long)
    gt_chars_all = torch.tensor(gt_chars_list, dtype=torch.long)
    preds_all = torch.tensor(preds_list, dtype=torch.long)
    
    # 推断序列长度和特征维度（基于首个 flow 推断衍生特征形状）
    sample_flow = all_token_results[selected_positions[0]]['flows'][0]
    sample_feat = compute_feature_from_flow(sample_flow, feature_type)
    original_seq_len = sample_feat.shape[0]
    feature_dim = sample_feat.shape[1]
    
    # 根据 pooling 设置实际的 seq_len
    if pooling_type in ['avg', 'max']:
        seq_len = 1  # pooling 后 seq_len 变为 1
    else:
        seq_len = original_seq_len
    
    print(f"\n数据加载完成:")
    print(f"  总样本数: {len(X_all)}，正样本数: {sum(y_all)}，负样本数: {len(y_all) - sum(y_all)}")
    print(f"  X shape: {X_all.shape}")
    print(f"  y shape: {y_all.shape}")
    print(f"  原始序列长度: {original_seq_len}")
    print(f"  Pooling类型: {pooling_type}")
    print(f"  处理后序列长度 (seq_len): {seq_len}")
    print(f"  特征维度 (feature_dim): {feature_dim}")
    print(f"  特征类型: {feature_type}")
    
    # PCA降维处理
    if use_pca:
        print(f"\n应用PCA降维到 {pca_dim} 维...")
        X_numpy = X_all.numpy()
        pca = PCA(n_components=pca_dim, random_state=SEED)
        X_reduced = pca.fit_transform(X_numpy)
        X_all = torch.tensor(X_reduced, dtype=torch.float32)
        explained_variance_ratio = pca.explained_variance_ratio_.sum()
        print(f"PCA完成. 新 X shape: {X_all.shape}, 保留方差比例: {explained_variance_ratio:.4f}")
        # PCA后需要更新维度信息
        seq_len = 1
        feature_dim = pca_dim
    
    return X_all, y_all, position_indices, selected_positions, seq_len, feature_dim, gt_chars_all, preds_all


def balance_dataset(X, y, position_indices):
    """
    平衡数据集，使两个类别的数量相等，并保持位置索引同步
    """
    y_numpy = y.numpy()
    unique, counts = np.unique(y_numpy, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print(f"\n原始数据类别分布:")
    for cls, count in sorted(class_counts.items()):
        print(f"  类别 {cls}: {count} 个样本")
    
    min_count = min(class_counts.values())
    
    X_balanced_list = []
    y_balanced_list = []
    pos_balanced_list = []
    total_kept = 0
    total_dropped = 0
    
    for cls in sorted(class_counts.keys()):
        cls_indices = np.where(y_numpy == cls)[0]
        cls_count = len(cls_indices)
        
        if cls_count > min_count:
            np.random.seed(SEED)
            selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
            dropped_count = cls_count - min_count
            total_dropped += dropped_count
        else:
            selected_indices = cls_indices
            dropped_count = 0
        
        X_balanced_list.append(X[selected_indices])
        y_balanced_list.append(y[selected_indices])
        pos_balanced_list.append(position_indices[selected_indices])
        total_kept += len(selected_indices)
        
        print(f"  类别 {cls}: 保留 {len(selected_indices)} 个，舍弃 {dropped_count} 个")
    
    X_balanced = torch.cat(X_balanced_list, dim=0)
    y_balanced = torch.cat(y_balanced_list, dim=0)
    pos_balanced = torch.cat(pos_balanced_list, dim=0)
    
    # 打乱数据顺序
    indices = torch.randperm(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    pos_balanced = pos_balanced[indices]
    
    print(f"\n数据平衡完成:")
    print(f"  保留样本数: {total_kept}")
    print(f"  舍弃样本数: {total_dropped}")
    print(f"  平衡后数据形状: X {X_balanced.shape}, y {y_balanced.shape}\n")
    
    return X_balanced, y_balanced, pos_balanced


def balance_dataset_per_position(X, y, position_indices, position_names=None):
    """
    先在每个位置内分别平衡正负样本，再合并。
    对于只有单一类别的位置，保持原样并给出提示。
    """
    np.random.seed(SEED)
    unique_positions = torch.unique(position_indices).tolist()
    
    X_parts = []
    y_parts = []
    pos_parts = []
    total_kept = 0
    total_dropped = 0
    
    print("\n按位置强平衡: 每个位置内先平衡，再合并")
    for pos_idx in unique_positions:
        mask = position_indices == pos_idx
        X_pos = X[mask]
        y_pos = y[mask]
        pos_pos = position_indices[mask]
        pos_label = (
            position_names[pos_idx]
            if position_names is not None and pos_idx < len(position_names)
            else pos_idx
        )
        
        y_numpy = y_pos.numpy()
        if len(y_numpy) == 0:
            print(f"  位置 {pos_label}: 无样本，跳过")
            continue
        
        unique, counts = np.unique(y_numpy, return_counts=True)
        if len(unique) < 2:
            print(f"  位置 {pos_label}: 仅有单一类别，跳过平衡，保留 {len(y_numpy)} 个样本")
            X_parts.append(X_pos)
            y_parts.append(y_pos)
            pos_parts.append(pos_pos)
            total_kept += len(y_numpy)
            continue
        
        class_counts = dict(zip(unique, counts))
        min_count = counts.min()
        print(f"  位置 {pos_label}: 类别分布 {class_counts}，目标各保留 {min_count}")
        
        for cls in sorted(class_counts.keys()):
            cls_indices = np.where(y_numpy == cls)[0]
            cls_count = len(cls_indices)
            if cls_count > min_count:
                selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
                dropped_count = cls_count - min_count
                total_dropped += dropped_count
            else:
                selected_indices = cls_indices
                dropped_count = 0
            
            X_parts.append(X_pos[selected_indices])
            y_parts.append(y_pos[selected_indices])
            pos_parts.append(pos_pos[selected_indices])
            total_kept += len(selected_indices)
            
            print(f"    类别 {cls}: 保留 {len(selected_indices)}，舍弃 {dropped_count}")
    
    if not X_parts:
        print("按位置强平衡后无样本，返回原始数据")
        return X, y, position_indices
    
    X_balanced = torch.cat(X_parts, dim=0)
    y_balanced = torch.cat(y_parts, dim=0)
    pos_balanced = torch.cat(pos_parts, dim=0)
    
    # 打乱数据顺序
    indices = torch.randperm(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    pos_balanced = pos_balanced[indices]
    
    print(f"\n按位置强平衡完成:")
    print(f"  保留样本数: {total_kept}")
    print(f"  舍弃样本数: {total_dropped}")
    print(f"  平衡后数据形状: X {X_balanced.shape}, y {y_balanced.shape}\n")
    
    return X_balanced, y_balanced, pos_balanced


# ==========================================
# ========== 模型定义 ==========
# ==========================================

class SpiralProbe(nn.Module):
    """螺旋探针模型（改进版）：使用多频率螺旋嵌入表示数字
    
    改进点：
    1. 添加MLP编码器，从高维输入提取特征
    2. 简化预测逻辑，主要依赖第一谐波（周期=10，无歧义）
    3. 改进损失函数，分离角度损失和半径损失
    
    基于螺旋基 B(a) = [cos(2πa/T_1), sin(2πa/T_1), ..., cos(2πa/T_k), sin(2πa/T_k)]
    学习从隐藏状态到螺旋基的映射，再还原为数字预测。
    
    损失函数：L = L_angle + λ · L_radius
        - L_angle: 角度匹配损失（核心）
        - L_radius: 单位圆正则化（约束 cos²+sin² ≈ 1）
    """
    def __init__(self, input_dim, num_classes=10, n_harmonics=4, 
                 base_period=10.0, helix_loss_weight=0.1,
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.n_harmonics = n_harmonics  # k 值
        self.base_period = base_period
        self.helix_loss_weight = helix_loss_weight
        
        # 螺旋基维度: n = 2k (只有cos/sin对，不包含线性项)
        self.helix_dim = 2 * n_harmonics
        
        # 周期设置: T_i = base_period / 2^(i-1)
        # T = [10, 5, 2.5, 1.25] for base_period=10, n_harmonics=4
        periods = torch.tensor([base_period / (2 ** i) for i in range(n_harmonics)])
        self.register_buffer('periods', periods)
        
        # 线性投影：直接从 input_dim 映射到 spiral basis
        # 不再使用 MLP encoder
        self.projection = nn.Linear(input_dim, self.helix_dim)
        
    def compute_helix_basis(self, a):
        """
        计算给定数字的螺旋基表示 B(a)
        
        Args:
            a: (batch_size,) 数字值 (float)
        Returns:
            basis: (batch_size, helix_dim) 螺旋基表示
        """
        batch_size = a.shape[0]
        basis = torch.zeros(batch_size, self.helix_dim, device=a.device, dtype=a.dtype)
        
        # cos/sin 对: cos(2πa/T_i), sin(2πa/T_i)
        for i in range(self.n_harmonics):
            T_i = self.periods[i]
            angle = 2 * np.pi * a / T_i
            basis[:, 2*i] = torch.cos(angle)
            basis[:, 2*i + 1] = torch.sin(angle)
        
        return basis
    
    def forward(self, x):
        """
        从隐藏状态预测螺旋基表示
        
        Args:
            x: (batch_size, input_dim) 隐藏状态
        Returns:
            predicted_basis: (batch_size, helix_dim) 预测的螺旋基
        """
        return self.projection(x)
    
    def predict_continuous(self, x):
        """
        从预测的螺旋基还原连续数字值
        
        只使用第一谐波（周期=10）来预测，因为它没有歧义
        
        Args:
            x: (batch_size, input_dim)
        Returns:
            a_pred: (batch_size,) 预测的连续数字值
        """
        predicted_basis = self.forward(x)
        
        # 使用第一谐波 (T_1 = base_period = 10) 来预测
        # 这个谐波覆盖完整的 0-9 范围，没有歧义
        cos_val = predicted_basis[:, 0]
        sin_val = predicted_basis[:, 1]
        
        # 计算角度 θ ∈ [-π, π]
        angle = torch.atan2(sin_val, cos_val)
        
        # 将角度映射到数字: a = angle * T_1 / (2π)
        # angle ∈ [-π, π] → a ∈ [-5, 5]
        a_pred = angle * self.base_period / (2 * np.pi)
        
        # 映射到 [0, 10): 负值加上周期
        a_pred = torch.where(a_pred < 0, a_pred + self.base_period, a_pred)
        
        return a_pred
    
    def predict_class(self, x):
        """
        预测离散类别（用于推理）
        
        Args:
            x: (batch_size, input_dim)
        Returns:
            pred_class: (batch_size,) 预测的类别 (0 到 num_classes-1)
        """
        a_pred = self.predict_continuous(x)
        return a_pred.round().long() % self.num_classes
    
    def compute_loss(self, predicted_basis, target):
        """
        计算总损失 = 角度损失 + λ · 半径正则化损失
        
        Args:
            predicted_basis: (batch_size, helix_dim) 预测的螺旋基
            target: (batch_size,) 目标数字值
        Returns:
            total_loss: 总损失
            angle_loss: 角度匹配损失
            radius_loss: 半径正则化损失
        """
        target_basis = self.compute_helix_basis(target.float())
        
        angle_loss = torch.tensor(0.0, device=predicted_basis.device)
        radius_loss = torch.tensor(0.0, device=predicted_basis.device)
        
        for i in range(self.n_harmonics):
            cos_pred = predicted_basis[:, 2*i]
            sin_pred = predicted_basis[:, 2*i + 1]
            cos_target = target_basis[:, 2*i]
            sin_target = target_basis[:, 2*i + 1]
            
            # 角度损失：使用余弦相似度
            # cos(θ_pred - θ_target) = cos_pred*cos_target + sin_pred*sin_target
            # 我们希望这个值接近1，所以损失 = 1 - cos(θ_pred - θ_target)
            cos_diff = cos_pred * cos_target + sin_pred * sin_target
            
            # 归一化预测值的半径（使角度损失不受半径影响）
            pred_radius = torch.sqrt(cos_pred ** 2 + sin_pred ** 2 + 1e-8)
            cos_diff_normalized = cos_diff / pred_radius
            
            # 低频谐波权重更高（更重要）
            weight = 1.0 / (i + 1)
            angle_loss = angle_loss + weight * (1 - cos_diff_normalized).mean()
            
            # 半径正则化：约束 cos²+sin² ≈ 1
            radius_sq = cos_pred ** 2 + sin_pred ** 2
            radius_loss = radius_loss + ((radius_sq - 1) ** 2).mean()
        
        total_loss = angle_loss + self.helix_loss_weight * radius_loss
        
        return total_loss, angle_loss, radius_loss
        
        return total_loss, main_loss, helix_loss


class CircularProbe(nn.Module):
    """圆形探针模型：将数字0-9假定在圆周上等间隔排列
    
    学习两个线性投影 w1, w2 来计算角度:
        θ = atan2(w1^T·x, w2^T·x) ∈ [0, 2π)
        ŷ = θ · (num_classes / 2π)
    
    训练策略：可以训练两个独立的探针
        - 一个预测模型输出
        - 一个预测真实答案
        - 两者不一致则判定为幻觉
    """
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # 两个线性投影层
        self.w1 = nn.Linear(input_dim, 1, bias=False)
        self.w2 = nn.Linear(input_dim, 1, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes) 用于交叉熵损失
        """
        batch_size = x.shape[0]
        
        # 计算两个投影
        proj1 = self.w1(x).squeeze(-1)  # (batch_size,)
        proj2 = self.w2(x).squeeze(-1)  # (batch_size,)
        
        # 计算角度 θ ∈ [0, 2π)
        theta = torch.atan2(proj1, proj2)  # (batch_size,)
        # 将 [-π, π] 映射到 [0, 2π)
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        
        # 将角度映射到数字 ŷ = θ · (num_classes / 2π)
        predictions = theta * (self.num_classes / (2 * np.pi))  # (batch_size,)
        
        # 为了使用CrossEntropyLoss，需要构造logits
        # 使用von Mises分布的近似：在预测的数字附近给高分
        angles_per_class = 2 * np.pi / self.num_classes
        class_angles = torch.arange(self.num_classes, device=x.device).float() * angles_per_class
        
        # 计算每个样本与每个类别角度的距离（在圆周上）
        # 使用cosine相似度的形式
        theta_expanded = theta.unsqueeze(1)  # (batch_size, 1)
        class_angles_expanded = class_angles.unsqueeze(0)  # (1, num_classes)
        
        # 圆周距离：使用cos(θ - θ_class)，值越大越接近
        # 乘以大的常数使其更像one-hot
        logits = torch.cos(theta_expanded - class_angles_expanded) * 10.0
        
        return logits
    
    def predict_class(self, x):
        """直接预测类别（用于推理）"""
        batch_size = x.shape[0]
        proj1 = self.w1(x).squeeze(-1)
        proj2 = self.w2(x).squeeze(-1)
        theta = torch.atan2(proj1, proj2)
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        predictions = theta * (self.num_classes / (2 * np.pi))
        return predictions.round().long() % self.num_classes

class ProbeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, num_classes=2):
        super(ProbeMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, feature_dim, d_model=256, nhead=8, num_layers=2, 
                 dim_feedforward=1024, dropout=0.1, num_classes=2):
        super(TransformerClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        x = self.input_projection(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output


class RoPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # Apply rotation
        x_half1 = x[..., :self.d_model//2]
        x_half2 = x[..., self.d_model//2:]
        x_rotated = torch.cat((-x_half2, x_half1), dim=-1)
        return x * cos + x_rotated * sin

class AutoregressiveTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, feature_dim, d_model=256, nhead=8, num_layers=2, 
                 dim_feedforward=1024, dropout=0.1, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.rope = RoPE(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        x = self.input_projection(x)
        x = self.rope(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x.device), diagonal=1).bool()
        
        x = self.transformer_encoder(x, mask=mask)
        # Use the last token's representation for classification
        x = x[:, -1, :]
        output = self.classifier(x)
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, feature_dim, hidden_dim=256, num_layers=2, dropout=0.1, bidirectional=True, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        combined_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state (for bidirectional, we might want to concatenate or just take the last output)
        # Here we take the last time step's output which contains both directions' info if bidirectional
        last_hidden = lstm_out[:, -1, :]
        output = self.classifier(last_hidden)
        return output

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, feature_dim, num_filters=[64, 128, 256], 
                 kernel_sizes=[3, 3, 3], dropout=0.2, num_classes=2):
        super(CNNClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        conv_layers = []
        in_channels = feature_dim
        
        for out_channels, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        fc_input_dim = num_filters[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.feature_dim, self.seq_len)
        x = self.conv_layers(x)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)
        max_pooled = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        output = self.classifier(x)
        return output


class CNN2DClassifier(nn.Module):
    """把 vertical flow (seq_len, feature_dim) 当作图像处理的 2D CNN"""
    def __init__(self, input_dim, seq_len, feature_dim, channels=[32, 64, 128, 256], 
                 dropout=0.2, num_classes=2):
        """
        Args:
            input_dim: 展平后的输入维度 (seq_len * feature_dim)
            seq_len: 序列长度（层数），作为图像高度
            feature_dim: 特征维度，作为图像宽度
            channels: 每层输出通道数列表
            dropout: Dropout比率
            num_classes: 分类类别数
        """
        super(CNN2DClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # 使用非对称卷积核，适应 seq_len × feature_dim 的扁平形状
        # seq_len 较小(37)，feature_dim 较大(2560)
        self.conv_layers = nn.Sequential(
            # 第一层：捕获局部层间模式
            nn.Conv2d(1, channels[0], kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # seq_len × (feature_dim/4)
            
            # 第二层
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # seq_len × (feature_dim/16)
            
            # 第三层
            nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4)),  # (seq_len/2) × (feature_dim/64)
            
            # 第四层
            nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化到 1×1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[3], channels[3] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[3] // 4, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        # 重塑为图像格式: (batch, 1, seq_len, feature_dim)
        x = x.view(batch_size, 1, self.seq_len, self.feature_dim)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class LogisticRegressionClassifier(nn.Module):
    """线性逻辑回归（softmax），适合做线性可分基线"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LinearSVMClassifier(nn.Module):
    """线性 SVM（使用 MultiMarginLoss 的 score 输入）"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# ==========================================
# ========== 训练与评估工具函数 ==========
# ==========================================

def create_model(input_dim, seq_len, feature_dim, num_classes=2):
    """根据全局 MODEL_TYPE 创建模型实例"""
    if MODEL_TYPE == 'mlp':
        return ProbeMLP(
            input_dim=input_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            dropout=MLP_DROPOUT,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'mlp10':
        return ProbeMLP(
            input_dim=input_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            dropout=MLP_DROPOUT,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'transformer':
        return TransformerClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            num_layers=TRANSFORMER_NUM_LAYERS,
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
            dropout=TRANSFORMER_DROPOUT,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'cnn':
        return CNNClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            num_filters=CNN_NUM_FILTERS,
            kernel_sizes=CNN_KERNEL_SIZES,
            dropout=CNN_DROPOUT,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'cnn2d':
        return CNN2DClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            channels=CNN2D_CHANNELS,
            dropout=CNN2D_DROPOUT,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'logreg':
        return LogisticRegressionClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'svm':
        return LinearSVMClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'ar_transformer':
        return AutoregressiveTransformerClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            d_model=AR_TRANSFORMER_D_MODEL,
            nhead=AR_TRANSFORMER_NHEAD,
            num_layers=AR_TRANSFORMER_NUM_LAYERS,
            dim_feedforward=AR_TRANSFORMER_DIM_FEEDFORWARD,
            dropout=AR_TRANSFORMER_DROPOUT,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'lstm':
        return LSTMClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            bidirectional=LSTM_BIDIRECTIONAL,
            num_classes=num_classes
        ).to(DEVICE)
    if MODEL_TYPE == 'circular_probe':
        return CircularProbe(
            input_dim=input_dim,
            num_classes=CIRCULAR_PROBE_NUM_CLASSES
        ).to(DEVICE)
    if MODEL_TYPE == 'spiral_probe':
        return SpiralProbe(
            input_dim=input_dim,
            num_classes=SPIRAL_PROBE_NUM_CLASSES,
            n_harmonics=SPIRAL_PROBE_N_HARMONICS,
            base_period=SPIRAL_PROBE_BASE_PERIOD,
            helix_loss_weight=SPIRAL_PROBE_HELIX_LOSS_WEIGHT,
            hidden_dim=SPIRAL_PROBE_HIDDEN_DIM,
            dropout=SPIRAL_PROBE_DROPOUT
        ).to(DEVICE)
    raise ValueError(f"未知的模型类型: {MODEL_TYPE}")


def evaluate(model, data_loader, criterion):
    """在验证集上评估，返回指标字典（支持二分类与多分类）"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels, _pos in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    loss_avg = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    # 根据类别数量选择合适的评估方式
    num_classes = all_probs[0].shape[-1] if all_probs else 2
    if num_classes == 2:
        # 二分类：使用第1类概率计算ROC-AUC；F1以正类=0保持向后兼容
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        f1 = f1_score(all_labels, all_preds, pos_label=0)
    else:
        # 多分类：使用macro OVR的ROC-AUC与macro F1
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
        f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        "loss": loss_avg,
        "acc": acc,
        "auc": auc,
        "f1": f1
    }


def train_single_run(X_train, y_train, pos_train, X_val, y_val, pos_val, seq_len, feature_dim, label_prefix=""):
    """
    单次训练流程，返回模型和最佳 AUC 结果
    
    label_prefix: 日志前缀，便于按层评估时区分输出
    """
    input_dim = X_train.shape[1]
    # 动态确定类别数（适配二分类与多分类）
    num_classes = int(torch.unique(y_train).numel())
    
    train_dataset = TensorDataset(X_train, y_train, pos_train)
    val_dataset = TensorDataset(X_val, y_val, pos_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = create_model(input_dim, seq_len, feature_dim, num_classes=num_classes)
    print(f"{label_prefix}使用模型: {MODEL_TYPE}")
    print(f"{label_prefix}模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    if MODEL_TYPE == 'svm':
        criterion = nn.MultiMarginLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    
    # CircularProbe特殊训练逻辑：训练两个探针
    if MODEL_TYPE == 'circular_probe':
        # CircularProbe使用平滑L1损失（回归损失）
        criterion = nn.SmoothL1Loss()
        print(f"{label_prefix}CircularProbe双探针训练模式：")
        print(f"{label_prefix}  探针1: 学习预测模型输出 (preds)")
        print(f"{label_prefix}  探针2: 学习预测真实答案 (gt_chars)")
        print(f"{label_prefix}模型参数量: {sum(p.numel() for p in model.parameters()):,} (每个探针)")
        
        # 创建第二个探针
        probe2 = create_model(input_dim, seq_len, feature_dim, num_classes=num_classes)
        optimizer2 = optim.Adam(probe2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # 创建数据加载器（探针1用preds，探针2用gt_chars）
        train_dataset_probe1 = TensorDataset(X_train, GLOBAL_PREDS_TRAIN, pos_train)
        val_dataset_probe1 = TensorDataset(X_val, GLOBAL_PREDS_VAL, pos_val)
        train_loader_probe1 = DataLoader(train_dataset_probe1, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_probe1 = DataLoader(val_dataset_probe1, batch_size=BATCH_SIZE, shuffle=False)
        
        train_dataset_probe2 = TensorDataset(X_train, GLOBAL_GT_CHARS_TRAIN, pos_train)
        val_dataset_probe2 = TensorDataset(X_val, GLOBAL_GT_CHARS_VAL, pos_val)
        train_loader_probe2 = DataLoader(train_dataset_probe2, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_probe2 = DataLoader(val_dataset_probe2, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"{label_prefix}开始训练{CIRCULAR_PROBE_EPOCHS}个epoch (带早停)...")
        
        best_avg_acc = -1.0
        best_epoch = -1
        no_improve_epochs = 0
        best_state_model = None
        best_state_probe2 = None
        
        for epoch in range(CIRCULAR_PROBE_EPOCHS):
            # 训练探针1 (model)
            model.train()
            loss1_sum = 0.0
            for inputs, labels, _pos in train_loader_probe1:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                # 直接计算连续的预测值（保留梯度图）
                proj1 = model.w1(inputs).squeeze(-1)
                proj2 = model.w2(inputs).squeeze(-1)
                theta = torch.atan2(proj1, proj2)
                theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
                preds = theta * (model.num_classes / (2 * np.pi))
                loss = criterion(preds, labels.float())
                loss.backward()
                optimizer.step()
                loss1_sum += loss.item()
            
            # 训练探针2
            probe2.train()
            loss2_sum = 0.0
            for inputs, labels, _pos in train_loader_probe2:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer2.zero_grad()
                
                # 直接计算连续的预测值（保留梯度图）
                proj1 = probe2.w1(inputs).squeeze(-1)
                proj2 = probe2.w2(inputs).squeeze(-1)
                theta = torch.atan2(proj1, proj2)
                theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
                preds = theta * (probe2.num_classes / (2 * np.pi))
                loss = criterion(preds, labels.float())
                loss.backward()
                optimizer2.step()
                loss2_sum += loss.item()
            
            # 验证 (每个 epoch)
            model.eval()
            probe2.eval()
            with torch.no_grad():
                # 探针1 验证
                pred1_list = []
                label1_list = []
                for inputs, labels, _pos in val_loader_probe1:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    preds = model.predict_class(inputs)
                    pred1_list.extend(preds.cpu().numpy())
                    label1_list.extend(labels.cpu().numpy())
                acc1 = accuracy_score(label1_list, pred1_list)
                
                # 探针2 验证
                pred2_list = []
                label2_list = []
                for inputs, labels, _pos in val_loader_probe2:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    preds = probe2.predict_class(inputs)
                    pred2_list.extend(preds.cpu().numpy())
                    label2_list.extend(labels.cpu().numpy())
                acc2 = accuracy_score(label2_list, pred2_list)
            
            avg_acc = (acc1 + acc2) / 2
            
            if (epoch + 1) % 10 == 0:
                print(f"{label_prefix}Epoch [{epoch+1}/{CIRCULAR_PROBE_EPOCHS}] "
                      f"Loss: {loss1_sum/len(train_loader_probe1):.4f} | {loss2_sum/len(train_loader_probe2):.4f} "
                      f"Acc: {acc1:.4f} | {acc2:.4f}")
            
            # 早停检查
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_epoch = epoch + 1
                best_state_model = copy.deepcopy(model.state_dict())
                best_state_probe2 = copy.deepcopy(probe2.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"{label_prefix}连续 {EARLY_STOP_PATIENCE} 个 epoch 验证 Acc 未提升，提前停止。")
                break
        
        print(f"{label_prefix}训练完成，恢复最佳权重 (Epoch {best_epoch}, Acc: {best_avg_acc:.4f})")
        if best_state_model:
            model.load_state_dict(best_state_model)
        if best_state_probe2:
            probe2.load_state_dict(best_state_probe2)
        
        # 最终评估 (使用最佳权重)
        model.eval()
        probe2.eval()
        with torch.no_grad():
            pred1_list = []
            for inputs, _, _ in val_loader_probe1:
                inputs = inputs.to(DEVICE)
                pred1_list.extend(model.predict_class(inputs).cpu().numpy())
            
            pred2_list = []
            for inputs, _, _ in val_loader_probe2:
                inputs = inputs.to(DEVICE)
                pred2_list.extend(probe2.predict_class(inputs).cpu().numpy())
        
        disagreement = sum(np.array(pred1_list) != np.array(pred2_list)) / len(pred1_list)
        print(f"{label_prefix}探针预测不一致率: {disagreement:.4f} (潜在幻觉指标)")
        
        return (model, probe2), {"best_auc": best_avg_acc, "best_epoch": best_epoch, "val_loader": val_loader}
    
    # SpiralProbe特殊训练逻辑：训练两个探针
    if MODEL_TYPE == 'spiral_probe':
        print(f"{label_prefix}SpiralProbe双探针训练模式：")
        print(f"{label_prefix}  探针1: 学习预测模型输出 (preds)")
        print(f"{label_prefix}  探针2: 学习预测真实答案 (gt_chars)")
        print(f"{label_prefix}  螺旋基维度: {model.helix_dim} (1 + 2×{model.n_harmonics})")
        print(f"{label_prefix}  周期: {model.periods.tolist()}")
        print(f"{label_prefix}模型参数量: {sum(p.numel() for p in model.parameters()):,} (每个探针)")
        
        # 创建第二个探针
        probe2 = create_model(input_dim, seq_len, feature_dim, num_classes=num_classes)
        optimizer2 = optim.Adam(probe2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # 创建数据加载器（探针1用preds，探针2用gt_chars）
        train_dataset_probe1 = TensorDataset(X_train, GLOBAL_PREDS_TRAIN, pos_train)
        val_dataset_probe1 = TensorDataset(X_val, GLOBAL_PREDS_VAL, pos_val)
        train_loader_probe1 = DataLoader(train_dataset_probe1, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_probe1 = DataLoader(val_dataset_probe1, batch_size=BATCH_SIZE, shuffle=False)
        
        train_dataset_probe2 = TensorDataset(X_train, GLOBAL_GT_CHARS_TRAIN, pos_train)
        val_dataset_probe2 = TensorDataset(X_val, GLOBAL_GT_CHARS_VAL, pos_val)
        train_loader_probe2 = DataLoader(train_dataset_probe2, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_probe2 = DataLoader(val_dataset_probe2, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"{label_prefix}开始训练{SPIRAL_PROBE_EPOCHS}个epoch (带早停)...")
        
        best_avg_acc = -1.0
        best_epoch = -1
        no_improve_epochs = 0
        best_state_model = None
        best_state_probe2 = None
        
        for epoch in range(SPIRAL_PROBE_EPOCHS):
            # 训练探针1 (model)
            model.train()
            loss1_sum = 0.0
            main_loss1_sum = 0.0
            helix_loss1_sum = 0.0
            for inputs, labels, _pos in train_loader_probe1:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                predicted_basis = model(inputs)
                total_loss, main_loss, helix_loss = model.compute_loss(predicted_basis, labels)
                total_loss.backward()
                optimizer.step()
                
                loss1_sum += total_loss.item()
                main_loss1_sum += main_loss.item()
                helix_loss1_sum += helix_loss.item()
            
            # 训练探针2
            probe2.train()
            loss2_sum = 0.0
            main_loss2_sum = 0.0
            helix_loss2_sum = 0.0
            for inputs, labels, _pos in train_loader_probe2:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer2.zero_grad()
                
                predicted_basis = probe2(inputs)
                total_loss, main_loss, helix_loss = probe2.compute_loss(predicted_basis, labels)
                total_loss.backward()
                optimizer2.step()
                
                loss2_sum += total_loss.item()
                main_loss2_sum += main_loss.item()
                helix_loss2_sum += helix_loss.item()
            
            # 验证 (每个 epoch)
            model.eval()
            probe2.eval()
            with torch.no_grad():
                # 探针1 验证
                pred1_list = []
                label1_list = []
                for inputs, labels, _pos in val_loader_probe1:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    preds = model.predict_class(inputs)
                    pred1_list.extend(preds.cpu().numpy())
                    label1_list.extend(labels.cpu().numpy())
                acc1 = accuracy_score(label1_list, pred1_list)
                
                # 探针2 验证
                pred2_list = []
                label2_list = []
                for inputs, labels, _pos in val_loader_probe2:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    preds = probe2.predict_class(inputs)
                    pred2_list.extend(preds.cpu().numpy())
                    label2_list.extend(labels.cpu().numpy())
                acc2 = accuracy_score(label2_list, pred2_list)
            
            avg_acc = (acc1 + acc2) / 2
            
            if (epoch + 1) % 10 == 0:
                n_batches1 = len(train_loader_probe1)
                n_batches2 = len(train_loader_probe2)
                print(f"{label_prefix}Epoch [{epoch+1}/{SPIRAL_PROBE_EPOCHS}] "
                      f"Probe1 [Total: {loss1_sum/n_batches1:.4f}, Angle: {main_loss1_sum/n_batches1:.4f}, Radius: {helix_loss1_sum/n_batches1:.4f}] | "
                      f"Probe2 [Total: {loss2_sum/n_batches2:.4f}, Angle: {main_loss2_sum/n_batches2:.4f}, Radius: {helix_loss2_sum/n_batches2:.4f}] "
                      f"Acc: {acc1:.4f} | {acc2:.4f}")
            
            # 早停检查
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_epoch = epoch + 1
                best_state_model = copy.deepcopy(model.state_dict())
                best_state_probe2 = copy.deepcopy(probe2.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"{label_prefix}连续 {EARLY_STOP_PATIENCE} 个 epoch 验证 Acc 未提升，提前停止。")
                break
        
        print(f"{label_prefix}训练完成，恢复最佳权重 (Epoch {best_epoch}, Acc: {best_avg_acc:.4f})")
        if best_state_model:
            model.load_state_dict(best_state_model)
        if best_state_probe2:
            probe2.load_state_dict(best_state_probe2)
        
        # 最终评估
        model.eval()
        probe2.eval()
        with torch.no_grad():
            pred1_list = []
            for inputs, _, _ in val_loader_probe1:
                inputs = inputs.to(DEVICE)
                pred1_list.extend(model.predict_class(inputs).cpu().numpy())
            
            pred2_list = []
            for inputs, _, _ in val_loader_probe2:
                inputs = inputs.to(DEVICE)
                pred2_list.extend(probe2.predict_class(inputs).cpu().numpy())
        
        disagreement = sum(np.array(pred1_list) != np.array(pred2_list)) / len(pred1_list)
        print(f"{label_prefix}探针预测不一致率: {disagreement:.4f} (潜在幻觉指标)")
        
        return (model, probe2), {"best_auc": best_avg_acc, "best_epoch": best_epoch, "val_loader": val_loader}
    
    # 其他模型的正常训练流程（带验证）
    # 初始验证（随机初始化）
    init_metrics = evaluate(model, val_loader, criterion)
    print(f"{label_prefix}Epoch [-1] "
          f"Train Loss: ------ | "
          f"Val Loss: {init_metrics['loss']:.4f} | "
          f"Val Acc: {init_metrics['acc']:.4f} | "
          f"Val AUC: {init_metrics['auc']:.4f} | "
          f"Val F1: {init_metrics['f1']:.4f}")
    
    best_val_auc = float("-inf")
    best_epoch = -1
    best_state = None
    no_improve_epochs = 0
    epoch = 0
    
    while True:
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels, _pos in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss_avg = running_loss / len(train_loader)
        
        # 验证阶段
        val_metrics = evaluate(model, val_loader, criterion)
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1  # epoch 从 0 开始计数，这里输出 1-based
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        print(f"{label_prefix}Epoch [{epoch+1}] "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"No Improve: {no_improve_epochs}/{EARLY_STOP_PATIENCE}")
        
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"{label_prefix}连续 {EARLY_STOP_PATIENCE} 个 epoch 验证 AUC 未提升，提前停止。")
            break
        
        epoch += 1
    
    # 恢复最佳 AUC 的权重
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"{label_prefix}训练完成! 最佳验证AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    return model, {
        "best_auc": best_val_auc,
        "best_epoch": best_epoch,
        "val_loader": val_loader  # 便于后续位置统计
    }


# ==========================================
# ========== 主程序 ==========
# ==========================================

def main():
    training_executed = False  # 用于 parallel_runner 下判断是否真正训练
    setup_logger()
    
    pooling_type = POOLING_TYPE
    # CNN2D 需要保持原始形状，不能使用 pooling
    if MODEL_TYPE == 'cnn2d' and POOLING_TYPE is not None:
        print(f"警告: CNN2D 模型需要原始形状，已将 POOLING_TYPE 从 '{POOLING_TYPE}' 改为 None")
        pooling_type = None
    if EVALUATE_EACH_LAYER and pooling_type is not None:
        print(f"警告: 按层评估需要保留原始层信息，已将 POOLING_TYPE 从 '{pooling_type}' 改为 None")
        pooling_type = None
    if EVALUATE_EACH_LAYER and USE_PCA:
        raise ValueError("按层评估模式不支持 PCA，请将 USE_PCA 设为 False")
    
    print("=" * 60)
    print("配置参数:")
    print(f"  数据文件: {DATA_FILE_PATH}")
    print(f"  位置选择: {POSITION_SELECT}")
    print(f"  特征类型: {FEATURE_TYPE}")
    print(f"  Pooling类型: {pooling_type}")
    print(f"  模型类型: {MODEL_TYPE}")
    print(f"  早停耐心: {EARLY_STOP_PATIENCE}")
    print(f"  数据平衡: {BALANCE_DATASET}")
    print(f"  按位置强平衡: {STRONG_BALANCE_BY_POSITION}")
    print(f"  使用PCA: {USE_PCA}")
    print(f"  按层评估: {EVALUATE_EACH_LAYER}")
    print(f"  按位置评估: {EVALUATE_EACH_POSITION}")
    print("=" * 60)
    
    # 加载数据
    X_all, y_all, position_indices, selected_positions, seq_len, feature_dim, gt_chars_all, preds_all = load_and_process_data(
        DATA_FILE_PATH,
        position_select=POSITION_SELECT,
        feature_type=FEATURE_TYPE,
        pooling_type=pooling_type,
        use_pca=USE_PCA,
        pca_dim=PCA_DIM
    )
    
    # 若为10类MLP，使用数据中的 gt_chars 作为标签，并过滤无效样本
    if MODEL_TYPE == 'mlp10':
        print("使用10类MLP：标签采用数据文件中的 gt_chars (0-9)")
        valid_mask = (gt_chars_all >= 0) & (gt_chars_all < 10)
        kept = int(valid_mask.sum().item())
        dropped = int((~valid_mask).sum().item())
        if dropped > 0:
            print(f"  过滤无效标签样本: 保留 {kept}，舍弃 {dropped}（gt_chars为-1或不在0-9）")
        X_all = X_all[valid_mask]
        y_all = gt_chars_all[valid_mask]
        position_indices = position_indices[valid_mask]
        gt_chars_all = gt_chars_all[valid_mask]
        preds_all = preds_all[valid_mask]
    
    # 数据平衡（仅适用于二分类）。多分类时跳过。
    if MODEL_TYPE != 'mlp10':
        if STRONG_BALANCE_BY_POSITION:
            X_all, y_all, position_indices = balance_dataset_per_position(
                X_all, y_all, position_indices, position_names=selected_positions
            )
        elif BALANCE_DATASET:
            X_all, y_all, position_indices = balance_dataset(X_all, y_all, position_indices)
    else:
        if STRONG_BALANCE_BY_POSITION or BALANCE_DATASET:
            print("提示: mlp10 为多分类任务，已跳过二分类的平衡步骤。")
    
    # 数据划分
    indices = np.arange(len(X_all))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all.numpy()
    )
    
    # 为CircularProbe/SpiralProbe准备gt_chars和preds数据（所有分支都需要）
    global GLOBAL_GT_CHARS_TRAIN, GLOBAL_GT_CHARS_VAL, GLOBAL_PREDS_TRAIN, GLOBAL_PREDS_VAL
    GLOBAL_GT_CHARS_TRAIN = gt_chars_all[train_idx]
    GLOBAL_GT_CHARS_VAL = gt_chars_all[val_idx]
    GLOBAL_PREDS_TRAIN = preds_all[train_idx]
    GLOBAL_PREDS_VAL = preds_all[val_idx]
    
    if not EVALUATE_EACH_LAYER and not EVALUATE_EACH_POSITION:
        # 常规单次训练
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        pos_train, pos_val = position_indices[train_idx], position_indices[val_idx]
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        
        print(f"\n{'='*60}")
        print(f"开始训练 - 设备: {DEVICE}, 模型: {MODEL_TYPE}")
        print(f"{'='*60}\n")
        
        models_or_model, result = train_single_run(
            X_train, y_train, pos_train,
            X_val, y_val, pos_val,
            seq_len=seq_len, feature_dim=feature_dim
        )
        training_executed = True
        
        # 解包CircularProbe/SpiralProbe的两个探针
        if MODEL_TYPE in ['circular_probe', 'spiral_probe']:
            model, probe2 = models_or_model
        else:
            model = models_or_model
        
        best_val_auc = result["best_auc"]
        best_epoch = result["best_epoch"]
        val_loader = result["val_loader"]
        
        # 保存模型（可选）
        if SAVE_MODEL:
            save_dir = Path(SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"{MODEL_TYPE}_{SAVE_NAME}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": MODEL_TYPE,
                    "feature_type": FEATURE_TYPE,
                    "position_select": POSITION_SELECT,
                    "pooling_type": pooling_type,
                    "use_pca": USE_PCA,
                    "pca_dim": PCA_DIM,
                    "seq_len": seq_len,
                    "feature_dim": feature_dim,
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                },
                save_path,
            )
            print(f"\n模型已保存到: {save_path}")
        
        # 训练结束后，按位置统计验证集AUC
        print("\n按位置统计验证AUC:")
        # CircularProbe/SpiralProbe跳过AUC统计（已在训练时评估）
        if MODEL_TYPE in ['circular_probe', 'spiral_probe']:
            print(f"  {MODEL_TYPE}已在训练时完成评估，跳过按位置AUC统计")
            # 计算最终一致性准确率
            model.eval()
            probe2.eval()
            with torch.no_grad():
                X_val_tensor = X_val.to(DEVICE)
                probe1_preds = model.predict_class(X_val_tensor).cpu().numpy()
                probe2_preds = probe2.predict_class(X_val_tensor).cpu().numpy()
                
                # 两探针预测一致为1，不一致为0
                agreement = (probe1_preds == probe2_preds).astype(int)
                
                # 与真实标签比较
                from sklearn.metrics import accuracy_score
                final_acc = accuracy_score(y_val.numpy(), agreement)
                print(f"\n{'='*60}")
                print(f"最终一致性准确率: {final_acc:.4f}")
                print(f"  (两探针一致判为正确=1, 不一致判为错误=0, 与labels比较)")
                print(f"{'='*60}\n")
        else:
            per_pos_labels = {i: [] for i in range(len(selected_positions))}
            per_pos_probs = {i: [] for i in range(len(selected_positions))}

            model.eval()
            with torch.no_grad():
                for inputs, labels, pos_idx in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    probs_full = torch.softmax(outputs, dim=1).cpu().numpy()
                    pos_idx_np = pos_idx.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    for i in range(len(pos_idx_np)):
                        idx = int(pos_idx_np[i])
                        if probs_full.shape[1] == 2:
                            per_pos_probs[idx].append(probs_full[i, 1])
                        else:
                            per_pos_probs[idx].append(probs_full[i])
                        per_pos_labels[idx].append(labels_np[i])

            for idx, pos in enumerate(selected_positions):
                y_true = np.array(per_pos_labels[idx])
                y_score = np.array(per_pos_probs[idx])
                if len(y_true) == 0:
                    print(f"  位置 {pos}: 验证样本数为 0，无法计算AUC")
                elif len(np.unique(y_true)) < 2:
                    print(f"  位置 {pos}: 仅有一种标签，无法计算AUC")
                else:
                    if MODEL_TYPE == 'mlp10':
                        # 多分类：macro OVR AUC
                        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
                    else:
                        auc = roc_auc_score(y_true, y_score)
                    print(f"  位置 {pos}: 验证AUC {auc:.4f} (样本数 {len(y_true)})")
    
            
            # 计算总的AUC（合并所有位置）
            all_labels = []
            all_probs = []
            for idx in range(len(selected_positions)):
                all_labels.extend(per_pos_labels[idx])
                all_probs.extend(per_pos_probs[idx])
            
            if len(all_labels) > 0 and len(np.unique(all_labels)) >= 2:
                if MODEL_TYPE == 'mlp10':
                    overall_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
                else:
                    overall_auc = roc_auc_score(all_labels, all_probs)
                print(f"\n  总体验证AUC: {overall_auc:.4f} (总样本数 {len(all_labels)})")
    elif EVALUATE_EACH_POSITION and not EVALUATE_EACH_LAYER:
        # 按位置评估
        print(f"\n{'='*60}")
        print("开启按位置评估模式：每个位置都会独立训练并报告最佳验证 AUC。")
        print(f"{'='*60}\n")
        
        position_results = []
        for pos_idx, pos in enumerate(selected_positions):
            mask = (position_indices == pos_idx)
            sample_count = int(mask.sum().item())
            if sample_count == 0:
                print(f"位置 {pos} 无样本，跳过。")
                continue
            
            X_pos = X_all[mask]
            y_pos = y_all[mask]
            pos_pos = position_indices[mask]

            pos_pos_count = int((y_pos == 1).sum().item())
            pos_neg_count = int((y_pos == 0).sum().item())
            if pos_pos_count < 200 or pos_neg_count < 200:
                print(f"位置 {pos} 正样本 {pos_pos_count} / 负样本 {pos_neg_count}，低于 200，跳过。")
                continue
            
            sub_indices = np.arange(len(X_pos))
            strat_labels = y_pos.numpy()
            stratify_param = strat_labels if len(np.unique(strat_labels)) > 1 else None
            try:
                train_sub_idx, val_sub_idx = train_test_split(
                    sub_indices,
                    test_size=TEST_SIZE,
                    random_state=SEED,
                    stratify=stratify_param
                )
            except ValueError:
                train_sub_idx, val_sub_idx = train_test_split(
                    sub_indices,
                    test_size=TEST_SIZE,
                    random_state=SEED,
                    stratify=None
                )
            
            print(f"\n{'-'*40}")
            print(f"开始位置 {pos} 的训练，样本数 {sample_count}")
            
            model, result = train_single_run(
                X_pos[train_sub_idx], y_pos[train_sub_idx], pos_pos[train_sub_idx],
                X_pos[val_sub_idx], y_pos[val_sub_idx], pos_pos[val_sub_idx],
                seq_len=seq_len, feature_dim=feature_dim,
                label_prefix=f"[Pos {pos}] "
            )
            training_executed = True
            
            position_results.append({
                "position": pos,
                "position_idx": pos_idx,
                "best_auc": result["best_auc"],
                "best_epoch": result["best_epoch"]
            })
        
        print(f"\n{'='*60}")
        print("按位置评估结果汇总（按 AUC 降序）：")
        position_results_sorted = sorted(position_results, key=lambda x: x["best_auc"], reverse=True)
        for item in position_results_sorted:
            print(f"  位置 {item['position']}: 最佳 AUC = {item['best_auc']:.4f} (Epoch {item['best_epoch']})")
        if position_results_sorted:
            best_pos = position_results_sorted[0]
            print(f"\n最佳位置: {best_pos['position']}，最佳验证 AUC: {best_pos['best_auc']:.4f} (Epoch {best_pos['best_epoch']})")
    
    else:
        # 按层评估（可与按位置评估组合）
        print(f"\n{'='*60}")
        if EVALUATE_EACH_POSITION:
            print("开启按位置 + 按层评估模式：每个位置、每层都会独立训练并报告最佳验证 AUC。")
        else:
            print("开启按层评估模式：每一层都会独立训练并报告最佳验证 AUC。")
        print(f"{'='*60}\n")
        
        # 恢复每层的二维形状 (seq_len, feature_dim)
        X_reshaped = X_all.view(len(X_all), seq_len, feature_dim)
        
        if not EVALUATE_EACH_POSITION:
            layer_results = []
            for layer_idx in range(seq_len):
                if SPECIFIC_LAYER_INDEX is not None and layer_idx != SPECIFIC_LAYER_INDEX:
                    continue
                print(f"\n{'-'*40}")
                print(f"开始第 {layer_idx} 层的训练")
                X_layer = X_reshaped[:, layer_idx, :]
                X_train, X_val = X_layer[train_idx], X_layer[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]
                pos_train, pos_val = position_indices[train_idx], position_indices[val_idx]
                
                model, result = train_single_run(
                    X_train, y_train, pos_train,
                    X_val, y_val, pos_val,
                    seq_len=1, feature_dim=feature_dim,
                    label_prefix=f"[Layer {layer_idx}] "
                )
                training_executed = True
                
                layer_results.append({
                    "layer": layer_idx,
                    "best_auc": result["best_auc"],
                    "best_epoch": result["best_epoch"]
                })
            
            print(f"\n{'='*60}")
            print("按层评估结果汇总（按 AUC 降序）：")
            layer_results_sorted = sorted(layer_results, key=lambda x: x["best_auc"], reverse=True)
            for item in layer_results_sorted:
                print(f"  层 {item['layer']:>2}: 最佳 AUC = {item['best_auc']:.4f} (Epoch {item['best_epoch']})")
            if layer_results_sorted:
                best_layer = layer_results_sorted[0]
                print(f"\n最佳层: {best_layer['layer']}，最佳验证 AUC: {best_layer['best_auc']:.4f} (Epoch {best_layer['best_epoch']})")
        else:
            # 按位置 + 按层
            pos_layer_results = []
            for pos_idx, pos in enumerate(selected_positions):
                mask = (position_indices == pos_idx)
                sample_count = int(mask.sum().item())
                if sample_count == 0:
                    print(f"位置 {pos} 无样本，跳过。")
                    continue
                
                X_pos = X_reshaped[mask]
                y_pos = y_all[mask]
                pos_pos = position_indices[mask]

                pos_pos_count = int((y_pos == 1).sum().item())
                pos_neg_count = int((y_pos == 0).sum().item())
                if pos_pos_count < 200 or pos_neg_count < 200:
                    print(f"位置 {pos} 正样本 {pos_pos_count} / 负样本 {pos_neg_count}，低于 200，跳过。")
                    continue
                
                sub_indices = np.arange(len(X_pos))
                strat_labels = y_pos.numpy()
                stratify_param = strat_labels if len(np.unique(strat_labels)) > 1 else None
                try:
                    train_sub_idx, val_sub_idx = train_test_split(
                        sub_indices,
                        test_size=TEST_SIZE,
                        random_state=SEED,
                        stratify=stratify_param
                    )
                except ValueError:
                    train_sub_idx, val_sub_idx = train_test_split(
                        sub_indices,
                        test_size=TEST_SIZE,
                        random_state=SEED,
                        stratify=None
                    )
                
                print(f"\n{'-'*40}")
                print(f"开始位置 {pos} 的按层训练，样本数 {sample_count}")
                
                for layer_idx in range(seq_len):
                    if SPECIFIC_LAYER_INDEX is not None and layer_idx != SPECIFIC_LAYER_INDEX:
                        continue
                    print(f"  -> 位置 {pos} | 层 {layer_idx}")
                    X_layer = X_pos[:, layer_idx, :]
                    
                    model, result = train_single_run(
                        X_layer[train_sub_idx], y_pos[train_sub_idx], pos_pos[train_sub_idx],
                        X_layer[val_sub_idx], y_pos[val_sub_idx], pos_pos[val_sub_idx],
                        seq_len=1, feature_dim=feature_dim,
                        label_prefix=f"[Pos {pos}][Layer {layer_idx}] "
                    )
                    training_executed = True
                    
                    pos_layer_results.append({
                        "position": pos,
                        "position_idx": pos_idx,
                        "layer": layer_idx,
                        "best_auc": result["best_auc"],
                        "best_epoch": result["best_epoch"]
                    })
            
            print(f"\n{'='*60}")
            print("按位置按层评估结果汇总（按位置分组，层内按 AUC 降序）：")
            for pos_idx, pos in enumerate(selected_positions):
                pos_results = [r for r in pos_layer_results if r["position_idx"] == pos_idx]
                if not pos_results:
                    print(f"  位置 {pos}: 无结果")
                    continue
                pos_results_sorted = sorted(pos_results, key=lambda x: x["best_auc"], reverse=True)
                print(f"\n位置 {pos}:")
                for item in pos_results_sorted:
                    print(f"  层 {item['layer']:>2}: 最佳 AUC = {item['best_auc']:.4f} (Epoch {item['best_epoch']})")
                best_item = pos_results_sorted[0]
                print(f"  最佳层: {best_item['layer']}，最佳验证 AUC: {best_item['best_auc']:.4f} (Epoch {best_item['best_epoch']})")

    # parallel_runner 环境下，若未进行训练（全部因样本不足等被跳过），删除空日志文件
    if os.getenv("PARALLEL_RUNNER") == "1" and not training_executed:
        log_file = os.getenv("LOG_PARALLEL_FILE")
        if log_file:
            try:
                Path(log_file).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
