import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import logging
from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from tqdm import tqdm

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
LOG_DIR = Path("MCOT/Vertical_Flow/log/log_classify_mind")
LOGGER = None
ORIGINAL_PRINT = print

# --- 1. 数据配置 ---
DATA_FILE_PATH = 'MCOT/Vertical_Flow/results/mul_num2len5_Qwen3-8B'
BALANCE_DATASET = True           # 是否平衡数据集
BALANCE_BY_POSITION = True       # True: 对每个位置分别平衡, False: 所有位置混合平衡
TEST_SIZE = 0.2                  # 划分验证集比例
SEED = 42                        # 随机种子

# 选择使用哪些位置的数据
POSITION_SELECT = 'all'  # 'all', 'all_no_extra', [0,1,2], 0, 'extra'

# 按位置统计评估指标
EVALUATE_BY_POSITION = True      # 是否按位置分别报告 AUC/Acc/F1

# --- 2. MIND 模型配置 ---
HIDDEN_DIM = 4096        # Qwen3-8B 隐藏层维度
NUM_LAYERS = 37          # Qwen3-8B 层数
INPUT_DIM = HIDDEN_DIM * 2  # last_token_mean + last_mean
DROPOUT = 0.2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
TRAIN_EPOCHS = 100

# --- 3. 计算配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 早停配置
EARLY_STOP_PATIENCE = 20  # 连续多少个 epoch 验证 AUC 未提升则停止

# 模型保存配置
SAVE_MODEL = True
SAVE_DIR = "MCOT/Vertical_Flow/saved_models"
SAVE_NAME = "mind_classifier"  # 保存文件名

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
    
    logger = logging.getLogger("classifier_mind")
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
# ========== 数据加载（复用 baseline.py）==========
# ==========================================

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
            
            all_token_results[pos] = pos_data
            print(f"    位置 {pos}: 加载完成")
    
    return all_token_results


def load_flows_and_labels(file_path, position_select='all'):
    """
    加载 flows 和 labels 数据
    
    Returns:
        flows_all: List[np.ndarray], 每个元素形状 (num_layers, hidden_dim)
        labels_all: List[int], 标签列表 (0 或 1)
        position_indices: List[int], 每个样本对应的位置索引
        selected_positions: List, 实际使用的位置列表
        num_layers: int, 层数
        hidden_dim: int, 隐藏层维度
    """
    file_path = Path(file_path)
    h5_path = file_path.with_suffix('.h5')

    if h5_path.exists() and HAS_H5PY:
        all_token_results = load_data_from_hdf5(h5_path, position_select)
    elif file_path.suffix == '.h5' and HAS_H5PY:
        all_token_results = load_data_from_hdf5(file_path, position_select)
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
    position_indices = []
    pos_to_idx = {pos: i for i, pos in enumerate(selected_positions)}
    
    for pos in selected_positions:
        pos_data = all_token_results[pos]
        flows = pos_data['flows']
        labels = pos_data['labels']
        
        print(f"  位置 {pos}: {len(labels)} 个样本, 正样本 {sum(labels)}, 负样本 {len(labels) - sum(labels)}")
        
        for flow, label in zip(flows, labels):
            flows_all.append(flow)
            labels_all.append(1 if label else 0)
            position_indices.append(pos_to_idx[pos])
    
    # 获取维度信息
    sample_flow = flows_all[0]
    num_layers = sample_flow.shape[0]
    hidden_dim = sample_flow.shape[1]
    
    print(f"\n数据加载完成:")
    print(f"  总样本数: {len(flows_all)}, 正样本 {sum(labels_all)}, 负样本 {len(labels_all) - sum(labels_all)}")
    print(f"  Flow 形状: ({num_layers}, {hidden_dim})")
    
    return flows_all, labels_all, position_indices, selected_positions, num_layers, hidden_dim


def balance_dataset(flows, labels, position_indices, selected_positions, by_position=False):
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
                pos_balanced.append(position_indices[idx])
            
            print(f"  类别 {cls}: 保留 {len(selected_indices)} 个")
        
        # 打乱顺序
        indices = np.random.permutation(len(flows_balanced))
        flows_balanced = [flows_balanced[i] for i in indices]
        labels_balanced = [labels_balanced[i] for i in indices]
        pos_balanced = [pos_balanced[i] for i in indices]
        
        print(f"\n数据平衡完成: 总样本数 {len(flows_balanced)}")
        
    else:
        # 新方法：对每个位置分别平衡
        print(f"\n按位置平衡数据集:")
        flows_balanced = []
        labels_balanced = []
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
                    pos_balanced.append(pos_idx)
            
            print(f"→ 平衡后 {min_count * len(class_counts)} 个")
        
        # 打乱顺序
        indices = np.random.permutation(len(flows_balanced))
        flows_balanced = [flows_balanced[i] for i in indices]
        labels_balanced = [labels_balanced[i] for i in indices]
        pos_balanced = [pos_balanced[i] for i in indices]
        
        print(f"\n数据平衡完成: 总样本数 {len(flows_balanced)}")
    
    return flows_balanced, labels_balanced, pos_balanced


# ==========================================
# ========== MIND 特征提取 ==========
# ==========================================

def extract_mind_features(flows):
    """
    提取 MIND 特征: last_token_mean + last_mean
    
    Args:
        flows: List[np.ndarray], 每个元素形状 (num_layers, hidden_dim)
    
    Returns:
        features: np.ndarray, 形状 (num_samples, hidden_dim * 2)
    """
    print("\n提取 MIND 特征...")
    
    features = []
    for flow in tqdm(flows, desc="特征提取"):
        # last_token_mean: 最后一个 token 的所有层均值
        last_token_mean = flow.mean(axis=0)  # (hidden_dim,)
        
        # last_mean: 最后一层的均值
        last_mean = flow[-1, :]  # (hidden_dim,)
        
        # 拼接
        feature = np.concatenate([last_token_mean, last_mean])  # (hidden_dim * 2,)
        features.append(feature)
    
    features = np.array(features)
    print(f"特征提取完成: {features.shape}")
    
    return features


# ==========================================
# ========== MIND 模型定义 ==========
# ==========================================

class MINDClassifier(nn.Module):
    """MIND 分类器：4层 MLP"""
    
    def __init__(self, input_dim=8192, dropout=0.2):
        super(MINDClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)


# ==========================================
# ========== 训练与评估 ==========
# ==========================================



def evaluate_epoch(model, data_loader, criterion):
    """
    在数据加载器上评估模型
    
    Returns:
        loss, auc, acc, f1
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    loss_avg = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, pos_label=0)
    
    return loss_avg, auc, acc, f1

def train_mind_classifier(features_train, labels_train, features_val, labels_val):
    """
    训练 MIND 分类器，支持早停策略和模型保存
    
    Returns:
        model: 训练好的模型（已加载最佳权重）
        best_val_auc: 最佳验证 AUC
        best_epoch: 达到最佳 AUC 的 epoch
    """
    print("\n" + "="*60)
    print("开始训练 MIND 分类器")
    print("="*60)
    
    # 转换为 Tensor
    X_train = torch.FloatTensor(features_train)
    y_train = torch.LongTensor(labels_train)
    X_val = torch.FloatTensor(features_val)
    y_val = torch.LongTensor(labels_val)
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = MINDClassifier(input_dim=INPUT_DIM, dropout=DROPOUT).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算类别权重（处理不平衡数据）
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(DEVICE)
    print(f"类别权重: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    print(f"训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")
    
    # 初始验证（随机初始化）
    print("\n初始验证 (Epoch [-1]):")
    init_val_loss, init_val_auc, init_val_acc, init_val_f1 = evaluate_epoch(model, val_loader, criterion)
    print(f"  Val Loss: {init_val_loss:.4f} | Val AUC: {init_val_auc:.4f} | Val Acc: {init_val_acc:.4f} | Val F1: {init_val_f1:.4f}")
    
    # 早停相关变量
    best_val_auc = float("-inf")
    best_epoch = -1
    best_state = None
    no_improve_epochs = 0
    
    # 训练循环
    for epoch in range(TRAIN_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss_avg = running_loss / len(train_loader)
        
        # 验证阶段（每个 epoch 后计算 AUC）
        val_loss, val_auc, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion)
        
        # 早停逻辑
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1  # epoch 从 0 开始，输出 1-based
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        print(f"Epoch [{epoch+1}/{TRAIN_EPOCHS}] "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"No Improve: {no_improve_epochs}/{EARLY_STOP_PATIENCE}")
        
        # 检查早停
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n连续 {EARLY_STOP_PATIENCE} 个 epoch 验证 AUC 未提升，提前停止。")
            break
    
    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n训练完成! 已恢复最佳模型权重 (Epoch {best_epoch}, AUC: {best_val_auc:.4f})")
    else:
        print("\n训练完成! 未找到更好的模型（使用当前权重）")
    
    # 保存模型
    if SAVE_MODEL:
        save_dir = Path(SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{SAVE_NAME}_{timestamp}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_type": "MIND_MLP",
            "input_dim": INPUT_DIM,
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "class_weights": class_weights.cpu().numpy(),
        }, save_path)
        print(f"模型已保存至: {save_path}")
    
    return model, best_val_auc, best_epoch


def evaluate_mind_classifier(model, features, labels, position_indices, selected_positions):
    """评估 MIND 分类器"""
    print("\n评估 MIND 分类器...")
    
    model.eval()
    X = torch.FloatTensor(features).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X)
        probs = F.softmax(outputs, dim=1)[:, 1]  # 幻觉概率
        _, predicted = torch.max(outputs, 1)
    
    scores = probs.cpu().numpy()
    preds = predicted.cpu().numpy()
    
    # 整体指标
    auc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label=1)
    
    print("\n" + "=" * 60)
    print("整体评估结果:")
    print(f"  AUC:  {auc:.4f}")
    print(f"  Acc:  {acc:.4f}")
    print(f"  F1:   {f1:.4f}")
    print("=" * 60)
    
    # 按位置统计
    if EVALUATE_BY_POSITION:
        print("\n" + "=" * 60)
        print("按位置统计指标:")
        print("=" * 60)
        
        position_indices = np.array(position_indices)
        labels = np.array(labels)
        
        for idx, pos in enumerate(selected_positions):
            pos_mask = (position_indices == idx)
            if pos_mask.sum() == 0:
                print(f"位置 {pos}: 样本数为 0")
                continue
            
            pos_labels = labels[pos_mask]
            pos_scores = scores[pos_mask]
            pos_preds = preds[pos_mask]
            
            pos_acc = accuracy_score(pos_labels, pos_preds)
            
            if len(np.unique(pos_labels)) > 1:
                pos_auc = roc_auc_score(pos_labels, pos_scores)
                pos_f1 = f1_score(pos_labels, pos_preds, pos_label=1)
                print(f"位置 {pos:>2}: Acc={pos_acc:.4f} | AUC={pos_auc:.4f} | F1={pos_f1:.4f} | 样本数={pos_mask.sum()}")
            else:
                print(f"位置 {pos:>2}: Acc={pos_acc:.4f} | AUC=N/A (只有一个类别) | 样本数={pos_mask.sum()}")
    
    return auc, acc, f1


# ==========================================
# ========== 主程序 ==========
# ==========================================

def main():
    setup_logger()
    
    print("=" * 60)
    print("MIND Classifier (基于内部状态的幻觉检测)")
    print("=" * 60)
    print(f"配置参数:")
    print(f"  数据文件: {DATA_FILE_PATH}")
    print(f"  位置选择: {POSITION_SELECT}")
    print(f"  数据平衡: {BALANCE_DATASET}")
    print(f"  按位置平衡: {BALANCE_BY_POSITION}")
    print(f"  按位置评估: {EVALUATE_BY_POSITION}")
    print(f"  训练轮数: {TRAIN_EPOCHS}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print("=" * 60)
    
    # 1. 加载数据
    flows, labels, position_indices, selected_positions, num_layers, hidden_dim = load_flows_and_labels(
        DATA_FILE_PATH,
        position_select=POSITION_SELECT
    )
    
    # 2. 数据平衡
    if BALANCE_DATASET:
        flows, labels, position_indices = balance_dataset(flows, labels, position_indices, selected_positions, BALANCE_BY_POSITION)
    
    # 转为 numpy 数组
    labels = np.array(labels)
    position_indices = np.array(position_indices)
    
    # 3. 提取 MIND 特征
    features = extract_mind_features(flows)
    
    # 4. 数据划分
    indices = np.arange(len(features))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels
    )
    
    features_train = features[train_idx]
    features_val = features[val_idx]
    labels_train = labels[train_idx]
    labels_val = labels[val_idx]
    pos_train = position_indices[train_idx]
    pos_val = position_indices[val_idx]
    
    print(f"\n训练集: {len(features_train)} 样本")
    print(f"验证集: {len(features_val)} 样本")
    
    # 5. 训练 MIND 分类器
    model, best_val_auc, best_epoch = train_mind_classifier(features_train, labels_train, features_val, labels_val)
    
    # 6. 在验证集上评估
    evaluate_mind_classifier(model, features_val, labels_val, pos_val, selected_positions)


if __name__ == "__main__":
    main()
