import torch
import random
import re
import pickle
import json
import logging
import os
import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3ForCausalLM


# ===========================
# 配置参数
# ===========================

# 设备配置
DEVICE = 'auto'

# 模型路径
# MODEL_NAME = "/data0/wenliuyuan/models/Qwen3-0.6B"
MODEL_NAME = "/data/Models/Qwen3-4b"
# MODEL_NAME = "/data0/wenliuyuan/models/Qwen3-8B"
# MODEL_NAME = "/data0/wenliuyuan/models/Qwen3-30B-A3B-Instruct-2507"

# 数据集路径
# DATA_PATH = '/home/wenliuyuan/llm/vertical-flow/dataset/num2len2-10000.pkl'
# DATA_PATH = '/home/wenliuyuan/llm/vertical-flow/dataset/num2len5-10000.pkl'
# DATA_PATH = '/home/wenliuyuan/llm/vertical-flow/dataset/num2len10-10000.pkl'
# DATA_PATH = '/home/wenliuyuan/llm/vertical-flow/dataset/num3len3-10000.pkl'
# DATA_PATH = '/home/wenliuyuan/llm/vertical-flow/dataset/num3len10-10000.pkl'
DATA_PATH = 'VerticalFlow/num3len10-10000.pkl'

# 运算符配置
SIGN = 'plus'  # 'plus', 'mul', 'sub', 'div'

# 生成配置
MAX_NEW_TOKENS = 25

# 开关：是否check所有tokens
# True: 每个token都判断并存储
# False: 只check到第一个错误的token，记录后就停止（但如果全对，还会check后面一位）
CHECK_ALL_TOKENS = True

# 每处理多少个样本保存一次结果
SAVE_INTERVAL = 200

# 输出后端：'hdf5' 或 'pickle'
OUTPUT_BACKEND = "hdf5"  # 默认增量写 HDF5，兼容需要时可改为 'pickle'

# 结果保存路径（按后端）
suffix = "_paritial" if not CHECK_ALL_TOKENS else ""
RESULTS_PATH_PKL = "VerticalFlow/results/" + SIGN + "_" + DATA_PATH.split("/")[-1].split(".")[0].split("-")[0] + "_" + MODEL_NAME.split("/")[-1] + suffix + ".pkl"
RESULTS_PATH_H5 = "VerticalFlow/results/" + SIGN + "_" + DATA_PATH.split("/")[-1].split(".")[0].split("-")[0] + "_" + MODEL_NAME.split("/")[-1] + suffix +"/"+ SIGN + "_" + DATA_PATH.split("/")[-1].split(".")[0].split("-")[0] + "_" + MODEL_NAME.split("/")[-1] + suffix + ".h5"

# 是否在使用 HDF5 时额外导出最终 pickle（可能占用大量内存/磁盘）
ENABLE_FINAL_PICKLE_EXPORT = False

# 日志目录
LOG_DIR = "./VerticalFlow/log/log_generate"


# ===========================
# 初始化 Logger
# ===========================

def setup_logger(log_dir=LOG_DIR):
    """设置 logger，按时间命名日志文件"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 按时间命名日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generate_{timestamp}.log")
    
    # 创建 logger
    logger = logging.getLogger("generate")
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers（避免重复添加）
    logger.handlers.clear()
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger


# ===========================
# 运算符设置
# ===========================

def _fold_operands(operands, binary_fn, op_desc):
    """对任意长度的数字序列按指定二元函数折叠计算。"""
    if len(operands) < 2:
        raise ValueError(f"数据项数字数量不足，无法进行{op_desc}运算（至少需要2个数字）")
    result = operands[0]
    for value in operands[1:]:
        result = binary_fn(result, value)
    return result


def get_operator(sign):
    """根据符号获取运算符函数和名称，支持多数字序列"""
    operators = {
        'plus': (lambda nums: _fold_operands(nums, lambda a, b: a + b, "加法"), "+"),
        'mul': (lambda nums: _fold_operands(nums, lambda a, b: a * b, "乘法"), "*"),
        'sub': (lambda nums: _fold_operands(nums, lambda a, b: a - b, "减法"), "-"),
        'div': (lambda nums: _fold_operands(nums, lambda a, b: a / b, "除法"), "/"),
    }
    if sign not in operators:
        raise ValueError(f"不支持的运算符: {sign}")
    return operators[sign]


# ===========================
# Norm 验证与提取工具
# ===========================

def calculate_logit_diff(hidden_state, lm_head, true_logits, norm_layer=None):
    """
    计算给定 hidden_state 经过 lm_head 后与 true_logits 的差异。
    如果提供了 norm_layer，则先应用 Norm。
    自动处理设备不匹配问题。
    """
    # 1. 移动 hidden_state 到目标模块设备
    if norm_layer:
        target_device = norm_layer.weight.device
        h = hidden_state.to(target_device)
        h = norm_layer(h)
    else:
        h = hidden_state
        
    target_device_head = lm_head.weight.device
    h = h.to(target_device_head)
    
    # 2. 计算 logits
    computed_logits = lm_head(h)
    
    # 3. 移动到 CPU 计算差异
    diff = (computed_logits.to("cpu") - true_logits.to("cpu")).abs().max().item()
    return diff

def analyze_model_layers(model, tokenizer):
    """
    遍历模型每一层，分析 hidden_states 输出的状态（是否已 Normalized）。
    返回最后一层是否 Normalized 的结论。
    """
    print(f"\n{'='*80}\n分析模型各层 Hidden States 输出状态\n{'='*80}")
    
    prompt = "1 + 1 = "
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    # Forward Pass
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        true_logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states
    
    lm_head = model.lm_head
    norm = model.model.norm
    
    print(f"{'Layer':<6} | {'Diff (No Norm)':<15} | {'Diff (With Norm)':<15} | {'RMS':<10} | {'Conclusion'}")
    print("-" * 80)
    
    last_layer_normalized = False
    
    for i, state in enumerate(hidden_states):
        # 取最后一个 token
        h = state[:, -1, :]
        
        # 计算 RMS
        rms = torch.sqrt(h.pow(2).mean()).item()
        
        # 计算差异
        diff_no_norm = calculate_logit_diff(h, lm_head, true_logits, norm_layer=None)
        diff_with_norm = calculate_logit_diff(h, lm_head, true_logits, norm_layer=norm)
        
        # 判断结论
        conclusion = ""
        is_last = (i == len(hidden_states) - 1)
        
        if is_last:
            if diff_no_norm < 0.1 and diff_no_norm < diff_with_norm:
                conclusion = "ALREADY NORMALIZED (SKIP NORM)"
                last_layer_normalized = True
            elif diff_with_norm < 0.1 and diff_with_norm < diff_no_norm:
                conclusion = "NEEDS NORM"
            else:
                conclusion = "AMBIGUOUS"
        else:
            conclusion = f"RMS={rms:.4f}"

        print(f"{i:<6} | {diff_no_norm:<15.4f} | {diff_with_norm:<15.4f} | {rms:<10.4f} | {conclusion}")
        
    return last_layer_normalized

# ===========================
# 工具函数
# ===========================

def get_vertical_flow(hidden_states):
    """获取prefill阶段的vertical flow"""
    vertical_dict = {}
    prompt_seq_len = hidden_states[0][0].shape[1]
    
    for token_idx in range(prompt_seq_len):
        flow = torch.stack([
            layer.detach().float().cpu().squeeze(0)[token_idx] 
            for layer in hidden_states[0]
        ], dim=0)
        vertical_dict[token_idx] = {
            "token_idx": token_idx,
            "flow": flow  # shape: (L, hid_dim)
        }
    return vertical_dict


def get_gen_token_flow(hidden_states, gen_token_idx, prenorm_states=None):
    """
    获取生成阶段第gen_token_idx个新token的vertical flow
    gen_token_idx: 0表示第一个生成的token，1表示第二个，以此类推
    hidden_states[0] 是prefill阶段
    hidden_states[1:] 是生成阶段，每个phase只有1个token
    
    prenorm_states: 如果提供了Hook捕获的Pre-Norm状态列表，则用其最后一层替换hidden_states中的对应层
    """
    if gen_token_idx == 0:
        # 第一个生成 token：取 prefill 阶段最后一列
        phase = hidden_states[0]
        
        # 构建 flow (L, dim)
        layers_data = [
            layer.detach().float().cpu().squeeze(0)[-1]  # 取最后一列
            for layer in phase
        ]
        
        # 如果有 captured prenorm，替换最后一层
        if prenorm_states is not None and len(prenorm_states) > 0:
            # prenorm_states[0] 对应 prefill，shape (1, seq_len, dim)
            # 我们需要最后一列
            pre_norm_tensor = prenorm_states[0].detach().float().cpu().squeeze(0)[-1]
            layers_data[-1] = pre_norm_tensor
            
        flow = torch.stack(layers_data, dim=0)

    else:
        # 后续生成 token：取前一个生成步骤的表示
        # 注意: hidden_states下标从1开始是generated tokens
        # 但我们传入的 gen_token_idx=1 对应 hidden_states[1]
        if gen_token_idx >= len(hidden_states):
            return None
        phase = hidden_states[gen_token_idx]
        
        layers_data = [
            layer.detach().float().cpu().squeeze(0).squeeze(0)
            for layer in phase
        ]
        
        # 替换最后一层
        if prenorm_states is not None and gen_token_idx < len(prenorm_states):
            # prenorm_states[gen_token_idx] 对应第i次generation
            # shape (1, 1, dim) -> squeeze -> (dim)
            pre_norm_tensor = prenorm_states[gen_token_idx].detach().float().cpu().squeeze(0).squeeze(0)
            layers_data[-1] = pre_norm_tensor
            
        flow = torch.stack(layers_data, dim=0)
        
    return flow  # shape: (num_layers, hid_dim)


def compute_velocity(flow):
    """计算速度 (一阶差分)"""
    return flow[1:] - flow[:-1]


def compute_curvature(flow):
    """计算曲率 (二阶差分)"""
    return compute_velocity(compute_velocity(flow))


def format_expression(operands, op_symbol):
    """将操作数格式化为形如 'a + b + c' 的表达式字符串。"""
    return f" {op_symbol} ".join(str(x) for x in operands)


def parse_operands(data_item, data_idx=None):
    """校验并返回样本中的数字列表。"""
    if not isinstance(data_item, (list, tuple)):
        prefix = f"样本 {data_idx}: " if data_idx is not None else ""
        raise ValueError(f"{prefix}数据格式应为 list/tuple，当前为 {type(data_item)}")
    operands = list(data_item)
    if len(operands) < 2:
        prefix = f"样本 {data_idx}: " if data_idx is not None else ""
        raise ValueError(f"{prefix}至少需要2个数字，实际 {len(operands)} 个")
    return operands


def _digits_lsd(num):
    """将数字拆成从低位到高位的数字列表。"""
    return [int(ch) for ch in str(abs(int(num)))[::-1]]


def compute_plus_in_carries_and_column_sums(operands, result_len):
    """计算加法时每一位（按输出顺序）对应的进位和各位数字和。"""
    digit_lists = [_digits_lsd(op) for op in operands]
    max_len = max(max(len(d) for d in digit_lists), result_len)

    carries_lsd = []
    column_sums_lsd = []
    carry = 0

    for i in range(max_len):
        column_sum = sum(d[i] if i < len(d) else 0 for d in digit_lists)
        column_sums_lsd.append(column_sum)
        carries_lsd.append(carry)
        carry = (column_sum + carry) // 10

    while len(carries_lsd) < result_len:
        column_sums_lsd.append(0)
        carries_lsd.append(carry)
        carry = carry // 10

    carries_lsd = carries_lsd[:result_len]
    column_sums_lsd = column_sums_lsd[:result_len]

    return list(reversed(carries_lsd)), list(reversed(column_sums_lsd))


def encode_position_counters(counters):
    """将位置计数字典转为可序列化形式（键转字符串）。"""
    encoded = {}
    for k, v in counters.items():
        encoded[str(k)] = {"correct": int(v.get("correct", 0)), "total": int(v.get("total", 0))}
    return encoded


def decode_position_counters(raw):
    """从序列化形式恢复位置计数，数字键恢复为 int。"""
    if not raw:
        return {}
    decoded = {}
    for k, v in raw.items():
        key = int(k) if str(k).isdigit() else k
        decoded[key] = {"correct": int(v.get("correct", 0)), "total": int(v.get("total", 0))}
    return decoded


def _extract_position_key(pos_name, pos_group):
    """从 HDF5 节点提取原始位置键。"""
    orig = pos_group.attrs.get("original_key")
    if isinstance(orig, bytes):
        orig = orig.decode("utf-8")
    if isinstance(orig, str):
        if orig.startswith("int:"):
            try:
                return int(orig.split(":", 1)[1])
            except Exception:
                return orig
        if orig.isdigit():
            return int(orig)
        return orig
    if isinstance(pos_name, str) and pos_name.startswith("pos_"):
        try:
            return int(pos_name.split("_", 1)[1])
        except Exception:
            return pos_name
    return pos_name


def _infer_samples_from_positions(positions_group):
    """根据 labels 长度粗略推断已处理样本数（使用最短长度）。"""
    lengths = []
    for pos_group in positions_group.values():
        labels = pos_group.get("labels")
        if labels is not None:
            lengths.append(len(labels))
    return min(lengths) if lengths else 0


def save_progress_to_h5(path, sample_total, sample_correct, token_total, token_correct, position_counters):
    """将进度信息写入 HDF5 的 meta 节点，便于续跑。"""
    with h5py.File(path, "a") as hf:
        meta = hf.require_group("meta")
        meta.attrs["processed_samples"] = int(sample_total)
        meta.attrs["sample_correct"] = int(sample_correct)
        meta.attrs["token_total"] = int(token_total)
        meta.attrs["token_correct"] = int(token_correct)
        encoded = json.dumps(encode_position_counters(position_counters), ensure_ascii=False)
        if "position_counters" in meta:
            del meta["position_counters"]
        meta.create_dataset("position_counters", data=np.bytes_(encoded))


def load_progress_from_h5(path):
    """
    读取已存在的 HDF5 进度。
    返回: start_idx, sample_total, sample_correct, token_total, token_correct, position_counters
    """
    if not os.path.exists(path):
        return 0, 0, 0, 0, 0, {}
    
    with h5py.File(path, "r") as hf:
        positions_group = hf.get("all_token_results")
        if positions_group is None or len(positions_group) == 0:
            return 0, 0, 0, 0, 0, {}
        
        processed_samples = sample_correct = token_total = token_correct = 0
        position_counters = {}
        
        meta = hf.get("meta")
        if meta is not None:
            processed_samples = int(meta.attrs.get("processed_samples", 0))
            sample_correct = int(meta.attrs.get("sample_correct", 0))
            token_total = int(meta.attrs.get("token_total", 0))
            token_correct = int(meta.attrs.get("token_correct", 0))
            if "position_counters" in meta:
                raw = meta["position_counters"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    position_counters = decode_position_counters(json.loads(str(raw)))
                except Exception:
                    position_counters = {}
        
        # 重新统计 token 级计数，确保与文件一致
        token_total = 0
        token_correct = 0
        position_counters = {}
        for pos_name, pos_group in positions_group.items():
            labels_ds = pos_group.get("labels")
            if labels_ds is None:
                continue
            labels = np.asarray(labels_ds[:], dtype=np.bool_)
            total = len(labels)
            correct = int(labels.sum())
            key = _extract_position_key(pos_name, pos_group)
            position_counters[key] = {"correct": correct, "total": total}
            token_total += total
            token_correct += correct
        
        if processed_samples == 0:
            processed_samples = _infer_samples_from_positions(positions_group)
        
        return processed_samples, processed_samples, sample_correct, token_total, token_correct, position_counters


def check_all_tokens(gen_content_tokens_readable, hidden_states, operands, op_func, check_all=True, prenorm_states=None, sign=None):
    """
    逐个token判断正确性，跳过逗号，直到正确答案结束
    
    参数:
        gen_content_tokens_readable: 生成的token列表
        hidden_states: 模型输出的hidden states
        operands: 运算数列表
        op_func: 运算符函数，接收完整的数字序列
        check_all: 
            True - 每个token都判断并存储
            False - 只check到第一个错误的token，记录后就停止（但如果全对，还会check后面一位）
        prenorm_states: 捕获的 pre-norm 状态列表
    
    返回: list of dict
    """
    gt_str = str(op_func(operands))
    is_plus = (sign == 'plus') if sign is not None else (SIGN == 'plus')
    true_in_carries = []
    column_sums = []
    if is_plus:
        true_in_carries, column_sums = compute_plus_in_carries_and_column_sums(operands, len(gt_str))
    results = []
    gt_idx = 0
    all_gt_correct = True
    
    for gen_idx, token in enumerate(gen_content_tokens_readable):
        # 跳过逗号和空格
        if token.strip() in [',', '']:
            continue
        
        flow = get_gen_token_flow(hidden_states, gen_idx, prenorm_states=prenorm_states)
        if flow is None:
            break
        
        # 判断是否是额外检测位（gt已经check完了）
        is_extra = (gt_idx >= len(gt_str))
        
        if is_extra:
            # 额外位：检测模型是否输出了多余的数字
            pred_token = token.strip()
            is_digit = pred_token.isdigit()
            correct = not is_digit
            
            results.append({
                'gen_idx': gen_idx,
                'gt_idx': gt_idx,
                'pred': pred_token,
                'gt_char': '<END>',
                'correct': correct,
                'flow': flow,
                'is_extra': True,
                'true_in_carry': float('nan'),
                'pred_in_carry': float('nan'),
            })
            break
        else:
            # 正常位：与gt比较
            gt_char = gt_str[gt_idx]
            correct = (token.strip() == gt_char)

            true_carry = float(true_in_carries[gt_idx]) if (is_plus and gt_idx < len(true_in_carries)) else float('nan')
            if is_plus and gt_idx < len(column_sums) and token.strip().isdigit():
                pred_value = int(token.strip())
                # 预测进位 = (预测数字 - (该位加数之和 mod 10)) mod 10
                pred_carry = float((pred_value - (int(column_sums[gt_idx]) % 10)) % 10)
            else:
                pred_carry = float('nan')
            
            results.append({
                'gen_idx': gen_idx,
                'gt_idx': gt_idx,
                'pred': token.strip(),
                'gt_char': gt_char,
                'correct': correct,
                'flow': flow,
                'is_extra': False,
                'true_in_carry': true_carry,
                'pred_in_carry': pred_carry,
            })
            
            if not correct:
                all_gt_correct = False
                if not check_all:
                    break
            
            gt_idx += 1
    
    # 如果check_all=False但所有gt位置都正确了，还需要check额外一位
    if not check_all and all_gt_correct and gt_idx == len(gt_str):
        last_gen_idx = results[-1]['gen_idx'] if results else -1
        for gen_idx, token in enumerate(gen_content_tokens_readable):
            if gen_idx <= last_gen_idx:
                continue
            if token.strip() in [',', '']:
                continue
            
            flow = get_gen_token_flow(hidden_states, gen_idx, prenorm_states=prenorm_states)
            if flow is None:
                break
            
            pred_token = token.strip()
            is_digit = pred_token.isdigit()
            correct = not is_digit
            
            results.append({
                'gen_idx': gen_idx,
                'gt_idx': len(gt_str),
                'pred': pred_token,
                'gt_char': '<END>',
                'correct': correct,
                'flow': flow,
                'is_extra': True,
                'true_in_carry': float('nan'),
                'pred_in_carry': float('nan'),
            })
            break
    
    return results


def save_results(all_token_results, all_sample_results, path):
    """保存结果到文件"""
    with open(path, "wb") as f:
        pickle.dump({
            'all_token_results': all_token_results,
            'all_sample_results': all_sample_results,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


class HDF5IncrementalWriter:
    """按位置增量写入 token 结果到 HDF5，避免整文件重写。"""

    def __init__(self, path, compression=None, compression_opts=None):
        self.path = path
        self.compression = compression
        self.compression_opts = compression_opts
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _append_dataset(self, group, name, data, dtype=None):
        """将批量数据追加到 dataset，必要时创建。"""
        data = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
        if name not in group:
            maxshape = (None,) + data.shape[1:]
            group.create_dataset(
                name,
                data=data,
                maxshape=maxshape,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=True,
            )
        else:
            ds = group[name]
            old_len = ds.shape[0]
            new_len = old_len + data.shape[0]
            ds.resize((new_len,) + ds.shape[1:])
            ds[old_len:new_len] = data

        
    def save_sample_input_ids(self, sample_idx, input_ids):
        """为每个样本保存 input_ids 到 HDF5 的 samples 组。"""
        with h5py.File(self.path, "a") as hf:
            samples_group = hf.require_group("samples")
            sample_name = f"sample_{sample_idx}"
            sample_group = samples_group.require_group(sample_name)
            input_ids_array = np.asarray(input_ids, dtype=np.int32)
            if "input_ids" in sample_group:
                del sample_group["input_ids"]
            sample_group.create_dataset("input_ids", data=input_ids_array)

    def append_batch(self, batch_token_results):
        """将当前批次的 all_token_results 风格数据追加到 HDF5。"""
        if not batch_token_results:
            return
        with h5py.File(self.path, "a") as hf:
            positions_group = hf.require_group("all_token_results")
            for pos_key, pos_data in batch_token_results.items():
                pos_name = f"pos_{pos_key}" if isinstance(pos_key, int) else str(pos_key)
                pos_group = positions_group.require_group(pos_name)
                if "original_key" not in pos_group.attrs:
                    pos_group.attrs["original_key"] = pos_key if isinstance(pos_key, str) else f"int:{pos_key}"

                # 仅保存 flows
                feats = pos_data.get("flows", [])
                if feats:
                    feats_array = np.stack(feats)
                    self._append_dataset(pos_group, "flows", feats_array)

                # 标量/字符串特征
                if pos_data.get("labels"):
                    labels_array = np.asarray(pos_data["labels"], dtype=np.bool_)
                    self._append_dataset(pos_group, "labels", labels_array)

                if pos_data.get("preds"):
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    preds_array = np.asarray(pos_data["preds"], dtype=str_dtype)
                    self._append_dataset(pos_group, "preds", preds_array, dtype=str_dtype)

                if pos_data.get("gt_chars"):
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    gt_array = np.asarray(pos_data["gt_chars"], dtype=str_dtype)
                    self._append_dataset(pos_group, "gt_chars", gt_array, dtype=str_dtype)

                if pos_data.get("true_in_carry"):
                    tic_array = np.asarray(pos_data["true_in_carry"], dtype=np.float32)
                    self._append_dataset(pos_group, "true_in_carry", tic_array)

                if pos_data.get("pred_in_carry"):
                    pic_array = np.asarray(pos_data["pred_in_carry"], dtype=np.float32)
                    self._append_dataset(pos_group, "pred_in_carry", pic_array)
                # 保存 sample_ids 以便离线按样本精确对齐
                if pos_data.get("sample_ids"):
                    ids_array = np.asarray(pos_data["sample_ids"], dtype=np.int32)
                    self._append_dataset(pos_group, "sample_ids", ids_array)

    def append_sample_input_ids(self, sample_idx, input_ids):
        """将样本的 input_ids 存储到 HDF5 的 samples 组。"""
        with h5py.File(self.path, "a") as hf:
            samples_group = hf.require_group("samples")
            sample_name = f"sample_{sample_idx}"
            sample_group = samples_group.require_group(sample_name)
            input_ids_array = np.asarray(input_ids, dtype=np.int32)
            if "input_ids" in sample_group:
                del sample_group["input_ids"]
            sample_group.create_dataset("input_ids", data=input_ids_array)


def print_accuracy_stats_incremental(position_counters, sample_total, sample_correct, token_total, token_correct, logger):
    """使用计数器打印增量准确率统计，避免存全量列表。"""
    sample_acc = sample_correct / sample_total if sample_total else 0
    token_acc = token_correct / token_total if token_total else 0
    logger.info(f"Sample Accuracy (all tokens correct): {sample_acc:.4f}")
    logger.info(f"Token Accuracy: {token_acc:.4f}")

    logger.info("\nPer-position accuracy:")
    numeric_keys = sorted(k for k in position_counters.keys() if isinstance(k, int))
    for pos in numeric_keys:
        correct = position_counters[pos]["correct"]
        total = position_counters[pos]["total"]
        pos_acc = correct / total if total else 0
        logger.info(f"  Position {pos}: {correct}/{total} = {pos_acc:.4f}")

    if "extra" in position_counters:
        correct = position_counters["extra"]["correct"]
        total = position_counters["extra"]["total"]
        pos_acc = correct / total if total else 0
        logger.info(f"  Position [EXTRA]: {correct}/{total} = {pos_acc:.4f}")


def print_accuracy_stats(all_token_results, all_sample_results, logger):
    """打印准确率统计"""
    # 样本准确率
    sample_acc = sum(r['all_correct'] for r in all_sample_results) / len(all_sample_results)
    logger.info(f"Sample Accuracy (all tokens correct): {sample_acc:.4f}")
    
    # 每个位置的准确率
    logger.info("\nPer-position accuracy:")
    numeric_keys = sorted([k for k in all_token_results.keys() if isinstance(k, int)])
    for pos in numeric_keys:
        pos_labels = all_token_results[pos]['labels']
        pos_acc = sum(pos_labels) / len(pos_labels)
        logger.info(f"  Position {pos}: {sum(pos_labels)}/{len(pos_labels)} = {pos_acc:.4f}")
    
    # 额外位准确率
    if 'extra' in all_token_results:
        pos_labels = all_token_results['extra']['labels']
        pos_acc = sum(pos_labels) / len(pos_labels)
        logger.info(f"  Position [EXTRA]: {sum(pos_labels)}/{len(pos_labels)} = {pos_acc:.4f}")


# ===========================
# 主函数
# ===========================

def main():
    # 初始化 logger
    logger = setup_logger()
    
    # 记录配置
    logger.info("=" * 50)
    logger.info("配置参数:")
    logger.info(f"  MODEL_NAME: {MODEL_NAME}")
    logger.info(f"  DATA_PATH: {DATA_PATH}")
    logger.info(f"  SIGN: {SIGN}")
    logger.info(f"  MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    logger.info(f"  CHECK_ALL_TOKENS: {CHECK_ALL_TOKENS}")
    logger.info(f"  SAVE_INTERVAL: {SAVE_INTERVAL}")
    logger.info("=" * 50)
    
    # 获取运算符
    op_func, op_name = get_operator(SIGN)
    
    # 加载模型
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        dtype="auto",
        device_map=DEVICE,
        do_sample=False,
        temperature=0.0,
    )
    logger.info("模型加载完成")
    
    # 验证是否需要 Hook
    last_layer_normalized = analyze_model_layers(model, tokenizer)
    logger.info(f"模型最后一层检测结果: {'ALREADY NORMALIZED' if last_layer_normalized else 'NEEDS NORM'}")
    
    CAPTURED_PRENORM_STATES = []
    hook_handle = None
    
    if last_layer_normalized:
        logger.info(">>> 启用 Pre-Norm Hook 捕获")
        def prenorm_hook(module, args, output):
            # args[0] is input to norm layer (pre-norm hidden states)
            # 及时移至 CPU 节省显存
            CAPTURED_PRENORM_STATES.append(args[0].detach().cpu()) 
            
        hook_handle = model.model.norm.register_forward_hook(prenorm_hook)
    else:
        logger.info(">>> 无需 Hook (直接使用 output_hidden_states)")
    
    # 加载数据集
    logger.info(f"加载数据集: {DATA_PATH}")
    with open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 统计每条样本的数字个数，便于动态适配
    operand_len_counter = {}
    for idx, item in enumerate(dataset):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"样本 {idx} 类型为 {type(item)}，期望 list 或 tuple")
        operand_len_counter[len(item)] = operand_len_counter.get(len(item), 0) + 1
    logger.info(f"数字数量分布(长度:数量): {sorted(operand_len_counter.items())}")
    
    # 初始化结果存储
    backend = OUTPUT_BACKEND.lower()
    use_pickle = backend == "pickle" or (backend == "hdf5" and ENABLE_FINAL_PICKLE_EXPORT)
    writer = None
    if backend == "hdf5":
        writer = HDF5IncrementalWriter(RESULTS_PATH_H5)
        logger.info(f"使用 HDF5 增量写入: {RESULTS_PATH_H5}")
        if ENABLE_FINAL_PICKLE_EXPORT:
            logger.info("已启用额外最终 pickle 导出，可能占用较多内存/磁盘。")
    else:
        logger.info(f"使用 pickle 全量写入: {RESULTS_PATH_PKL}")

    all_token_results = {} if use_pickle else None
    all_sample_results = [] if use_pickle else None
    batch_token_results = {}

    # 增量统计计数器
    position_counters = {}
    sample_total = sample_correct = 0
    token_total = token_correct = 0

    # 断点续跑（仅 hdf5）
    start_idx = 0
    if backend == "hdf5" and os.path.exists(RESULTS_PATH_H5):
        (
            start_idx,
            sample_total,
            sample_correct,
            token_total,
            token_correct,
            position_counters,
        ) = load_progress_from_h5(RESULTS_PATH_H5)
        if start_idx > 0:
            logger.info(f"检测到已有 HDF5 结果，已处理 {start_idx} 个样本，将继续追加。")
        else:
            logger.info("检测到 HDF5 文件但未能获取有效进度，将从头开始。")
    
    # 主循环
    logger.info("\n开始处理...")
    for data_idx in range(start_idx, len(dataset)):
        data_item = dataset[data_idx]
        logger.info(f"Sample {data_idx}")
        operands = parse_operands(data_item, data_idx)
        expr = format_expression(operands, op_name)
        gt_value = op_func(operands)
        
        # 构建输入
        prompt = f"Calculate {expr}. Only output a number."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        text = text + expr + " = "
        
        # 生成
        CAPTURED_PRENORM_STATES.clear()  # 清空上一轮的捕获
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generate_outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        
        # 解析输出
        output_ids = generate_outputs.sequences[0].tolist()
        input_ids = output_ids[:len(model_inputs.input_ids[0])]
        gen_ids = output_ids[len(model_inputs.input_ids[0]):]
        
        input_content = tokenizer.decode(input_ids, skip_special_tokens=False)
        gen_content = tokenizer.decode(gen_ids, skip_special_tokens=False)
        gen_content_tokens_readable = [tokenizer.decode([idx]) for idx in gen_ids]
        
        # logger.info(repr(input_content))
        logger.info(f"Q: {expr}, GT: {gt_value}")
        logger.info(f"gen_content: {repr(gen_content)}")
        
        # 逐个token判断正确性
        token_results = check_all_tokens(
            gen_content_tokens_readable, 
            generate_outputs.hidden_states, 
            operands, op_func,
            check_all=CHECK_ALL_TOKENS,
            prenorm_states=CAPTURED_PRENORM_STATES,
            sign=SIGN,
        )
        
        # 计算整体正确性
        all_correct = all(r['correct'] for r in token_results) if token_results else False

        # 更新统计计数
        sample_total += 1
        if all_correct:
            sample_correct += 1
        token_total += len(token_results)
        token_correct += sum(1 for tr in token_results if tr['correct'])
        for tr in token_results:
            key = 'extra' if tr.get('is_extra', False) else tr['gt_idx']
            if key not in position_counters:
                position_counters[key] = {'correct': 0, 'total': 0}
            position_counters[key]['total'] += 1
            if tr['correct']:
                position_counters[key]['correct'] += 1
        
        # 打印第一个错误token，extra位全部打印
        logged_first_error = False
        for tr in token_results:
            is_extra = tr.get('is_extra', False)
            if is_extra:
                status = "✓" if tr['correct'] else "✗"
                logger.info(f"  Position {tr['gt_idx']}: pred='{tr['pred']}' gt='{tr['gt_char']}' {status} [EXTRA]")
                continue
            if (not tr['correct']) and (not logged_first_error):
                status = "✓" if tr['correct'] else "✗"
                logger.info(f"  Position {tr['gt_idx']}: pred='{tr['pred']}' gt='{tr['gt_char']}' {status}")
                logged_first_error = True
        
        # 按位置存储结果（写入 pickle 或 HDF5 批缓存）
        target_token_results = all_token_results if use_pickle else batch_token_results
        sample_input_ids = input_ids if (backend == "hdf5" and writer is not None) else None
        for tr in token_results:
            key = 'extra' if tr.get('is_extra', False) else tr['gt_idx']
            if target_token_results is not None:
                if key not in target_token_results:
                    target_token_results[key] = {
                        'flows': [],
                        'labels': [],
                        'preds': [],
                        'gt_chars': [],
                        'is_extra': tr.get('is_extra', False),
                        'true_in_carry': [],
                        'pred_in_carry': [],
                        'sample_ids': [],
                    }
                target_token_results[key]['flows'].append(tr['flow'].numpy())
                target_token_results[key]['labels'].append(tr['correct'])
                target_token_results[key]['preds'].append(tr['pred'])
                target_token_results[key]['gt_chars'].append(tr['gt_char'])
                target_token_results[key]['true_in_carry'].append(tr.get('true_in_carry', float('nan')))
                target_token_results[key]['pred_in_carry'].append(tr.get('pred_in_carry', float('nan')))
                # 记录样本索引，便于后续按 sample_id 对齐
                target_token_results[key]['sample_ids'].append(data_idx)
        
        # 保存样本结果（仅在需要 pickle 导出时保留全量）
        if use_pickle and all_sample_results is not None:
            all_sample_results.append({
                "question": expr,
                'gt': gt_value,
                "token_results": token_results,
                "all_correct": all_correct,
            })
        
        # 保存当前样本的 input_ids（每个样本都保存）
        if backend == "hdf5" and writer is not None:
            writer.save_sample_input_ids(data_idx, input_ids)
        
        # 计算当前准确率
        sample_acc = sample_correct / sample_total if sample_total else 0
        token_acc = token_correct / token_total if token_total else 0
        logger.info(f"Sample Accuracy: {sample_acc:.4f}, Token Accuracy: {token_acc:.4f}")
        
        # 打印extra位统计
        if 'extra' in position_counters:
            extra_stat = position_counters['extra']
            extra_acc = extra_stat['correct'] / extra_stat['total'] if extra_stat['total'] else 0
            logger.info(f"Extra Position Accuracy: {extra_stat['correct']}/{extra_stat['total']} = {extra_acc:.4f}")
        
        logger.info('-' * 40)
        
        # 定期保存
        if (data_idx + 1) % SAVE_INTERVAL == 0:
            if backend == "hdf5" and writer is not None:
                writer.append_batch(batch_token_results)
                batch_token_results.clear()
                writer.append_sample_input_ids(data_idx, sample_input_ids)
                save_progress_to_h5(
                    RESULTS_PATH_H5,
                    sample_total,
                    sample_correct,
                    token_total,
                    token_correct,
                    position_counters,
                )
                logger.info(f">>> HDF5 checkpoint saved at sample {data_idx + 1}")
            elif use_pickle:
                save_results(all_token_results, all_sample_results, RESULTS_PATH_PKL)
                logger.info(f">>> Pickle checkpoint saved at sample {data_idx + 1}")
    
    # 处理尾批
    if backend == "hdf5" and writer is not None and batch_token_results:
        writer.append_batch(batch_token_results)
        batch_token_results.clear()
        save_progress_to_h5(
            RESULTS_PATH_H5,
            sample_total,
            sample_correct,
            token_total,
            token_correct,
            position_counters,
        )
        logger.info(">>> Final HDF5 flush completed")
    elif use_pickle:
        save_results(all_token_results, all_sample_results, RESULTS_PATH_PKL)
        logger.info(">>> Final pickle saved")

    # 最终统计
    logger.info("\n" + "=" * 50)
    logger.info(f"Experiment done. Total samples: {sample_total}")
    if use_pickle and all_token_results is not None and all_sample_results is not None:
        print_accuracy_stats(all_token_results, all_sample_results, logger)
        logger.info(f"\nResults saved to {RESULTS_PATH_PKL}")
    else:
        print_accuracy_stats_incremental(position_counters, sample_total, sample_correct, token_total, token_correct, logger)
        if backend == "hdf5":
            logger.info(f"\nHDF5 results stored at {RESULTS_PATH_H5}")


if __name__ == "__main__":
    main()
