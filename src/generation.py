import torch
import random
import pickle
import json
import logging
import argparse
from utils.cli import str_to_bool
import os
import re
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.flow_utils import extract_position_key



# ===========================
# Configuration
# ===========================


# Device configuration
DEVICE = 'auto'

# Model path

# MODEL_PATH = "path/to/your/model"
MODEL_PATH = "Qwen/Qwen3-4B"

# Dataset path
DATA_PATH = 'data/num3len10-100000.pkl'


# Operator configuration
SIGN = 'plus'  # 'plus', 'mul', 'sub', 'div'

# Generation configuration
MAX_NEW_TOKENS = 25

# How many problems the LLM should solve (flow dataset size)
# Count of valid processed samples; skipped samples are not included
MAX_SAMPLES = 5000

# Comma handling mode:
# 'skip': skip comma tokens and continue with subsequent tokens (original behavior)
# 'abandon': abandon the current problem when a comma is encountered and move to the next
COMMA_HANDLING_MODE = 'abandon'  # 'skip' or 'abandon'

# Whether to check all tokens
# True: evaluate and store every token
# False: stop after the first incorrect token (if all are correct, still check one extra position)
CHECK_ALL_TOKENS = True

# Save results every N processed samples
SAVE_INTERVAL = 200

# Random seed
SEED = 42

# Output path defaults are derived from parsed CLI arguments in default_output_paths().

# Log directory
LOG_DIR = "results/logs/log_generate"


def simplify_model_path(model_path: str) -> str:
    """Return a compact model name for result file paths."""
    name = model_path.rstrip("/").split("/")[-1]
    return re.sub(r"-Instruct-\w+$", "", name)


# ===========================
# Logger setup
# ===========================

def ensure_parent_dir(file_path):
    """Create parent directory for file_path if it does not exist."""
    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)


def setup_logger(log_dir=LOG_DIR):
    """Configure logger with a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generate_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("generate")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers (avoid duplicates)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    return logger


# ===========================
# Operator setup
# ===========================

def _fold_operands(operands, binary_fn, op_desc):
    """Fold a numeric sequence of arbitrary length with the given binary function."""
    if len(operands) < 2:
        raise ValueError(f"Insufficient operands for {op_desc} (at least 2 numbers required)")
    result = operands[0]
    for value in operands[1:]:
        result = binary_fn(result, value)
    return result


def get_operator(sign):
    """Return operator function and symbol for the given sign; supports multi-operand sequences."""
    operators = {
        'plus': (lambda nums: _fold_operands(nums, lambda a, b: a + b, "addition"), "+"),
        'mul': (lambda nums: _fold_operands(nums, lambda a, b: a * b, "multiplication"), "*"),
        'sub': (lambda nums: _fold_operands(nums, lambda a, b: a - b, "subtraction"), "-"),
        'div': (lambda nums: _fold_operands(nums, lambda a, b: a / b, "division"), "/"),
    }
    if sign not in operators:
        raise ValueError(f"Unsupported operator: {sign}")
    return operators[sign]



def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# Utility functions
# ===========================

class CapturePreNorm:
    """
    Context manager to capture inputs to model.model.norm (Final Norm), i.e. pre-norm residual stream.
    """
    def __init__(self, model):
        self.model = model
        self.captured = []
        self.handle = None

    def __enter__(self):
        self.captured = []
        def hook(module, args, output):
            # args[0] is input hidden_state (pre-norm)
            # Move to CPU and detach to save GPU memory
            self.captured.append(args[0].detach().cpu())
        
        # Register hook on final norm layer
        # Assume Qwen-style structure (model.model.norm)
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model") and hasattr(self.model.model.language_model, "norm"):
             target_module = self.model.model.language_model.norm
        elif hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            target_module = self.model.model.norm
        elif hasattr(self.model, "norm"): # Some architectures expose norm at top level
            target_module = self.model.norm
        else:
            # Fallback exploration (for other archs if needed)
            raise AttributeError("Cannot locate Final Norm layer (model.model.norm)")
            
        self.handle = target_module.register_forward_hook(hook)
        return self.captured

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.remove()


class CapturePostAttnResidual:
    """
    Context manager to capture post-attention residual in each Transformer block
    (hidden state after attention + residual, before FFN).
    Hooks each layer's post_attention_layernorm; its input is the post-attn residual.
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.num_layers = None
        self._buffer = []           # Per-layer accumulation within current forward pass
        self.captured_per_step = [] # list[list[Tensor]]; outer=step, inner=layer

    def __enter__(self):
        self.captured_per_step = []
        self._buffer = []
        self.handles = []

        # Locate decoder layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model") and hasattr(self.model.model.language_model, "layers"):
            decoder_layers = self.model.model.language_model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            decoder_layers = self.model.model.layers
        else:
            raise AttributeError("Cannot locate decoder layers (model.model.layers)")

        self.num_layers = len(decoder_layers)

        for layer_idx, layer in enumerate(decoder_layers):
            target_ln = layer.post_attention_layernorm

            def make_hook(idx):
                def hook(module, args, output):
                    # args[0] is post-attn residual (input to post_attention_layernorm)
                    self._buffer.append(args[0].detach().cpu())
                    if len(self._buffer) == self.num_layers:
                        self.captured_per_step.append(list(self._buffer))
                        self._buffer.clear()
                return hook

            handle = target_ln.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        if self._buffer:
            self.captured_per_step.append(list(self._buffer))
            self._buffer.clear()


def get_activation_trace(hidden_states):
    """Get residual-stream activations during the prefill phase."""
    activation_trace = {}
    prompt_seq_len = hidden_states[0][0].shape[1]
    
    for token_idx in range(prompt_seq_len):
        activation = torch.stack([
            layer.detach().float().cpu().squeeze(0)[token_idx] 
            for layer in hidden_states[0]
        ], dim=0)
        activation_trace[token_idx] = {
            "token_idx": token_idx,
            "flow": activation  # shape: (L, hid_dim)
        }
    return activation_trace


def get_gen_token_flow(hidden_states, gen_token_idx, pre_norm_states=None):
    """
    Get activation vectors (post-FFN residual) used to predict the gen_token_idx-th generated token.
    gen_token_idx: 0 = first generated token, 1 = second, etc.
    pre_norm_states: optional list of final-layer pre-norm states from CapturePreNorm
    """
    if gen_token_idx == 0:
        # First generated token: last column from prefill phase
        phase = hidden_states[0]
        flow_layers = [
            layer.detach().float().cpu().squeeze(0)[-1]  # Last column
            for layer in phase
        ]
        
        # If pre-norm captured, replace last layer (phase often lacks final norm output;
        # with output_hidden_states=True, Qwen2 hidden_states[-1] is post-norm — replace with pre-norm)
        if pre_norm_states is not None and len(pre_norm_states) > 0:
            # pre_norm_states[0] is prompt-phase tensor (Batch, Seq, Dim)
            # Take last column
            last_layer_pre_norm = pre_norm_states[0].float().squeeze(0)[-1]
            flow_layers[-1] = last_layer_pre_norm
            
        flow = torch.stack(flow_layers, dim=0)

    else:
        # Later generated tokens: representation from previous decode step
        if gen_token_idx >= len(hidden_states):
            return None
        phase = hidden_states[gen_token_idx]  # Note: no longer +1 here
        flow_layers = [
            layer.detach().float().cpu().squeeze(0).squeeze(0)
            for layer in phase
        ]
        
        # Replace last layer
        if pre_norm_states is not None and gen_token_idx < len(pre_norm_states):
            # pre_norm_states[idx] is decode-step tensor (Batch, 1, Dim)
            last_layer_pre_norm = pre_norm_states[gen_token_idx].float().squeeze(0).squeeze(0)
            flow_layers[-1] = last_layer_pre_norm
            
        flow = torch.stack(flow_layers, dim=0)
    return flow


def get_gen_token_flow_post_attn(post_attn_ctx, gen_token_idx, hidden_states=None):
    """
    Extract flow from post-attn residuals captured by CapturePostAttnResidual.
    
    Args:
        post_attn_ctx: CapturePostAttnResidual instance
        gen_token_idx: index of generated token
        hidden_states: hidden_states from model.generate, used for embedding layer
    Returns:
        flow: shape (num_layers+1, hidden_dim), embedding + per-layer post-attn residuals
    """
    if post_attn_ctx is None:
        return None
    captured = post_attn_ctx.captured_per_step
    if gen_token_idx >= len(captured):
        return None

    step_states = captured[gen_token_idx]  # list of N tensors, one per layer
    flow_layers = []

    # Add embedding layer (same dimensionality as post_ffn flow)
    if hidden_states is not None:
        if gen_token_idx == 0:
            emb = hidden_states[0][0].detach().float().cpu().squeeze(0)[-1]
        elif gen_token_idx < len(hidden_states):
            emb = hidden_states[gen_token_idx][0].detach().float().cpu().squeeze(0).squeeze(0)
        else:
            emb = None
        if emb is not None:
            flow_layers.append(emb)

    # Add per-layer post-attn residual
    if gen_token_idx == 0:
        # Prefill: last token position
        for state in step_states:
            flow_layers.append(state.float().squeeze(0)[-1])
    else:
        # Decode: squeeze to single token
        for state in step_states:
            flow_layers.append(state.float().squeeze(0).squeeze(0))

    flow = torch.stack(flow_layers, dim=0)
    return flow


def compute_velocity(flow):
    """Compute velocity (first-order difference)."""
    return flow[1:] - flow[:-1]


def compute_curvature(flow):
    """Compute curvature (second-order difference)."""
    return compute_velocity(compute_velocity(flow))


def format_expression(operands, op_symbol):
    """Format operands as an expression string like 'a + b + c'."""
    return f" {op_symbol} ".join(str(x) for x in operands)


def parse_operands(data_item, data_idx=None):
    """Validate and return the operand list from a sample."""
    if not isinstance(data_item, (list, tuple)):
        prefix = f"Sample {data_idx}: " if data_idx is not None else ""
        raise ValueError(f"{prefix}expected list/tuple, got {type(data_item)}")
    operands = list(data_item)
    if len(operands) < 2:
        prefix = f"Sample {data_idx}: " if data_idx is not None else ""
        raise ValueError(f"{prefix}at least 2 operands required, got {len(operands)}")
    return operands


def calc_carries_any(*args, op="add"):
    """
    Support addition and multiplication:
    Compute per-digit incoming/outgoing carries for multi-operand add/mul
    (carry for addition, tens-place carry for multiplication).

    Args:
        *args: positive integers to operate on
        op: "add" or "mul"
    Returns:
        incoming_carries: carry from lower digit into each position (LSB always 0)
        outgoing_carries: carry out of each position to higher digit
    LSB is first (ones place at index=0).
    """
    if op not in ("add", "mul"):
        raise ValueError("op must be 'add' or 'mul'")

    if op == "add":
        result = sum(args)
    else:
        result = 1
        for n in args:
            result *= n

    args_strs = [str(n)[::-1] for n in args]  # Reverse strings; LSB first
    res_str = str(result)[::-1]  # Reverse result string; LSB first
    maxlen = len(res_str)
    incoming_carries = []
    outgoing_carries = []
    carry = 0  # Initial carry into LSB

    for i in range(maxlen):
        if op == "add":
            curr = carry
            for num_str in args_strs:
                curr += int(num_str[i]) if i < len(num_str) else 0
            incoming_carries.append(carry)
            next_carry = curr // 10
            outgoing_carries.append(next_carry)
            carry = next_carry
        else:  # Long multiplication: sum digit products where i1+i2+...+in=i
            # For result digit i, sum a1[i1]*a2[i2]*...*an[in] over all index tuples with sum i
            curr = carry
            
            # Use itertools.product for all digit index combinations
            from itertools import product
            
            # All combinations with i1+i2+...+in=i
            total_sum = 0
            # Per-operand index ranges (<= i and within operand length)
            ranges = []
            for num_str in args_strs:
                ranges.append(range(min(i + 1, len(num_str))))
            
            # Iterate all combinations
            for digit_indices in product(*ranges):
                if sum(digit_indices) == i:
                    # Product for this combination
                    product_val = 1
                    for idx, num_str in enumerate(args_strs):
                        product_val *= int(num_str[digit_indices[idx]])
                    total_sum += product_val
            
            curr += total_sum
            incoming_carries.append(carry)
            next_carry = curr // 10
            outgoing_carries.append(next_carry)
            carry = next_carry
    return incoming_carries, outgoing_carries


def get_in_carry_at_pos(operands, pos):
    """
    Compute incoming_carry at the given position for addition (pos=0 is MSB).
    
    Args:
        operands: operand list, e.g. [123, 456, 789]
        pos: position index, 0 = most significant digit
        
    Returns:
        in_carry: incoming carry at that position
    """
    result = sum(operands)
    result_len = len(str(result))
    
    # Compute all carries
    incoming_carries, outgoing_carries = calc_carries_any(*operands, op="add")
    # Reverse lists to match pos index (pos=0 is MSB)
    incoming_carries = incoming_carries[::-1]
    outgoing_carries = outgoing_carries[::-1]
    
    # Per umap_plot_script.py: in_carry comes from outgoing_carry of the lower digit
    if pos + 1 < len(outgoing_carries):
        return outgoing_carries[pos + 1]
    else:
        return 0


def encode_position_counters(counters):
    """Convert position counter dict to serializable form (keys as strings)."""
    encoded = {}
    for k, v in counters.items():
        encoded[str(k)] = {"correct": int(v.get("correct", 0)), "total": int(v.get("total", 0))}
    return encoded


def decode_position_counters(raw):
    """Restore position counters from serialized form; numeric keys back to int."""
    if not raw:
        return {}
    decoded = {}
    for k, v in raw.items():
        key = int(k) if str(k).isdigit() else k
        decoded[key] = {"correct": int(v.get("correct", 0)), "total": int(v.get("total", 0))}
    return decoded


def _infer_samples_from_positions(positions_group):
    """Roughly infer processed sample count from label lengths (minimum length)."""
    lengths = []
    for pos_group in positions_group.values():
        labels = pos_group.get("labels")
        if labels is not None:
            lengths.append(len(labels))
    return min(lengths) if lengths else 0


def save_progress_to_h5(path, sample_total, sample_correct, token_total, token_correct, position_counters):
    """Write progress to HDF5 meta group for resume."""
    ensure_parent_dir(path)
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
    Load existing HDF5 progress.
    Returns: start_idx, sample_total, sample_correct, token_total, token_correct, position_counters
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
        
        # Recompute token-level counts to match file contents
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
            key = extract_position_key(pos_name, pos_group)
            position_counters[key] = {"correct": correct, "total": total}
            token_total += total
            token_correct += correct
        
        if processed_samples == 0:
            processed_samples = _infer_samples_from_positions(positions_group)
        
        return processed_samples, processed_samples, sample_correct, token_total, token_correct, position_counters


def check_all_tokens(gen_content_tokens_readable, hidden_states, operands, op_func, check_all=True, comma_mode='skip', pre_norm_states=None, record_activations=True, post_attn_states=None, hook_point='post_ffn'):
    """
    Check each token for correctness; handle commas/spaces per config until GT ends.
    
    Args:
        gen_content_tokens_readable: list of generated tokens
        hidden_states: model hidden states
        operands: operand list
        op_func: operator function over full operand sequence
        check_all: 
            True - evaluate and store every token
            False - stop at first error (if all correct, still check one extra position)
        comma_mode:
            'skip' - skip comma tokens and continue
            'abandon' - return None on comma (abandon problem)
        pre_norm_states: captured pre-norm state list
        post_attn_states: CapturePostAttnResidual instance (post_attn/both mode)
        hook_point: 'post_ffn', 'post_attn', 'both'
    
    Returns: 
        (results, abandon_reason)
        - results: list of dict; empty if abandon_reason is not None
        - abandon_reason: None if OK, else reason string (e.g. 'space' or 'comma')
    """
    gt_str = str(op_func(operands))
    results = []
    gt_idx = 0
    all_gt_correct = True
    
    # --- Pre-check: abandon immediately on space or comma (abandon mode) ---
    for token in gen_content_tokens_readable:
        # if ' ' in token:
        #     return [], 'space'
        if '+' in token or '-' in token or '=' in token:
            return [], 'operator'
        if token.strip() == ',' and comma_mode == 'abandon':
            return [], 'comma'

    for gen_idx, token in enumerate(gen_content_tokens_readable):
        # Skip comma (only reached when comma_mode == 'skip')
        if token.strip() == ',':
            continue
        
        # Skip empty tokens (e.g. lone \n \t or empty string)
        if token.strip() == '':
            continue
        
        if record_activations:
            flow = None
            if hook_point == 'post_ffn':
                flow = get_gen_token_flow(hidden_states, gen_idx, pre_norm_states=pre_norm_states)
            elif hook_point == 'post_attn':
                flow = get_gen_token_flow_post_attn(post_attn_states, gen_idx, hidden_states)
            elif hook_point == 'both':
                flow_ffn = get_gen_token_flow(hidden_states, gen_idx, pre_norm_states=pre_norm_states)
                flow_attn = get_gen_token_flow_post_attn(post_attn_states, gen_idx, hidden_states)
                if flow_ffn is None or flow_attn is None:
                    flow = None
                else:
                    flow = {'post_ffn': flow_ffn, 'post_attn': flow_attn}
            if flow is None:
                break
        else:
            flow = None
        
        # Whether this is an extra check position (GT already fully checked)
        is_extra = (gt_idx >= len(gt_str))
        
        if is_extra:
            # Extra position: detect spurious digits after GT
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
            })
            break
        else:
            # Normal position: compare to GT
            gt_char = gt_str[gt_idx]
            correct = (token.strip() == gt_char)
            
            results.append({
                'gen_idx': gen_idx,
                'gt_idx': gt_idx,
                'pred': token.strip(),
                'gt_char': gt_char,
                'correct': correct,
                'flow': flow,
                'is_extra': False,
            })
            
            if not correct:
                all_gt_correct = False
                if not check_all:
                    break
            
            gt_idx += 1
    
    # If check_all=False but all GT positions correct, still check one extra position
    if not check_all and all_gt_correct and gt_idx == len(gt_str):
        last_gen_idx = results[-1]['gen_idx'] if results else -1
        for gen_idx, token in enumerate(gen_content_tokens_readable):
            if gen_idx <= last_gen_idx:
                continue
            
            # Skip comma
            if token.strip() == ',':
                continue
            
            # Skip empty token
            if token.strip() == '':
                continue
            
            if record_activations:
                flow = None
                if hook_point == 'post_ffn':
                    flow = get_gen_token_flow(hidden_states, gen_idx, pre_norm_states=pre_norm_states)
                elif hook_point == 'post_attn':
                    flow = get_gen_token_flow_post_attn(post_attn_states, gen_idx, hidden_states)
                elif hook_point == 'both':
                    flow_ffn = get_gen_token_flow(hidden_states, gen_idx, pre_norm_states=pre_norm_states)
                    flow_attn = get_gen_token_flow_post_attn(post_attn_states, gen_idx, hidden_states)
                    if flow_ffn is None or flow_attn is None:
                        flow = None
                    else:
                        flow = {'post_ffn': flow_ffn, 'post_attn': flow_attn}
                if flow is None:
                    break
            else:
                flow = None
            
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
            })
            break
    
    return results, None


def save_results(all_token_results, all_sample_results, path):
    """Save results to file."""
    ensure_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump({
            'all_token_results': all_token_results,
            'all_sample_results': all_sample_results,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


class HDF5IncrementalWriter:
    """Incrementally append per-position token results to HDF5 without rewriting the whole file."""

    def __init__(self, path, compression=None, compression_opts=4):
        self.path = path
        self.compression = compression
        self.compression_opts = compression_opts
        ensure_parent_dir(path)

    def _append_dataset(self, group, name, data, dtype=None):
        """Append batch data to dataset, creating it if needed."""
        data = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
        if name not in group:
            maxshape = (None,) + data.shape[1:]
            group.create_dataset(
                name,
                data=data,
                maxshape=maxshape,
                compression=self.compression,
                compression_opts=self.compression_opts if self.compression == "gzip" else None,
                chunks=True,
            )
        else:
            ds = group[name]
            old_len = ds.shape[0]
            new_len = old_len + data.shape[0]
            ds.resize((new_len,) + ds.shape[1:])
            ds[old_len:new_len] = data

    def append_batch(self, batch_token_results):
        """Append current batch (all_token_results style) to HDF5."""
        if not batch_token_results:
            return
        with h5py.File(self.path, "a") as hf:
            positions_group = hf.require_group("all_token_results")
            for pos_key, pos_data in batch_token_results.items():
                pos_name = f"pos_{pos_key}" if isinstance(pos_key, int) else str(pos_key)
                pos_group = positions_group.require_group(pos_name)
                if "original_key" not in pos_group.attrs:
                    pos_group.attrs["original_key"] = pos_key if isinstance(pos_key, str) else f"int:{pos_key}"

                # Save flows (single hook_point mode)
                feats = pos_data.get("flows", [])
                if feats:
                    feats_array = np.stack(feats)
                    self._append_dataset(pos_group, "flows", feats_array)

                # Save flows_post_ffn / flows_post_attn (both mode)
                for suffix in ('post_ffn', 'post_attn'):
                    key_name = f"flows_{suffix}"
                    feats = pos_data.get(key_name, [])
                    if feats:
                        feats_array = np.stack(feats)
                        self._append_dataset(pos_group, key_name, feats_array)

                # Scalar / string features
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

                # Save sample_indices
                if pos_data.get("sample_indices"):
                    idx_array = np.asarray(pos_data["sample_indices"], dtype=np.int64)
                    self._append_dataset(pos_group, "sample_indices", idx_array, dtype=np.int64)

                # Save carry information
                if pos_data.get("incoming_carries"):
                    incoming_carries_array = np.asarray(pos_data["incoming_carries"], dtype=np.int64)
                    self._append_dataset(pos_group, "incoming_carries", incoming_carries_array, dtype=np.int64)

                if pos_data.get("outgoing_carries"):
                    outgoing_carries_array = np.asarray(pos_data["outgoing_carries"], dtype=np.int64)
                    self._append_dataset(pos_group, "outgoing_carries", outgoing_carries_array, dtype=np.int64)

    def append_samples(self, sample_infos):
        """Append sample-level question / pred / gt records."""
        if not sample_infos:
            return
        with h5py.File(self.path, "a") as hf:
            samples_group = hf.require_group("samples")
            idx_array = np.asarray([info["sample_idx"] for info in sample_infos], dtype=np.int64)
            str_dtype = h5py.string_dtype(encoding="utf-8")
            q_array = np.asarray([info["question"] for info in sample_infos], dtype=str_dtype)
            pred_array = np.asarray([info["pred"] for info in sample_infos], dtype=str_dtype)
            gt_array = np.asarray([info["gt"] for info in sample_infos], dtype=str_dtype)
            self._append_dataset(samples_group, "sample_idx", idx_array, dtype=np.int64)
            self._append_dataset(samples_group, "question", q_array, dtype=str_dtype)
            self._append_dataset(samples_group, "pred", pred_array, dtype=str_dtype)
            self._append_dataset(samples_group, "gt", gt_array, dtype=str_dtype)


def print_accuracy_stats_incremental(position_counters, sample_total, sample_correct, token_total, token_correct, logger):
    """Print incremental accuracy stats from counters (no full lists in memory)."""
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
    """Print accuracy statistics."""
    # Sample-level accuracy
    sample_acc = sum(r['all_correct'] for r in all_sample_results) / len(all_sample_results)
    logger.info(f"Sample Accuracy (all tokens correct): {sample_acc:.4f}")
    
    # Per-position accuracy
    logger.info("\nPer-position accuracy:")
    numeric_keys = sorted([k for k in all_token_results.keys() if isinstance(k, int)])
    for pos in numeric_keys:
        pos_labels = all_token_results[pos]['labels']
        pos_acc = sum(pos_labels) / len(pos_labels)
        logger.info(f"  Position {pos}: {sum(pos_labels)}/{len(pos_labels)} = {pos_acc:.4f}")
    
    # Extra-position accuracy
    if 'extra' in all_token_results:
        pos_labels = all_token_results['extra']['labels']
        pos_acc = sum(pos_labels) / len(pos_labels)
        logger.info(f"  Position [EXTRA]: {sum(pos_labels)}/{len(pos_labels)} = {pos_acc:.4f}")



def balance_dataset_by_carry_and_digit(dataset, pos, logger, carry_ratio=None):
    """
    Balance dataset by (in_carry, digit) at the given position.
    
    Args:
        dataset: original dataset list
        pos: position index (0=MSB; alignment follows get_in_carry_at_pos semantics)
        logger: logger instance
        carry_ratio: per in_carry proportion dict, e.g. {0: 0.25, 1: 0.5, 2: 0.25};
                     None means equal proportions across in_carry values
        
    Returns:
        Filtered dataset
    """
    logger.info(f"\n=== Enabling in_carry & digit balancing (pos={pos}) ===")
    if carry_ratio is not None:
        logger.info(f"Using custom in_carry ratios: {carry_ratio}")
    else:
        logger.info("Using equal in_carry proportions")
    
    # key: (in_carry, digit) -> value: [sample_indices]
    buckets = {} 
    # Also group by in_carry for ratio validation and sampling
    carry_buckets = {}
    skipped_count = 0
    
    for idx, item in enumerate(dataset):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            skipped_count += 1
            continue
        operands = list(item)
        result = sum(operands)
        result_str = str(result)
        result_len = len(result_str)
        
        # Check position validity
        if pos >= result_len:
            # Position beyond result length; skip
            skipped_count += 1
            continue
        
        # Get in_carry
        in_carry = get_in_carry_at_pos(operands, pos)
        
        # Digit at this position
        # Note: pos=0 in get_in_carry_at_pos is typically MSB; we assume result_str[pos]
        # aligns with the same position used for in_carry.
        digit = int(result_str[pos])
        
        key = (in_carry, digit)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(idx)
        
        # Group by in_carry
        if in_carry not in carry_buckets:
            carry_buckets[in_carry] = []
        carry_buckets[in_carry].append(idx)
    
    # Log original distribution
    logger.info(f"Original dataset (in_carry, digit) distribution (pos={pos}):")
    sorted_keys = sorted(buckets.keys())
    for key in sorted_keys:
        c, d = key
        logger.info(f"  (in_carry={c}, digit={d}): {len(buckets[key])} samples")
    
    # Log distribution by in_carry
    logger.info(f"Original dataset in_carry distribution (pos={pos}):")
    actual_carries = sorted(carry_buckets.keys())
    for c in actual_carries:
        logger.info(f"  in_carry={c}: {len(carry_buckets[c])} samples")
        
    if skipped_count > 0:
        logger.info(f"  Skipped {skipped_count} invalid samples")
    
    # Compute balanced sample counts
    if not buckets:
        logger.warning("No valid samples after filtering!")
        return []
    
    # Validate carry_ratio
    if carry_ratio is not None:
        # Ensure carry_ratio keys match actual in_carry values
        ratio_keys = set(carry_ratio.keys())
        actual_keys = set(actual_carries)
        
        if ratio_keys != actual_keys:
            missing_in_ratio = actual_keys - ratio_keys
            extra_in_ratio = ratio_keys - actual_keys
            error_msg = f"carry_ratio keys do not match actual in_carry values in dataset!\n"
            if missing_in_ratio:
                error_msg += f"  Present in dataset but missing from carry_ratio: {missing_in_ratio}\n"
            if extra_in_ratio:
                error_msg += f"  Present in carry_ratio but not in dataset: {extra_in_ratio}\n"
            error_msg += f"  Actual in_carry values in dataset: {actual_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check proportions sum to 1.0
        ratio_sum = sum(carry_ratio.values())
        if not (0.999 <= ratio_sum <= 1.001):  # Allow floating-point tolerance
            error_msg = f"carry_ratio proportions must sum to 1.0, got {ratio_sum}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check all proportions are non-negative
        for k, v in carry_ratio.items():
            if v < 0:
                error_msg = f"carry_ratio proportion for in_carry={k} ({v}) cannot be negative"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.info(f"\ncarry_ratio validation passed")
        
        # Proportion-based sample counts
        # Min count per digit within each in_carry group
        carry_digit_min_counts = {}
        for c in actual_carries:
            # Sample counts per digit under this in_carry
            digit_counts = []
            for key in sorted_keys:
                if key[0] == c:
                    digit_counts.append(len(buckets[key]))
            if digit_counts:
                carry_digit_min_counts[c] = min(digit_counts)
            else:
                carry_digit_min_counts[c] = 0
        
        # Max samples each in_carry can contribute (digit-balanced)
        # Each in_carry has up to 10 digits (0-9), equal count per digit
        carry_max_samples = {}
        for c in actual_carries:
            digits_in_carry = sum(1 for key in sorted_keys if key[0] == c)
            carry_max_samples[c] = carry_digit_min_counts[c] * digits_in_carry
        
        logger.info(f"Max samples per in_carry: {carry_max_samples}")
        
        # Actual sample counts from proportions
        # Satisfy ratio constraints while using as much data as possible
        # Bottleneck: min(samples/ratio) across in_carry groups
        limiting_factors = {}
        for c in actual_carries:
            if carry_ratio[c] > 0:
                limiting_factors[c] = carry_max_samples[c] / carry_ratio[c]
            else:
                limiting_factors[c] = float('inf')  # Zero ratio: not a bottleneck
        
        total_possible = min(limiting_factors.values())
        
        # Target total samples per in_carry
        carry_target_samples = {}
        for c in actual_carries:
            carry_target_samples[c] = int(total_possible * carry_ratio[c])
        
        # Samples per (in_carry, digit) combination
        sample_counts = {}
        for c in actual_carries:
            digits_in_carry = sum(1 for key in sorted_keys if key[0] == c)
            if digits_in_carry > 0:
                per_digit_count = carry_target_samples[c] // digits_in_carry
            else:
                per_digit_count = 0
            for key in sorted_keys:
                if key[0] == c:
                    sample_counts[key] = per_digit_count
        
        logger.info(f"\nProportional sampling:")
        for c in actual_carries:
            logger.info(f"  in_carry={c}: target {carry_target_samples[c]} samples (ratio {carry_ratio[c]})")
        
    else:
        # Original logic: equal count per (in_carry, digit)
        min_count = min(len(indices) for indices in buckets.values())
        logger.info(f"\nForced balancing: {min_count} samples per (in_carry, digit) combination")
        sample_counts = {key: min_count for key in sorted_keys}
    
    # Build balanced subset
    balanced_indices = []
    for key in sorted_keys:
        indices = buckets[key]
        count = sample_counts.get(key, 0)
        # Simple slice
        sampled = indices[:count]
        balanced_indices.extend(sampled)
    
    # Sort by original index to preserve relative order
    balanced_indices.sort()
    
    # Rebuild dataset
    new_dataset = [dataset[i] for i in balanced_indices]
    
    # Shuffle so later truncation (MAX_SAMPLES) is random
    random.shuffle(new_dataset)
    
    logger.info(f"Final balanced dataset size: {len(new_dataset)}")
    logger.info(f"=== in_carry & digit balancing complete ===\n")
    
    return new_dataset



# ===========================
# CLI helpers
# ===========================



def parse_cli_args():
    """Parse command-line overrides while keeping paper-reproduction defaults."""
    parser = argparse.ArgumentParser(
        description="Generate arithmetic answers and save residual-stream activations."
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Hugging Face model path or name.")
    parser.add_argument("--dataset", dest="data_path", default=DATA_PATH, help="Pickle dataset path.")
    parser.add_argument("--sign", default=SIGN, choices=["plus", "mul", "sub", "div"], help="Arithmetic operator.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum generated tokens per sample.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="Number of valid samples to process.")
    parser.add_argument("--comma-handling", choices=["skip", "abandon"], default=COMMA_HANDLING_MODE, help="How to handle comma tokens.")
    parser.add_argument("--check-all-tokens", type=str_to_bool, default=CHECK_ALL_TOKENS, help="Whether to evaluate all answer tokens.")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL, help="Checkpoint interval in processed samples.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--out-dir", default=LOG_DIR, help="Directory for generation logs.")
    parser.add_argument("--output-h5", default=None, help="Explicit HDF5 output path.")
    return parser.parse_args()


# ===========================
# Main
# ===========================

def main():
    args = parse_cli_args()
    model_short_name = simplify_model_path(args.model)
    dataset_stem = Path(args.data_path).stem.split("-")[0]
    results_path_h5 = args.output_h5 or f"results/{args.sign}_{dataset_stem}_{model_short_name}_nocheckall_balance_both.h5"

    # Initialize logger
    logger = setup_logger(args.out_dir)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Log configuration
    logger.info("=" * 50)
    logger.info("Configuration:")
    logger.info(f"  args.model: {args.model}")
    logger.info(f"  args.data_path: {args.data_path}")
    logger.info(f"  args.sign: {args.sign}")
    logger.info(f"  args.max_new_tokens: {args.max_new_tokens}")
    logger.info(f"  args.max_samples: {args.max_samples}")
    logger.info(f"  args.comma_handling: {args.comma_handling}")
    logger.info(f"  args.check_all_tokens: {args.check_all_tokens}")
    logger.info(f"  seed: {args.seed}")
    logger.info(f"  args.save_interval: {args.save_interval}")
    logger.info("=" * 50)

    # Get operator
    op_func, op_name = get_operator(args.sign)

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        output_hidden_states=True,
        dtype="auto",
        device_map=DEVICE,
    )
    logger.info("Model loaded")

    # Load dataset
    logger.info(f"Loading dataset: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)
    logger.info(f"Dataset size: {len(dataset)}")

    # Count operands per sample for dynamic adaptation
    operand_len_counter = {}
    for idx, item in enumerate(dataset):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"Sample {idx} has type {type(item)}, expected list or tuple")
        operand_len_counter[len(item)] = operand_len_counter.get(len(item), 0) + 1
    logger.info(f"Operand count distribution (length:count): {sorted(operand_len_counter.items())}")

    # Paper default: fixed carry-balanced generation for plus tasks.
    if args.sign == 'plus':
        dataset = balance_dataset_by_carry_and_digit(dataset, 4, logger, carry_ratio=None)

    # Initialize result storage (fixed HDF5 output).
    writer = HDF5IncrementalWriter(results_path_h5)
    logger.info(f"Using incremental HDF5 write: {results_path_h5}")
    batch_token_results = {}
    batch_sample_infos = []
    position_counters = {}
    sample_total = sample_correct = 0
    token_total = token_correct = 0
    start_idx = 0
    if os.path.exists(results_path_h5):
        (
            start_idx,
            sample_total,
            sample_correct,
            token_total,
            token_correct,
            position_counters,
        ) = load_progress_from_h5(results_path_h5)
        if start_idx > 0:
            logger.info(f"Found existing HDF5 results; {start_idx} samples processed, appending.")
        else:
            logger.info("HDF5 file found but no valid progress; starting from scratch.")

    logger.info("\nStarting processing...")
    abandoned_samples = []
    data_idx = start_idx
    while sample_total < args.max_samples and data_idx < len(dataset):
        data_item = dataset[data_idx]
        logger.info(f"Sample {data_idx} (valid samples: {sample_total}/{args.max_samples})")
        operands = parse_operands(data_item, data_idx)
        expr = format_expression(operands, op_name)
        gt_value = op_func(operands)

        prompt = f"Calculate {expr}. Only output a number. Don't output commas."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text = text + expr + " = "

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        pre_norm_cm = CapturePreNorm(model)
        post_attn_cm = CapturePostAttnResidual(model)
        with pre_norm_cm as captured_pre_norm, post_attn_cm as captured_post_attn:
            generate_outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        output_ids = generate_outputs.sequences[0].tolist()
        input_ids = output_ids[:len(model_inputs.input_ids[0])]
        gen_ids = output_ids[len(model_inputs.input_ids[0]):]
        eval_ids = gen_ids
        input_content = tokenizer.decode(input_ids, skip_special_tokens=False)
        gen_content = tokenizer.decode(gen_ids, skip_special_tokens=False)
        gen_content_tokens_readable = [tokenizer.decode([idx]) for idx in eval_ids]

        logger.info(f"Q: {expr}, GT: {gt_value}")
        logger.info(f"Generated: {gen_content}")
        logger.info(f"gen_content (Final): {repr(gen_content)}")
        logger.info(f"gen_content_tokens_readable (Eval): {gen_content_tokens_readable}")

        token_results, abandon_reason = check_all_tokens(
            gen_content_tokens_readable,
            generate_outputs.hidden_states,
            operands,
            op_func,
            check_all=args.check_all_tokens,
            comma_mode=args.comma_handling,
            pre_norm_states=captured_pre_norm,
            record_activations=True,
            post_attn_states=captured_post_attn,
            hook_point="both",
        )

        if abandon_reason is not None:
            logger.info(f"  [ABANDONED] reason: {abandon_reason}")
            abandoned_samples.append({
                'data_idx': data_idx,
                'expr': expr,
                'gt': gt_value,
                'reason': abandon_reason,
                'gen_content': gen_content,
            })
            logger.info('-' * 40)
            data_idx += 1
            continue

        op_type = "add" if args.sign == "plus" else args.sign
        incoming_carries, outgoing_carries = calc_carries_any(*operands, op=op_type)
        incoming_carries = incoming_carries[::-1]
        outgoing_carries = outgoing_carries[::-1]

        for tr in token_results:
            if tr.get('is_extra', False):
                tr['incoming_carry'] = -1
                tr['outgoing_carry'] = -1
            else:
                gt_idx = tr['gt_idx']
                tr['incoming_carry'] = outgoing_carries[gt_idx + 1] if gt_idx + 1 < len(outgoing_carries) else 0
                tr['outgoing_carry'] = outgoing_carries[gt_idx] if gt_idx < len(outgoing_carries) else 0

        current_sample_id = sample_total
        all_correct = all(r['correct'] for r in token_results) if token_results else False
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

        logged_first_error = False
        for tr in token_results:
            is_extra = tr.get('is_extra', False)
            if is_extra:
                status = "?" if tr['correct'] else "?"
                logger.info(f"  Position {tr['gt_idx']}: pred='{tr['pred']}' gt='{tr['gt_char']}' {status} [EXTRA]")
                continue
            if (not tr['correct']) and (not logged_first_error):
                status = "?" if tr['correct'] else "?"
                logger.info(f"  Position {tr['gt_idx']}: pred='{tr['pred']}' gt='{tr['gt_char']}' {status}")
                logged_first_error = True

        for tr in token_results:
            key = 'extra' if tr.get('is_extra', False) else tr['gt_idx']
            if key not in batch_token_results:
                batch_token_results[key] = {
                    'labels': [], 'preds': [], 'gt_chars': [], 'sample_indices': [],
                    'incoming_carries': [], 'outgoing_carries': [], 'is_extra': tr.get('is_extra', False),
                    'flows_post_ffn': [], 'flows_post_attn': [],
                }
            batch_token_results[key]['flows_post_ffn'].append(tr['flow']['post_ffn'].numpy())
            batch_token_results[key]['flows_post_attn'].append(tr['flow']['post_attn'].numpy())
            batch_token_results[key]['labels'].append(tr['correct'])
            batch_token_results[key]['preds'].append(tr['pred'])
            batch_token_results[key]['gt_chars'].append(tr['gt_char'])
            batch_token_results[key]['sample_indices'].append(current_sample_id)
            batch_token_results[key]['incoming_carries'].append(tr.get('incoming_carry', -1))
            batch_token_results[key]['outgoing_carries'].append(tr.get('outgoing_carry', -1))

        batch_sample_infos.append({
            "sample_idx": current_sample_id,
            "question": expr,
            "pred": gen_content.strip(),
            "gt": str(gt_value),
        })

        sample_acc = sample_correct / sample_total if sample_total else 0
        token_acc = token_correct / token_total if token_total else 0
        logger.info(f"Sample Accuracy: {sample_acc:.4f}, Token Accuracy: {token_acc:.4f}")
        if 'extra' in position_counters:
            extra_stat = position_counters['extra']
            extra_acc = extra_stat['correct'] / extra_stat['total'] if extra_stat['total'] else 0
            logger.info(f"Extra Position Accuracy: {extra_stat['correct']}/{extra_stat['total']} = {extra_acc:.4f}")
        logger.info('-' * 40)

        if sample_total % args.save_interval == 0:
            writer.append_batch(batch_token_results)
            writer.append_samples(batch_sample_infos)
            batch_token_results.clear()
            batch_sample_infos.clear()
            save_progress_to_h5(
                results_path_h5,
                sample_total,
                sample_correct,
                token_total,
                token_correct,
                position_counters,
            )
            logger.info(f">>> HDF5 checkpoint saved at sample_total {sample_total}")

        data_idx += 1

    if batch_token_results:
        writer.append_batch(batch_token_results)
        writer.append_samples(batch_sample_infos)
        save_progress_to_h5(
            results_path_h5,
            sample_total,
            sample_correct,
            token_total,
            token_correct,
            position_counters,
        )
        logger.info(">>> Final HDF5 flush completed")

    logger.info("\n" + "=" * 50)
    logger.info(f"Experiment done. Total valid samples: {sample_total}")
    logger.info(f"Total abandoned samples: {len(abandoned_samples)}")
    print_accuracy_stats_incremental(position_counters, sample_total, sample_correct, token_total, token_correct, logger)
    logger.info(f"\nHDF5 results stored at {results_path_h5}")

    if abandoned_samples:
        logger.info("\n" + "=" * 50)
        logger.info("Abandoned samples (unhandled tokens):")
        logger.info("=" * 50)
        reason_counts = {}
        for sample in abandoned_samples:
            reason = sample['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        logger.info(f"Abandon reason counts:")
        for reason, count in sorted(reason_counts.items()):
            logger.info(f"  {reason}: {count} samples")
        logger.info("\nDetailed list:")
        for sample in abandoned_samples:
            logger.info(f"  data_idx={sample['data_idx']}, Q={sample['expr']}, GT={sample['gt']}, reason={sample['reason']}, gen_content={repr(sample['gen_content'])}")


if __name__ == "__main__":
    main()
