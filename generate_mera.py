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
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===========================
# 配置参数 (Defaults)
# ===========================
DEFAULT_MODEL_NAME = "Models/Qwen3-8B"
DEFAULT_DATA_PATH = 'MCOT/Vertical_Flow/num2len5-10000.pkl'
DEFAULT_SIGN = 'mul'
MAX_NEW_TOKENS = 25

class HDF5IncrementalWriter:
    """按位置增量写入 token 结果到 HDF5，避免整文件重写。"""
    def __init__(self, path, compression=None, compression_opts=None):
        self.path = path
        self.compression = compression
        self.compression_opts = compression_opts
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _append_dataset(self, group, name, data, dtype=None):
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
        with h5py.File(self.path, "a") as hf:
            samples_group = hf.require_group("samples")
            sample_name = f"sample_{sample_idx}"
            sample_group = samples_group.require_group(sample_name)
            input_ids_array = np.asarray(input_ids, dtype=np.int32)
            if "input_ids" in sample_group:
                del sample_group["input_ids"]
            sample_group.create_dataset("input_ids", data=input_ids_array)

    def append_batch(self, batch_token_results):
        if not batch_token_results:
            return
        with h5py.File(self.path, "a") as hf:
            positions_group = hf.require_group("all_token_results")
            for pos_key, pos_data in batch_token_results.items():
                pos_name = f"pos_{pos_key}" if isinstance(pos_key, int) else str(pos_key)
                pos_group = positions_group.require_group(pos_name)
                if "original_key" not in pos_group.attrs:
                    pos_group.attrs["original_key"] = pos_key if isinstance(pos_key, str) else f"int:{pos_key}"

                feats = pos_data.get("flows", [])
                if feats:
                    feats_array = np.stack(feats)
                    self._append_dataset(pos_group, "flows", feats_array)

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

class MathGenerator:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, device='auto', sign=DEFAULT_SIGN):
        self.model_name = model_name
        self.device = device
        self.sign = sign
        self.op_func = self.get_operator(sign)[0]
        self.op_name = self.get_operator(sign)[1]
        
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            dtype="auto",
            device_map=device,
            do_sample=False,
            temperature=0.0,
        )
        print("Model loaded.")

    @staticmethod
    def _fold_operands(operands, binary_fn, op_desc):
        if len(operands) < 2:
            raise ValueError(f"数据项数字数量不足（至少2个）")
        result = operands[0]
        for value in operands[1:]:
            result = binary_fn(result, value)
        return result

    def get_operator(self, sign):
        operators = {
            'plus': (lambda nums: self._fold_operands(nums, lambda a, b: a + b, "加法"), "+"),
            'mul': (lambda nums: self._fold_operands(nums, lambda a, b: a * b, "乘法"), "*"),
            'sub': (lambda nums: self._fold_operands(nums, lambda a, b: a - b, "减法"), "-"),
            'div': (lambda nums: self._fold_operands(nums, lambda a, b: a / b, "除法"), "/"),
        }
        return operators[sign]

    def format_expression(self, operands):
        return f" {self.op_name} ".join(str(x) for x in operands)

    def get_gen_token_flow(self, hidden_states, gen_token_idx):
        if gen_token_idx == 0:
            phase = hidden_states[0]
            flow = torch.stack([
                layer.detach().float().cpu().squeeze(0)[-1]
                for layer in phase
            ], dim=0)
        else:
            if gen_token_idx >= len(hidden_states):
                return None
            phase = hidden_states[gen_token_idx]
            flow = torch.stack([
                layer.detach().float().cpu().squeeze(0).squeeze(0)
                for layer in phase
            ], dim=0)
        return flow

    def check_all_tokens(self, gen_content_tokens_readable, hidden_states, operands, check_all=True):
        gt_value = self.op_func(operands)
        gt_str = str(gt_value)
        results = []
        gt_idx = 0
        all_gt_correct = True
        
        for gen_idx, token in enumerate(gen_content_tokens_readable):
            if token.strip() in [',', '']:
                continue
            
            flow = self.get_gen_token_flow(hidden_states, gen_idx)
            if flow is None: break
            
            is_extra = (gt_idx >= len(gt_str))
            
            if is_extra:
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
                gt_char = gt_str[gt_idx]
                token_clean = token.strip()
                correct = (token_clean == gt_char)
                
                # DEBUG: Check if token is multiple chars?
                if len(token_clean) > 1 and gt_idx < len(gt_str) - 1:
                     # Fallback check? Or just log?
                     # If token is "12" and gt is "12...", we might need to advance gt_idx more?
                     pass
                     
                results.append({
                    'gen_idx': gen_idx,
                    'gt_idx': gt_idx,
                    'pred': token_clean,
                    'gt_char': gt_char,
                    'correct': correct,
                    'flow': flow,
                    'is_extra': False,
                })
                if not correct:
                    # DEBUG LOG
                    # print(f"DEBUG MISMATCH: Token='{token_clean}' vs GT='{gt_char}'")
                    
                    all_gt_correct = False
                    if not check_all:
                        break
                
                # Assuming 1 token = 1 char. If token > 1 char, this is broken.
                # Let's fix it by advancing gt_idx by length of token?
                # But wait, we only have flow for THIS token.
                # If we produce "12", is the flow for "1" or "12"? It's for the token "12".
                # If the dataset expects per-digit flow, and we produce merged digits, we have a granularity mismatch.
                # For now, let's assume 1-to-1. But print if len > 1.
                
                gt_idx += 1
        
        if not check_all and all_gt_correct and gt_idx == len(gt_str):
            last_gen_idx = results[-1]['gen_idx'] if results else -1
            for gen_idx, token in enumerate(gen_content_tokens_readable):
                if gen_idx <= last_gen_idx: continue
                if token.strip() in [',', '']: continue
                
                flow = self.get_gen_token_flow(hidden_states, gen_idx)
                if flow is None: break
                
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
        
        return results

    def process_dataset(self, data_path, save_path, hooks=None, limit=None):
        """
        Main processing loop.
        hooks: List of hook objects to register during generation.
        """
        print(f"Loading dataset: {data_path}")
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        if limit:
            dataset = dataset[:limit]
        
        print(f"Processing {len(dataset)} samples...")
        
        writer = HDF5IncrementalWriter(save_path)
        batch_token_results = {}
        
        # Register hooks if any
        hook_handles = []
        if hooks:
            print("Registering hooks...")
            for hook in hooks:
                # Assuming hook is (layer_idx, hook_fn) or hook object with layer_idx
                # Adjust for direct hook object from MERAHook
                idx = hook.layer_idx
                handle = self.model.model.layers[idx].register_forward_hook(hook)
                hook_handles.append(handle)

        sample_correct = 0
        token_correct = 0
        token_total = 0

        try:
            for data_idx, data_item in enumerate(tqdm(dataset)):
                operands = list(data_item)
                expr = self.format_expression(operands)
                
                prompt = f"Calculate {expr}. Only output a number."
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                text = text + expr + " = "
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    generate_outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
                
                output_ids = generate_outputs.sequences[0].tolist()
                input_ids = output_ids[:len(model_inputs.input_ids[0])]
                gen_ids = output_ids[len(model_inputs.input_ids[0]):]
                
                gen_content_tokens_readable = [self.tokenizer.decode([idx]) for idx in gen_ids]
                
                token_results = self.check_all_tokens(
                    gen_content_tokens_readable,
                    generate_outputs.hidden_states,
                    operands,
                    check_all=True
                )
                
                # Update stats
                all_correct = all(r['correct'] for r in token_results) if token_results else False
                if all_correct:
                    sample_correct += 1
                token_total += len(token_results)
                token_correct += sum(1 for tr in token_results if tr['correct'])
                
                # Save input_ids
                writer.save_sample_input_ids(data_idx, input_ids)
                
                # Batch results
                for tr in token_results:
                    key = 'extra' if tr.get('is_extra', False) else tr['gt_idx']
                    if key not in batch_token_results:
                        batch_token_results[key] = {
                            'flows': [], 'labels': [], 'preds': [], 'gt_chars': [],
                        }
                    batch_token_results[key]['flows'].append(tr['flow'].numpy())
                    batch_token_results[key]['labels'].append(tr['correct'])
                    batch_token_results[key]['preds'].append(tr['pred'])
                    batch_token_results[key]['gt_chars'].append(tr['gt_char'])
                
                if (data_idx + 1) % 100 == 0:
                    writer.append_batch(batch_token_results)
                    batch_token_results.clear()
            
            # Final flush
            if batch_token_results:
                writer.append_batch(batch_token_results)
                
            # Print Final Stats
            sample_acc = sample_correct / len(dataset) if len(dataset) > 0 else 0
            token_acc = token_correct / token_total if token_total > 0 else 0
            
            print("\n" + "="*30)
            print(f"Evaluation Complete Results ({len(dataset)} samples):")
            print(f"  Sample Accuracy: {sample_acc:.4f} ({sample_correct}/{len(dataset)})")
            print(f"  Token Accuracy:  {token_acc:.4f} ({token_correct}/{token_total})")
            print("="*30 + "\n")
            
            return {
                "sample_acc": sample_acc,
                "token_acc": token_acc,
                "sample_correct": sample_correct,
                "token_correct": token_correct,
                "total_samples": len(dataset),
                "total_tokens": token_total
            }
                
        finally:
            # Remove hooks
            if hook_handles:
                for h in hook_handles:
                    h.remove()
                print("Hooks removed.")

if __name__ == "__main__":
    # Example usage
    gen = MathGenerator()
    gen.process_dataset(DEFAULT_DATA_PATH, "MCOT/Vertical_Flow/results/test_run.h5", limit=10)
