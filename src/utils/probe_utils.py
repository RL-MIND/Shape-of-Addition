from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.model_utils import apply_chat_template_safe
from src.utils.probe_data import compute_c_potential, compute_raw_sum


def _positions_tag(positions: Optional[List[int]]) -> str:
    if not positions:
        return "all"
    return "-".join(str(p) for p in positions)


def _safe_model_tag(model_name: str) -> str:
    tag = model_name.replace("\\", "_").replace("/", "_").replace(":", "_")
    return tag.strip("_") or "model"


def normalize_layer_index(layer_idx: int, total_layers: int) -> int:
    if layer_idx < 0:
        layer_idx = total_layers + layer_idx
    return max(0, min(total_layers - 1, layer_idx))


def resolve_selected_layers(layer_indices: Optional[List[int]], total_layers: int) -> Optional[List[int]]:
    if not layer_indices:
        return None
    resolved: List[int] = []
    seen = set()
    for layer_idx in layer_indices:
        normalized = normalize_layer_index(layer_idx, total_layers)
        if normalized not in seen:
            resolved.append(normalized)
            seen.add(normalized)
    return resolved or None


def parse_layer_candidates(
    num_layers: int,
    layers: Optional[List[int]],
    layer_start: Optional[int],
    layer_end: Optional[int],
) -> List[int]:
    """Resolve explicit or ranged layer arguments into valid layer indices."""
    if layers:
        return [normalize_layer_index(int(layer), num_layers) for layer in layers]
    if layer_start is None and layer_end is None:
        return [num_layers - 1]

    start = normalize_layer_index(layer_start if layer_start is not None else num_layers - 1, num_layers)
    end = normalize_layer_index(layer_end if layer_end is not None else start, num_layers)
    if start > end:
        start, end = end, start
    return list(range(start, end + 1))


def get_digit_token_ids(tokenizer: AutoTokenizer) -> Tuple[List[int], List[int]]:
    digit_ids = {}
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {d} is not a single token: {ids}")
        digit_ids[d] = ids[0]
    digit_id_list = [digit_ids[d] for d in sorted(digit_ids.keys())]
    digit_val_list = sorted(digit_ids.keys())
    return digit_id_list, digit_val_list


def online_baseline_eval(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int,
    device: torch.device,
    model_type: Optional[str] = None,
) -> Tuple[float, float]:
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    token_total = 0
    token_correct = 0
    sample_total = 0
    sample_correct = 0

    model.eval()
    with torch.no_grad():
        for operands in dataset:
            gt_val = sum(operands)
            gt_str = str(gt_val)

            expr = " + ".join(str(x) for x in operands)
            messages = [{"role": "user", "content": f"Calculate {expr}. Only output a number."}]
            text = apply_chat_template_safe(tokenizer, messages, model_type)
            text = text + expr + " = "

            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            outputs = model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values

            generated_digits: List[int] = []
            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]
                chosen_digit = select_digit(logits)
                generated_digits.append(chosen_digit)

                next_token_id = digit_id_list[chosen_digit]
                next_input = torch.tensor([[next_token_id]], device=device)
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = outputs.past_key_values

                if len(generated_digits) >= len(gt_str):
                    break

            sample_total += 1
            g_len = len(generated_digits)
            t_len = len(gt_str)
            for i in range(max(g_len, t_len)):
                token_total += 1
                if i < t_len and i < g_len and generated_digits[i] == int(gt_str[i]):
                    token_correct += 1
            if g_len == t_len and all(generated_digits[i] == int(gt_str[i]) for i in range(t_len)):
                sample_correct += 1

    token_acc = token_correct / token_total if token_total else 0.0
    sample_acc = sample_correct / sample_total if sample_total else 0.0
    return token_acc, sample_acc

