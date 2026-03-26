import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import apply_chat_template_safe, setup_prenorm_hook
from probe_data import compute_c_potential, compute_raw_sum


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


def inspect_teacher_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = "1 + 1 = ",
) -> Tuple[int, dict[str, object]]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    config_layers = getattr(config, "num_hidden_layers", None) if config is not None else None
    text_config_layers = getattr(text_config, "num_hidden_layers", None) if text_config is not None else None
    hidden_states_len = None
    forward_error = None

    model_device = getattr(model, "device", None)
    if model_device is None or str(model_device) == "meta":
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

    try:
        model_inputs = tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False, output_hidden_states=True)
        if getattr(outputs, "hidden_states", None) is not None:
            hidden_states_len = len(outputs.hidden_states)
        del outputs
    except Exception as exc:
        forward_error = f"{type(exc).__name__}: {exc}"

    if isinstance(hidden_states_len, int) and hidden_states_len > 0:
        total_layers = hidden_states_len
        source = "forward_hidden_states"
    elif isinstance(text_config_layers, int) and text_config_layers > 0:
        total_layers = text_config_layers + 1
        source = "config.text_config.num_hidden_layers"
    elif isinstance(config_layers, int) and config_layers > 0:
        total_layers = config_layers + 1
        source = "config.num_hidden_layers"
    else:
        raise RuntimeError(
            "Unable to determine teacher layer count from hidden states or config metadata."
        )

    diagnostics = {
        "config_class": config.__class__.__name__ if config is not None else "<missing>",
        "config_num_hidden_layers": config_layers if config_layers is not None else "<missing>",
        "text_config_num_hidden_layers": text_config_layers if text_config_layers is not None else "<missing>",
        "hidden_states_len": hidden_states_len if hidden_states_len is not None else "<missing>",
        "layer_source": source,
        "forward_error": forward_error,
    }
    return total_layers, diagnostics


def resolve_teacher_final_norm_local_index(
    selected_layers: Optional[List[int]],
    total_layers: int,
) -> Optional[int]:
    if total_layers <= 0:
        return None
    final_layer = total_layers - 1
    if selected_layers is None:
        return final_layer
    try:
        return selected_layers.index(final_layer)
    except ValueError:
        return None


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


def teacher_force_extract(
    dataset: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int,
    device: torch.device,
    use_prenorm: bool = False,
    valid_indices: Optional[set[int]] = None,
    model_type: Optional[str] = None,
    selected_layers: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del max_new_tokens
    digit_id_list, digit_val_list = get_digit_token_ids(tokenizer)

    def select_digit(logits: torch.Tensor) -> int:
        digit_logits = logits[:, digit_id_list]
        idx = torch.argmax(digit_logits, dim=1).item()
        return digit_val_list[idx]

    captured_prenorm: list[torch.Tensor] = []
    hook_handle = None
    if use_prenorm:
        captured_prenorm, hook_handle = setup_prenorm_hook(model)

    flows_list: List[np.ndarray] = []
    raw_labels: List[int] = []
    carry_labels: List[float] = []
    gt_digits: List[int] = []
    pred_digits: List[int] = []
    sample_ids: List[int] = []
    pos_ids: List[int] = []

    indices_to_process = [i for i in range(len(dataset)) if valid_indices is None or i in valid_indices]

    model.eval()
    with torch.no_grad():
        for sample_idx in indices_to_process:
            operands = dataset[sample_idx]
            gt_val = sum(operands)
            gt_str = str(gt_val)

            expr = " + ".join(str(x) for x in operands)
            messages = [{"role": "user", "content": f"Calculate {expr}. Only output a number."}]
            prompt = apply_chat_template_safe(tokenizer, messages, model_type) + expr + " = "
            full_text = prompt + gt_str
            input_ids = tokenizer([full_text], return_tensors="pt").input_ids.to(device)
            answer_indices = list(range(input_ids.shape[1] - len(gt_str), input_ids.shape[1]))

            captured_prenorm.clear()
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            max_layer = len(hidden_states) - 1
            layer_indices = selected_layers if selected_layers is not None else list(range(max_layer + 1))

            for pos_idx, seq_idx in enumerate(answer_indices):
                logits = outputs.logits[0, seq_idx - 1, :]
                d_pred = select_digit(logits.unsqueeze(0))

                layer_states = []
                for layer_idx in layer_indices:
                    if use_prenorm and layer_idx == max_layer and captured_prenorm:
                        hidden = captured_prenorm[-1][0, seq_idx - 1, :].float().cpu().numpy()
                    else:
                        hidden = hidden_states[layer_idx][0, seq_idx - 1, :].float().cpu().numpy()
                    layer_states.append(hidden)
                flow = np.stack(layer_states)

                question = " + ".join(str(x) for x in operands)
                raw_sum_val = compute_raw_sum(question, pos_idx)
                if raw_sum_val < 0:
                    continue

                flows_list.append(flow)
                raw_labels.append(raw_sum_val % 10)
                carry_labels.append(float(compute_c_potential(question, pos_idx)))
                gt_digits.append(int(gt_str[pos_idx]))
                pred_digits.append(d_pred)
                sample_ids.append(sample_idx)
                pos_ids.append(pos_idx)

            del outputs
            del input_ids

    if hook_handle is not None:
        hook_handle.remove()

    if not flows_list:
        raise RuntimeError("No samples extracted in teacher-forcing mode.")

    flows_all = np.stack(flows_list, axis=0)
    return (
        flows_all,
        np.asarray(raw_labels, dtype=np.int64),
        np.asarray(carry_labels, dtype=np.float32),
        np.asarray(gt_digits, dtype=np.int64),
        np.asarray(pred_digits, dtype=np.int64),
        np.asarray(sample_ids, dtype=np.int64),
        np.asarray(pos_ids, dtype=np.int64),
    )


def resolve_teacher_cache_path(
    dataset_path: Path,
    model_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    positions: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    use_prenorm: bool = False,
    selected_layers: Optional[List[int]] = None,
    cache_dir: Path = Path("saved_models/teacher_cache"),
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_samples_tag = "all" if max_samples is None else str(max_samples)
    prenorm_tag = "prenorm" if use_prenorm else "postnorm"
    layers_tag = "alllayers" if not selected_layers else "layers" + "-".join(str(layer) for layer in selected_layers)
    name = (
        f"teacher_features_{dataset_path.stem}_{_safe_model_tag(model_name)}_"
        f"pos{_positions_tag(positions)}_seed{seed}_"
        f"tr{train_ratio}_vr{val_ratio}_te{test_ratio}_"
        f"ms{max_samples_tag}_{prenorm_tag}_{layers_tag}.pt"
    )
    return cache_dir / name


def load_or_compute_teacher_features(
    dataset: List[List[int]],
    dataset_path: Path,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_name: str,
    max_new_tokens: int,
    device: torch.device,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    use_prenorm: bool = False,
    valid_indices: Optional[set[int]] = None,
    positions: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    model_type: Optional[str] = None,
    selected_layers: Optional[List[int]] = None,
    cache_dir: Path = Path("saved_models/teacher_cache"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = resolve_teacher_cache_path(
        dataset_path=dataset_path,
        model_name=model_name,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        positions=positions,
        max_samples=max_samples,
        use_prenorm=use_prenorm,
        selected_layers=selected_layers,
        cache_dir=cache_dir,
    )
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        print(f"Loaded teacher features from {cache_path}")
        return (
            payload["flows_all"],
            payload["raw_labels"],
            payload["carry_labels"],
            payload["gt_digits"],
            payload["pred_digits"],
            payload["sample_ids"],
            payload["pos_ids"],
        )

    outputs = teacher_force_extract(
        dataset,
        tokenizer,
        model,
        max_new_tokens,
        device,
        use_prenorm=use_prenorm,
        valid_indices=valid_indices,
        model_type=model_type,
        selected_layers=selected_layers,
    )
    payload = {
        "flows_all": outputs[0],
        "raw_labels": outputs[1],
        "carry_labels": outputs[2],
        "gt_digits": outputs[3],
        "pred_digits": outputs[4],
        "sample_ids": outputs[5],
        "pos_ids": outputs[6],
    }
    tmp_cache_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    with open(tmp_cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_cache_path.replace(cache_path)
    print(f"Saved teacher features to {cache_path}")
    return outputs
