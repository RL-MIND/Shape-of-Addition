"""Model utility helpers for multiple model families (Qwen3, Phi-3, Gemma, and related variants)."""

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_model_type(model_path: str) -> str:
    """Infer the model family from a model path or name."""
    model_path_lower = model_path.lower()
    if "gemma-3" in model_path_lower or "gemma3" in model_path_lower:
        return "gemma3"
    if "phi" in model_path_lower or "phi-3" in model_path_lower:
        return "phi3"
    if "qwen" in model_path_lower:
        return "qwen"
    return "qwen"


def apply_chat_template_safe(
    tokenizer: AutoTokenizer,
    messages: list,
    model_type: Optional[str] = None,
) -> str:
    """Apply a chat template with model-family-specific keyword arguments."""
    if model_type is None:
        model_name = getattr(tokenizer, "name_or_path", "").lower()
        model_type = detect_model_type(model_name)

    if model_type == "qwen":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_norm_module(model: AutoModelForCausalLM) -> Optional[nn.Module]:
    """Return the final normalization module when it can be found."""
    if hasattr(model, "model"):
        lm = getattr(model.model, "language_model", None)
        if lm is not None:
            for attr_name in ["norm", "norm_final", "final_layernorm", "ln_f"]:
                norm_module = getattr(lm, attr_name, None)
                if norm_module is not None:
                    return norm_module

        for attr_name in ["norm", "norm_final", "final_layernorm", "ln_f"]:
            norm_module = getattr(model.model, attr_name, None)
            if norm_module is not None:
                return norm_module

    for attr_name in ["norm", "final_norm", "norm_final", "final_layernorm", "ln_f"]:
        norm_module = getattr(model, attr_name, None)
        if norm_module is not None:
            return norm_module

    return None


def get_norm_weight_from_model(
    model_path: str,
    device: torch.device,
    model_type: Optional[str] = None,
) -> Tuple[torch.Tensor, float]:
    """Load a model and extract the final normalization weight and epsilon."""
    if model_type is None:
        model_type = detect_model_type(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype="auto",
    )

    norm_module = get_norm_module(model)
    if norm_module is None:
        raise RuntimeError(
            "Cannot find normalization layer in model. Checked model.model.language_model.norm, "
            "model.model.[norm|norm_final|final_layernorm|ln_f], and top-level variants."
        )

    weight = norm_module.weight.detach().clone()
    eps = getattr(
        norm_module,
        "variance_epsilon",
        getattr(norm_module, "eps", getattr(norm_module, "epsilon", 1e-6)),
    )

    del model
    torch.cuda.empty_cache()
    return weight, eps


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    add_unit_offset: bool = False,
) -> torch.Tensor:
    """Apply RMSNorm to the last dimension of ``x``."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if add_unit_offset:
        return x_normed * (1 + weight)
    return x_normed * weight


def setup_prenorm_hook(model: AutoModelForCausalLM) -> Tuple[list, Optional[object]]:
    """Register a hook that captures pre-normalization hidden states."""
    captured_prenorm = []

    def prenorm_hook(module, args, output):
        # args[0] is the normalization-layer input, i.e. the pre-norm hidden state.
        captured_prenorm.append(args[0].detach())

    norm_module = get_norm_module(model)
    hook_handle = None
    if norm_module is not None:
        hook_handle = norm_module.register_forward_hook(prenorm_hook)

    return captured_prenorm, hook_handle


def analyze_last_layer_normalized(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = "1 + 1 = ",
    threshold: float = 0.1,
) -> bool:
    """Check whether ``hidden_states[-1]`` is already final-normalized."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        true_logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        return False

    norm = get_norm_module(model)
    if norm is None:
        return False

    last_h = hidden_states[-1][:, -1, :]

    def _logit_diff(h: torch.Tensor, apply_norm: bool) -> float:
        if apply_norm:
            h = norm(h.to(norm.weight.device))
        h = h.to(lm_head.weight.device)
        diff = (lm_head(h).to("cpu") - true_logits.to("cpu")).abs().max().item()
        return float(diff)

    diff_no = _logit_diff(last_h, apply_norm=False)
    diff_norm = _logit_diff(last_h, apply_norm=True)
    return diff_no < threshold and diff_no < diff_norm
