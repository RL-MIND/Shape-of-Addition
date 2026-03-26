"""
模型相关的工具函数，支持不同的模型架构（Qwen3, Phi-3, Gemma-2等）
"""
from typing import Optional, Tuple
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_model_type(model_path: str) -> str:
    """
    检测模型类型，返回 'qwen'、'phi3'或 'gemma3'
    
    Args:
        model_path: 模型路径
    
    Returns:
        模型类型字符串
    """
    model_path_lower = model_path.lower()
    if "gemma-3" in model_path_lower or "gemma3" in model_path_lower:
        return "gemma3"
    elif "phi" in model_path_lower or "phi-3" in model_path_lower:
        return "phi3"
    elif "qwen" in model_path_lower or "Qwen" in model_path_lower:
        return "qwen"
    else:
        # 默认假设为 qwen 类型
        return "qwen"


def apply_chat_template_safe(
    tokenizer: AutoTokenizer,
    messages: list,
    model_type: Optional[str] = None,
) -> str:
    """
    安全地应用 chat template，根据模型类型使用不同的参数
    
    Args:
        tokenizer: tokenizer 实例
        messages: 消息列表
        model_type: 模型类型 ('qwen'、'phi3' 或 'gemma3')，如果为 None 则自动检测
    
    Returns:
        格式化后的文本
    """
    if model_type is None:
        # 尝试从 tokenizer 推断
        model_name = getattr(tokenizer, "name_or_path", "").lower()
        model_type = detect_model_type(model_name)
    
    if model_type == "qwen":
        # Qwen 模型支持 enable_thinking 参数
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        # Phi-3、Gemma-3 和其他模型不支持 enable_thinking
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def get_norm_module(model: AutoModelForCausalLM) -> Optional[nn.Module]:
    """
    从模型中获取归一化层（RMSNorm 或 LayerNorm）
    
    不同模型的归一化层位置可能不同：
    - Qwen3: model.model.norm
    - Phi-3: model.model.norm
    - Gemma-2: model.model.norm
    - Gemma-3: model.model.norm_final 或其他变体
    - 其他: 可能在不同位置
    
    Args:
        model: 语言模型实例
    
    Returns:
        归一化层模块，如果找不到则返回 None
    """
    # 尝试最常见的位置: model.model.norm
    if hasattr(model, "model"):
        # Gemma-3: model.model.language_model.norm
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

    # 尝试顶层位置
    for attr_name in ["norm", "final_norm", "norm_final", "final_layernorm", "ln_f"]:
        norm_module = getattr(model, attr_name, None)
        if norm_module is not None:
            return norm_module
    
    return None


def get_norm_weight_from_model(
    model_path: str, 
    device: torch.device,
    model_type: Optional[str] = None
) -> Tuple[torch.Tensor, float]:
    """
    从模型中提取归一化层的 weight 参数
    
    Args:
        model_path: 模型路径
        device: 设备
        model_type: 模型类型（可选，用于优化加载）
    
    Returns:
        (weight, eps): 归一化层的 weight 和 eps 参数
    """
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
    
    # 不同模型的 eps 参数名称可能不同
    eps = getattr(norm_module, "variance_epsilon", 
                  getattr(norm_module, "eps", 
                          getattr(norm_module, "epsilon", 1e-6)))
    
    del model
    torch.cuda.empty_cache()
    return weight, eps


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    add_unit_offset: bool = False,
) -> torch.Tensor:
    """
    对输入应用 RMSNorm
    
    Args:
        x: 输入张量，shape (..., hidden_dim)
        weight: RMSNorm 的 weight 参数，shape (hidden_dim,)
        eps: 防止除零的小常数
        add_unit_offset: 是否采用 Gemma 风格的单位偏移（True 时输出乘以 1+weight）
    
    Returns:
        归一化后的张量
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if add_unit_offset:
        return x_normed * (1 + weight)
    return x_normed * weight


def setup_prenorm_hook(model: AutoModelForCausalLM) -> Tuple[list, Optional[object]]:
    """
    设置 pre-normalization hook，用于捕获归一化前的 hidden states
    
    某些模型（如 Qwen3、Phi-3）的最后一层 hidden_states 已经过归一化，
    需要用 hook 捕获归一化前的状态
    
    Args:
        model: 语言模型实例
    
    Returns:
        (captured_prenorm, hook_handle): 捕获列表和 hook 句柄
    """
    captured_prenorm = []
    
    def prenorm_hook(module, args, output):
        # args[0] 是 norm 层的输入（pre-norm hidden states）
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
    """
    检测最后一层 hidden_states 是否已经归一化。

    Returns:
        True 表示最后一层已归一化（应使用 pre-norm hook），False 表示需应用 norm。
    """
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

    if diff_no < threshold and diff_no < diff_norm:
        return True
    return False
