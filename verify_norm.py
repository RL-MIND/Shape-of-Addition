
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_PATH = '/data/wenliuyuan/models/Qwen3-0.6B'
# MODEL_PATH = '/data/wenliuyuan/models/Qwen3-4B-Instruct-2507'
# MODEL_PATH = '/data/wenliuyuan/models/Qwen3-8B'
# MODEL_PATH = '/data/wenliuyuan/models/Qwen3-30B-A3B-Instruct-2507' 

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

def verify_pre_norm_extraction(model, tokenizer, last_layer_normalized):
    """
    验证是否可以通过 Hook 提取最后一层的 Pre-Norm 状态。
    只有当最后一层是 Already Normalized 时，这个提取才有意义（因为 outputs 里缺失了 pre-norm）。
    """
    print(f"\n{'='*80}\n验证 Pre-Norm 残差流提取 (Live Demo)\n{'='*80}")
    
    if not last_layer_normalized:
        print("最后一层输出本身就是 Unnormalized (Needs Norm)，因此直接使用 hidden_states[-1] 即可。")
        print("无需 Hook 提取。")
        return

    prompt = "1 + 1 = "
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    # 注册 Hook
    pre_norm_list = []
    def hook_fn(module, args, output):
        pre_norm_list.append(args[0].detach())
        
    handle = model.model.norm.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
            
        # 获取相关 Tensor
        captured_prenorm = pre_norm_list[0][:, -1, :]
        model_postnorm = outputs.hidden_states[-1][:, -1, :]
        
        # 验证: Norm(Captured) == Model_Output
        target_device = model.model.norm.weight.device
        manual_normed = model.model.norm(captured_prenorm.to(target_device))
        
        diff = (manual_normed.to("cpu") - model_postnorm.to("cpu")).abs().max().item()
        
        print(f"Captured Pre-Norm Shape:  {captured_prenorm.shape}")
        print(f"Diff (Norm(Pre) vs Post): {diff:.6f}")
        
        if diff < 1e-3:
            print(">>> 验证成功! Hook 提取的 Pre-Norm 状态与模型输出完全一致。")
        else:
            print(">>> 验证失败! 提取状态不匹配。")
            
    finally:
        handle.remove()

def main():
    print(f"正在加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    
    print(f"Norm类型: {type(model.model.norm)}")
    
    # 1. 分析层状态
    is_normalized = analyze_model_layers(model, tokenizer)
    
    print(f"\n>>> 检测结论: 模型最后一层输出{'[是 ALREADY NORMALIZED]' if is_normalized else '[需要 NORM]'}")
    
    # 2. 验证提取
    verify_pre_norm_extraction(model, tokenizer, is_normalized)

if __name__ == "__main__":
    main()
