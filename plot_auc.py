import re
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_layer_auc(log_file_path, method_name, desktop_path, input_all_layers_auc=None):
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Search for pattern like "entropy: AUC=0.5035"
    if method_name not in ["MLP", "Autoregressive Transformer", "LSTM", "Transformer"]:
        pattern = re.compile(rf"\b{re.escape(method_name)}:\s*AUC=([\d\.]+)")
    else:
        pattern = re.compile(r"最佳验证AUC: ([\d\.]+)")
    matches = pattern.findall(content)
    
    # Retain the first 37 records as layer0-layer36
    if len(matches) > 37:
        matches = matches[:37]
    
    aucs = [float(x) for x in matches]
    layers = list(range(len(aucs)))
    
    if not aucs:
        print(f"No AUC data found for method: {method_name}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(layers, aucs, marker='o', label=method_name)
    plt.xlabel('Layer')
    plt.ylabel('AUC')
    plt.title(f'{method_name} AUC-Layer Curve')
    plt.grid(True, linestyle='--', alpha=0.5)

    if input_all_layers_auc is not None:
    # 添加 input all layers 横线，带标签
        plt.axhline(input_all_layers_auc, color='g', linestyle='--', label=f'input all layers (no pooling, {input_all_layers_auc:.4f})')

    plt.legend()
    plt.tight_layout()
    
    # Save the figure to the desktop
    save_path = os.path.join(desktop_path, f"{method_name}_auc_layer_curve.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # Removed plt.show() as per user's request to save to desktop

def plot_all_metrics(log_file_path, methods, desktop_path):
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    plt.figure(figsize=(12, 8))
    
    for method_name in methods:
        # Search for pattern like "entropy: AUC=0.5035"
        pattern = re.compile(rf"\b{re.escape(method_name)}:\s*AUC=([\d\.]+)")
        matches = pattern.findall(content)
        
        # Retain the first 37 records as layer0-layer36
        if len(matches) > 37:
            matches = matches[:37]
        
        aucs = [float(x) for x in matches]
        layers = list(range(len(aucs)))
        
        if aucs:
            plt.plot(layers, aucs, marker='o', label=method_name)
        else:
            print(f"No AUC data found for method: {method_name}")
            

    plt.xlabel('Layer')
    plt.ylabel('AUC')
    plt.title('Logits-based Methods AUC-Layer Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to the desktop
    save_path = os.path.join(desktop_path, "Logits-based_combined_auc_layer_curve.png")
    plt.savefig(save_path)
    print(f"Combined plot saved to {save_path}")

def plot_all_metrics_2(log_file_path, methods, desktop_path, input_all_layers_auc=None):
    plt.figure(figsize=(12, 8))
    
    for i in range(len(methods)):
        with open(log_file_path[i], 'r', encoding='utf-8') as f:
            content = f.read()
        # Search for pattern like "entropy: AUC=0.5035"
        pattern = re.compile(r"最佳验证AUC: ([\d\.]+)")
        matches = pattern.findall(content)
        
        # Retain the first 37 records as layer0-layer36
        if len(matches) > 37:
            matches = matches[:37]
        
        aucs = [float(x) for x in matches]
        layers = list(range(len(aucs)))
        
        if aucs:
            line, = plt.plot(layers, aucs, marker='o', label=methods[i])
            color = line.get_color()
        else:
            print(f"No AUC data found for method: {methods[i]}")
            color = None
        if input_all_layers_auc is not None:
            # 添加 input all layers 横线，带标签
            if color:
                plt.axhline(input_all_layers_auc[i], color=color, linestyle='--', label=f'{methods[i]} input all layers (no pooling, {input_all_layers_auc[i]:.4f})')
            else:
                plt.axhline(input_all_layers_auc[i], linestyle='--', label=f'{methods[i]} input all layers (no pooling, {input_all_layers_auc[i]:.4f})')

    plt.xlabel('Layer')
    plt.ylabel('AUC')
    plt.title('Classifier-based Methods AUC-Layer Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to the desktop
    save_path = os.path.join(desktop_path, "Classifier-based_combined_auc_layer_curve.png")
    plt.savefig(save_path)
    print(f"Combined plot saved to {save_path}")

if __name__ == "__main__":
    base_dir = "D:\\OneDrive\\文档\\科研\\MCOT\\Vertival Flow\\Experiment\\mul_num2len5_checkall_Qwen3-8B\\"
    log_files = [base_dir + "lstm.log", base_dir + "mlp.log", base_dir + "transformer.log", base_dir + "ar_transformer.log"]
    # methods = ["entropy", "perplexity", "max_prob", "eubhd"]
    methods = ["LSTM", "MLP", "Transformer", "Autoregressive Transformer"]
    
    # Get desktop path
    desktop_path = base_dir
    
    # Plot individual metrics
    for log_file, method, value in zip(log_files, methods, [0.7533, 0.7705, 0.7628, 0.7648]):
        print(f"Plotting for {method}...")
        plot_layer_auc(log_file, method, desktop_path, value)
        
    # Plot combined metrics
    print("Plotting combined metrics...")
    plot_all_metrics_2(log_files, methods, desktop_path, [0.7533, 0.7705, 0.7628, 0.7648])

