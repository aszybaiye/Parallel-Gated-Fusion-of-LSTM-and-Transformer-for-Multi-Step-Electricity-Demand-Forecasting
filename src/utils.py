import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import torch

def calculate_energy_metrics(y_true, y_pred):
    """
    Calculate energy-specific metrics:
    - Peak Load Error (PLE): Relative error of the maximum value.
    - Ramp Rate Accuracy (RRA): 1 - MAPE of the first differences (ramp rates).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    epsilon = 1e-8

    # Peak Load Error
    max_true = np.max(y_true)
    max_pred = np.max(y_pred)
    ple = np.abs(max_true - max_pred) / (np.abs(max_true) + epsilon)

    # Ramp Rate
    diff_true = np.diff(y_true, axis=0)
    diff_pred = np.diff(y_pred, axis=0)
    
    # RRA = 1 - mean(abs(diff_true - diff_pred) / (abs(diff_true) + epsilon))
    # Or simply MAE of ramp rates
    ramp_mae = np.mean(np.abs(diff_true - diff_pred))
    
    return {
        "PLE": ple,
        "Ramp_MAE": ramp_mae
    }

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标: MSE, RMSE, MAE, R2, sMAPE, PLE, Ramp_MAE
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    epsilon = 1e-8
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
    
    energy_metrics = calculate_energy_metrics(y_true, y_pred)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "sMAPE": smape
    }
    metrics.update(energy_metrics)
    
    return metrics

def apply_publication_style():
    """
    Apply scientific publication style to matplotlib.
    Matches the style in plotting.py for consistency.
    """
    # Use seaborn style as base
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "0.15"})
    
    # Custom rcParams for high-quality figures
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'grid.alpha': 0.4,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# Apply style on import
apply_publication_style()

def count_parameters(model):
    """
    Count trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_figure(fig, save_path):
    """Helper to save figure in PNG format."""
    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Save as PNG
    fig.savefig(os.path.join(base_dir, f"{base_name}.png"), bbox_inches='tight', dpi=300)
    print(f"Saved figure: {base_name} (.png)")

def plot_loss(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线
    """
    if not train_losses or not val_losses:
        print("Warning: Empty loss lists, skipping plot.")
        return

    fig = plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2.5, color='#4c72b0')
    plt.plot(val_losses, label='Validation Loss', linewidth=2.5, linestyle='--', color='#dd8452')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    sns.despine()
    
    if save_path:
        save_figure(fig, save_path)
        plt.close()
    else:
        plt.show()

def plot_predictions(y_true, y_pred, title="Predictions vs True Values", save_path=None, sample_size=200):
    """
    绘制预测值与真实值对比图
    """
    fig = plt.figure(figsize=(12, 6))
    
    # 为了清晰展示，只绘制最后 sample_size 个点
    if len(y_true) > sample_size:
        y_true_plot = y_true[-sample_size:]
        y_pred_plot = y_pred[-sample_size:]
        x_axis = range(len(y_true) - sample_size, len(y_true))
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        x_axis = range(len(y_true))
        
    plt.plot(x_axis, y_true_plot, label='True Values', alpha=0.8, color='black', linewidth=2.5)
    plt.plot(x_axis, y_pred_plot, label='Predictions', alpha=0.8, linestyle='--', color='#d62728', linewidth=2)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Electricity Demand')
    plt.legend()
    sns.despine()
    
    if save_path:
        save_figure(fig, save_path)
        plt.close()
    else:
        plt.show()

def save_metrics(metrics, save_path):
    """
    保存指标到CSV
    """
    df = pd.DataFrame([metrics])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, index=False)

def set_seed(seed=42):
    """
    设置随机种子以保证可复现性
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
