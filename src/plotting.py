import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Publication Style Configuration ---
def apply_publication_style():
    """
    Apply style settings for publication-quality figures.
    Based on common practices for top AI conferences (NeurIPS, ICML, CVPR).
    """
    # Use seaborn as a base for better aesthetics
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "0.15"})
    
    # Custom matplotlib parameters for fine-tuning
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix', # Use STIX font for math to match Times
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,       # High DPI for display/PNG
        'savefig.dpi': 300,      # High DPI for saved files
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'grid.alpha': 0.4,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'pdf.fonttype': 42,      # Type 42 (TrueType) ensures fonts are embedded and editable
        'ps.fonttype': 42
    })

# Apply the style immediately when the module is imported
apply_publication_style()

# Define a consistent color palette (Colorblind friendly)
COLORS = {
    'PGF-Net': '#d62728',       # Red
    'DLinear': '#2ca02c',       # Green
    'PatchTST': '#ff7f0e',      # Orange
    'LSTM': '#9467bd',          # Purple
    'S-Naive': '#7f7f7f',       # Gray
    'Ground Truth': 'black',
    'History': 'gray',
    'Train': '#4c72b0',         # Blue
    'Val': '#dd8452'            # Orange-Red
}

def save_figure(save_dir, filename):
    """Helper to save figure in PNG format."""
    base_name = os.path.splitext(filename)[0]
    # Save as PNG (high resolution)
    plt.savefig(os.path.join(save_dir, f"{base_name}.png"), bbox_inches='tight', dpi=300)
    print(f"Saved figure: {base_name} (.png)")

def _pick_case_indices(n_samples, desired_cases=5):
    """Pick evenly spaced valid sample indices to avoid empty subplot panels."""
    if n_samples <= 0:
        return []
    n_cases = min(desired_cases, n_samples)
    if n_cases == 1:
        return [0]
    return np.linspace(0, n_samples - 1, n_cases, dtype=int).tolist()

def plot_horizon_mse_comparison(predictions_by_model, y_test_inv, output_seq_len, save_dir, use_log_scale=True):
    """
    Plot MSE vs Forecast Horizon for all models.
    """
    print("Plotting Horizon MSE Comparison...")
    fig = plt.figure(figsize=(10, 6))
    
    # Ensure y_test_inv is (N, 24) or (N, 24, 1)
    y_true = y_test_inv
    if y_true.ndim == 3:
        y_true = np.squeeze(y_true)
    
    # Use a distinct marker for each model if needed, or just circle
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (name, preds_list) in enumerate(predictions_by_model.items()):
        mse_runs = []
        for preds in preds_list:
            y_pred = preds
            if y_pred.ndim == 3:
                y_pred = np.squeeze(y_pred)
            mse_per_step = []
            for step in range(output_seq_len):
                mse = np.mean((y_pred[:, step] - y_true[:, step]) ** 2)
                mse_per_step.append(mse)
            mse_runs.append(mse_per_step)
        mse_runs = np.array(mse_runs)
        mse_mean = mse_runs.mean(axis=0)
        mse_std = mse_runs.std(axis=0)
        x_axis = range(1, output_seq_len + 1)
        
        color = COLORS.get(name, None) # Auto color if not in dict
        marker = markers[i % len(markers)]
        
        plt.plot(x_axis, mse_mean, marker=marker, label=name, color=color, markeredgecolor='white', markeredgewidth=1.0)
        plt.fill_between(x_axis, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2, color=color)
        
    plt.xlabel("Forecast Horizon (Hours)")
    plt.ylabel("MSE (Mean Squared Error)")
    plt.title("Multi-step Forecast Error (MSE vs. Horizon)")
    if use_log_scale:
        plt.yscale("log")
        plt.ylabel("MSE (Log Scale)")
        
    plt.legend(loc='best')
    
    # Remove top and right spines for cleaner look
    sns.despine()
    
    save_figure(save_dir, "horizon_mse_comparison.png")
    plt.close()

def plot_gate_analysis(gate_values_list, save_dir, load_series=None, hour_series=None):
    """
    Plot Gate Activation Analysis for PGF-Net.
    """
    if not gate_values_list:
        return

    print("Plotting Gate Analysis...")
    # gate_values_list is list of arrays (batch, d_model) or similar
    all_gates = np.concatenate(gate_values_list, axis=0) # (N, d_model)
    results_dir = save_dir
    eps = 1e-8
    
    # Average gate value across all test samples for each feature dimension
    avg_gate = np.mean(all_gates, axis=0)
    sorted_indices = np.argsort(avg_gate)
    sorted_gate = avg_gate[sorted_indices]
    gate_mean_per_origin = np.mean(all_gates, axis=1)

    quantiles = np.quantile(all_gates, [0.05, 0.25, 0.5, 0.75, 0.95])
    entropy = -(
        all_gates * np.log(all_gates + eps) +
        (1.0 - all_gates) * np.log(1.0 - all_gates + eps)
    )
    bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9])
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    flat_gates = all_gates.reshape(-1)
    bin_ids = np.digitize(flat_gates, bin_edges) - 1
    bin_rows = []
    for i, label in enumerate(bin_labels):
        mask = bin_ids == i
        bin_rows.append({
            "bin": label,
            "count": int(mask.sum()),
            "proportion": float(mask.mean()) if len(mask) else 0.0
        })
    pd.DataFrame([{
        "mean": float(np.mean(all_gates)),
        "std": float(np.std(all_gates)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy))
    }]).to_csv(os.path.join(results_dir, "gate_statistics_summary.csv"), index=False)
    pd.DataFrame(bin_rows).to_csv(os.path.join(results_dir, "gate_bin_proportions.csv"), index=False)
    if load_series is not None:
        load_subset = np.asarray(load_series[:len(gate_mean_per_origin)], dtype=float)
        ramp_subset = np.abs(np.diff(load_subset, prepend=load_subset[0]))
        corr_rows = [
            {
                "signal": "load_level",
                "pearson_r": float(np.corrcoef(gate_mean_per_origin, load_subset)[0, 1])
            },
            {
                "signal": "abs_ramp",
                "pearson_r": float(np.corrcoef(gate_mean_per_origin, ramp_subset)[0, 1])
            }
        ]
        pd.DataFrame(corr_rows).to_csv(os.path.join(results_dir, "gate_signal_correlations.csv"), index=False)
    
    # 1. Sorted Bar Plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(sorted_gate)), sorted_gate, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.8)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Balanced (0.5)', linewidth=2)
    plt.xlabel("Feature Dimension (Sorted by Gate Value)")
    plt.ylabel("Gate Value (0=Trans, 1=LSTM)")
    plt.title("Adaptive Fusion Mechanism: Gate Activation Distribution")
    plt.legend()
    sns.despine()
    save_figure(save_dir, "gate_analysis_sorted.png")
    plt.close()
    
    # 2. Temporal Analysis (First few dimensions)
    plt.figure(figsize=(12, 5))
    # Plot first 3 dimensions (sorted by variance or just first 3)
    # Let's plot the ones with highest variance to show dynamic behavior
    variances = np.var(all_gates, axis=0)
    top_var_indices = np.argsort(variances)[-3:]
    
    # Plot a subset of time (e.g., 200 steps)
    subset_len = min(200, all_gates.shape[0])
    t_axis = range(subset_len)
    
    colors = sns.color_palette("viridis", len(top_var_indices))
    
    for i, idx in enumerate(top_var_indices):
        plt.plot(t_axis, all_gates[:subset_len, idx], label=f'Dim {idx} (High Var)', color=colors[i], alpha=0.9)
        
    if hour_series is not None:
        hour_subset = np.array(hour_series[:subset_len])
        night_mask = (hour_subset <= 6) | (hour_subset >= 20)
        # Add shaded regions for night
        # We need to find contiguous regions
        is_night = False
        start_t = 0
        for t in range(subset_len):
            if night_mask[t] and not is_night:
                start_t = t
                is_night = True
            elif not night_mask[t] and is_night:
                plt.axvspan(start_t - 0.5, t - 0.5, color='gray', alpha=0.1, lw=0)
                is_night = False
        if is_night: # If ends in night
             plt.axvspan(start_t - 0.5, subset_len - 0.5, color='gray', alpha=0.1, lw=0)
        
    plt.xlabel("Consecutive Forecast Origins")
    plt.ylabel("Gate Value")
    plt.title("Dynamic Gating Across Consecutive Forecast Origins")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    sns.despine()
    save_figure(save_dir, "gate_analysis_temporal.png")
    plt.close()

    # 3. Gate Heatmap (New Requirement)
    # Plot heatmap for a 24-hour window (or 48) to show daily pattern
    # Assuming hourly data, take first 48 steps
    heatmap_len = min(48, all_gates.shape[0])
    heatmap_data = all_gates[:heatmap_len, :].T
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})
    if load_series is not None:
        load_subset = np.array(load_series[:heatmap_len])
        ax1.plot(range(heatmap_len), load_subset, color='black', linewidth=2.5)
        peak_idx = int(np.argmax(load_subset))
        ax1.axvline(peak_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel("Load (kW)")
        ax1.set_title("Load Curve (First 48 Hours)")
        ax1.grid(True, alpha=0.3)
        # Remove spines for ax1
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    else:
        ax1.axis("off")
    
    im = ax2.imshow(heatmap_data, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax2, label='Gate Value (0=Trans, 1=LSTM)', pad=0.02)
    cbar.outline.set_visible(False)
    
    ax2.set_xlabel("Consecutive Forecast Origins")
    ax2.set_ylabel("Feature Dimension (Hidden State)")
    ax2.set_title("Gate Activations Across 48 Consecutive Forecast Origins")
    ax2.grid(False) # Turn off grid for heatmap
    
    # save_figure handles tight_layout
    save_figure(save_dir, "gate_analysis_heatmap.png")
    plt.close()


def plot_forecast_comparison(predictions, X_test, y_test_inv, scaler, input_seq_len, output_seq_len, save_dir, prediction_std=None):
    """
    Plot specific forecast examples.
    """
    print("Plotting Forecast Comparison...")
    model_priority = [m for m in ["PGF-Net", "DLinear", "S-Naive"] if m in predictions]
    if not model_priority:
        model_priority = list(predictions.keys())[:3]

    y_true_all = np.squeeze(y_test_inv)
    if y_true_all.ndim == 1:
        y_true_all = y_true_all.reshape(1, -1)

    def _representative_case_indices(y_true_2d, desired_cases=4):
        if len(y_true_2d) == 0:
            return []
        if len(y_true_2d) <= desired_cases:
            return list(range(len(y_true_2d)))
        peaks = np.max(y_true_2d, axis=1)
        ramps = np.max(np.abs(np.diff(y_true_2d, axis=1)), axis=1)
        med_peak_rank = np.argsort(np.abs(peaks - np.median(peaks)))[0]
        candidates = [
            int(np.argmax(peaks)),
            int(np.argmax(ramps)),
            int(med_peak_rank),
            int(len(y_true_2d) - 1),
        ]
        selected = []
        for idx in candidates:
            if idx not in selected:
                selected.append(idx)
        if len(selected) < desired_cases:
            for idx in _pick_case_indices(len(y_true_2d), desired_cases=desired_cases):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= desired_cases:
                    break
        return selected[:desired_cases]

    indices = _representative_case_indices(y_true_all, desired_cases=4)
    if not indices:
        return

    titles = [
        "Representative High-Peak Case",
        "Representative Rapid-Ramp Case",
        "Representative Median Case",
        "Representative Late-Test Case",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for panel_idx, ax in enumerate(axes):
        if panel_idx >= len(indices):
            ax.axis("off")
            continue
        idx = indices[panel_idx]
        x_hist = X_test[idx]
        x_hist_inv = scaler.inverse_transform(x_hist.reshape(-1, 1)).flatten()
        y_true = y_true_all[idx]
        t_hist = np.arange(0, input_seq_len)
        t_future = np.arange(input_seq_len, input_seq_len + output_seq_len)

        ax.plot(t_hist, x_hist_inv, label="History", color=COLORS["History"], alpha=0.7, linestyle="-")
        ax.plot(t_future, y_true, label="Ground Truth", color=COLORS["Ground Truth"], linewidth=2.8, zorder=10)

        for name in model_priority:
            y_pred = np.squeeze(predictions[name])[idx]
            color = COLORS.get(name, None)
            ax.plot(t_future, y_pred, label=name, color=color, linestyle="--", linewidth=2.2, alpha=0.95)
            if prediction_std is not None and name in prediction_std:
                std_seq = np.squeeze(prediction_std[name])[idx]
                ax.fill_between(t_future, y_pred - std_seq, y_pred + std_seq, color=color, alpha=0.10)

        ax.set_title(titles[panel_idx] if panel_idx < len(titles) else f"Case {panel_idx + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Power Consumption")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(save_dir, "forecast_comparison_grid.png")
    plt.close()

def plot_data_decomposition(df_resampled, save_dir):
    """
    Plot STL decomposition of the data.
    """
    print("Plotting Data Decomposition...")
    # Take a subset to make it visible, e.g., last 1000 hours
    subset = df_resampled['Global_active_power'].iloc[-1000:]
    
    # Decompose
    result = seasonal_decompose(subset, model='additive', period=24)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    result.observed.plot(ax=ax1, color='black', linewidth=1.5)
    ax1.set_ylabel('Observed')
    ax1.set_title('STL Decomposition of Electricity Load (Last 1000 Hours)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    result.trend.plot(ax=ax2, color='#1f77b4', linewidth=2)
    ax2.set_ylabel('Trend')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    result.seasonal.plot(ax=ax3, color='#2ca02c', linewidth=1.5)
    ax3.set_ylabel('Seasonality')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    result.resid.plot(ax=ax4, color='#d62728', marker='o', linestyle='None', markersize=3, alpha=0.6)
    ax4.set_ylabel('Residuals')
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.xlabel("Date")
    save_figure(save_dir, "data_decomposition.png")
    plt.close()

def plot_robustness_boxplot(df_results, save_dir):
    """
    Plot boxplot of MSE across seeds for each model.
    """
    print("Plotting Robustness Boxplot...")
    plt.figure(figsize=(10, 6))
    counts = df_results.groupby("Model").size()
    use_violin = (not counts.empty) and int(counts.min()) >= 5

    if use_violin:
        sns.violinplot(x='Model', y='MSE', data=df_results, palette="Set3", inner=None, cut=0, linewidth=1.5)
    sns.boxplot(x='Model', y='MSE', data=df_results, width=0.18, color="white", fliersize=0, linewidth=1.5)
    sns.stripplot(x='Model', y='MSE', data=df_results, color=".2", size=5, jitter=True, alpha=0.6)

    if use_violin:
        plt.title("Per-seed MSE Distribution Across Models")
    else:
        plt.title("Per-seed MSE Across Models")
    plt.ylabel("MSE")
    plt.xlabel("Model")
    plt.xticks(rotation=25, ha="right")
    
    sns.despine()
    save_figure(save_dir, "robustness_boxplot.png")
    plt.close()

def plot_training_curves(df_curves, save_dir):
    if df_curves.empty:
        return
    y_min = min(df_curves["TrainLoss"].min(), df_curves["ValLoss"].min())
    y_max = max(df_curves["TrainLoss"].max(), df_curves["ValLoss"].max())
    models = df_curves["Model"].unique()
    for model_name in models:
        df_m = df_curves[df_curves["Model"] == model_name].copy()
        if df_m.empty:
            continue
        agg = df_m.groupby("Epoch").agg(
            Train_mean=("TrainLoss", "mean"),
            Train_std=("TrainLoss", "std"),
            Val_mean=("ValLoss", "mean"),
            Val_std=("ValLoss", "std")
        ).reset_index()
        
        plt.figure(figsize=(10, 6))
        
        # Plot Train
        plt.plot(agg["Epoch"], agg["Train_mean"], label="Train", color=COLORS['Train'], linewidth=2)
        plt.fill_between(agg["Epoch"], agg["Train_mean"] - agg["Train_std"], agg["Train_mean"] + agg["Train_std"], 
                         alpha=0.2, color=COLORS['Train'])
        
        # Plot Val
        plt.plot(agg["Epoch"], agg["Val_mean"], label="Val", color=COLORS['Val'], linewidth=2)
        plt.fill_between(agg["Epoch"], agg["Val_mean"] - agg["Val_std"], agg["Val_mean"] + agg["Val_std"], 
                         alpha=0.2, color=COLORS['Val'])
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(y_min, y_max)
        plt.title(f"Training Curves ({model_name})")
        plt.legend()
        
        sns.despine()
        save_figure(save_dir, f"training_curve_{model_name.lower().replace(' ', '_')}.png")
        plt.close()
