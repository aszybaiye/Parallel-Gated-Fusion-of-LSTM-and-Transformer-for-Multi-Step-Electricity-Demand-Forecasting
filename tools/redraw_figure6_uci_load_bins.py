import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.revision_p0_experiments import build_model, make_loaders, prepare_holdout_data, predict_model  # noqa: E402


OUTPUT_SEQ_LEN = 24
MODEL_ORDER = ["PGF-Net", "DLinear", "LSTM", "PatchTST", "S-Naive"]
COLORS = {
    "PGF-Net": "#d62728",
    "DLinear": "#2ca02c",
    "LSTM": "#9467bd",
    "PatchTST": "#ff7f0e",
    "S-Naive": "#7f7f7f",
}


def _first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in candidates: {paths}")


def _load_predictions(model_name, checkpoint_path, test_loader, scaler, device):
    model = build_model(model_name, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    pred_scaled = predict_model(model, test_loader, device, model_name)
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)


def main():
    device = torch.device("cpu")
    data_bundle = prepare_holdout_data("uci")
    _, _, test_loader = make_loaders(
        data_bundle["X_train"],
        data_bundle["y_train"],
        data_bundle["X_val"],
        data_bundle["y_val"],
        data_bundle["X_test"],
        data_bundle["y_test"],
        batch_size=64,
    )
    scaler = data_bundle["scaler"]
    y_true = scaler.inverse_transform(data_bundle["y_test"].reshape(-1, 1)).reshape(data_bundle["y_test"].shape)
    y_true = np.squeeze(y_true)

    checkpoint_root = os.path.join(PROJECT_ROOT, "Output_uci_rev", "model_output")
    pred_map = {
        "PGF-Net": np.squeeze(
            _load_predictions(
                "PGF-Net",
                _first_existing_path([os.path.join(checkpoint_root, "best_pgf_net_seed_0.pt")]),
                test_loader,
                scaler,
                device,
            )
        ),
        "DLinear": np.squeeze(
            _load_predictions(
                "DLinear",
                _first_existing_path([os.path.join(checkpoint_root, "best_dlinear_seed_0.pt")]),
                test_loader,
                scaler,
                device,
            )
        ),
        "LSTM": np.squeeze(
            _load_predictions(
                "LSTM",
                _first_existing_path([os.path.join(checkpoint_root, "best_lstm_seed_0.pt")]),
                test_loader,
                scaler,
                device,
            )
        ),
        "PatchTST": np.squeeze(
            _load_predictions(
                "PatchTST",
                _first_existing_path([os.path.join(checkpoint_root, "best_patchtst_seed_0.pt")]),
                test_loader,
                scaler,
                device,
            )
        ),
        "S-Naive": np.squeeze(
            scaler.inverse_transform(data_bundle["X_test"][:, -OUTPUT_SEQ_LEN:, 0:1].reshape(-1, 1)).reshape(
                data_bundle["X_test"].shape[0], OUTPUT_SEQ_LEN, 1
            )
        ),
    }

    peaks_true = np.max(y_true, axis=1)
    q25, q50, q75 = np.quantile(peaks_true, [0.25, 0.5, 0.75])
    labels = ["Low", "Medium", "High", "Peak"]
    edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=float)
    bin_idx = np.digitize(peaks_true, edges[1:-1], right=True)
    eps = 1e-8

    rows = []
    meta_rows = []
    for b, label in enumerate(labels):
        mask = bin_idx == b
        meta_rows.append(
            {
                "Bin": label,
                "LowerBoundInclusive": float(edges[b]) if np.isfinite(edges[b]) else None,
                "UpperBoundExclusive": float(edges[b + 1]) if np.isfinite(edges[b + 1]) else None,
                "SampleCount": int(mask.sum()),
                "Definition": "quartile_of_true_future_peak",
                "PLEDefinition": "samplewise_relative_peak_error_then_average",
            }
        )
        for model_name, y_pred in pred_map.items():
            if mask.sum() == 0:
                continue
            mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)
            pred_peaks = np.max(y_pred, axis=1)
            ple_per_sample = np.abs(peaks_true - pred_peaks) / (peaks_true + eps) * 100
            rows.append(
                {
                    "Model": model_name,
                    "Run": 0,
                    "Bin": label,
                    "MAE": float(mae_per_sample[mask].mean()),
                    "PLE": float(ple_per_sample[mask].mean()),
                }
            )

    df_bins = pd.DataFrame(rows)
    df_meta = pd.DataFrame(meta_rows)
    out_dir = os.path.join(PROJECT_ROOT, "Output", "revision_p0", "figure6_load_bins")
    os.makedirs(out_dir, exist_ok=True)
    df_bins.to_csv(os.path.join(out_dir, "error_by_load_condition.csv"), index=False)
    df_meta.to_csv(os.path.join(out_dir, "error_by_load_condition_bin_metadata.csv"), index=False)

    df_plot = df_bins.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharex=True)
    for metric, ax in [("MAE", axes[0]), ("PLE", axes[1])]:
        pivot = (
            df_plot.pivot(index="Bin", columns="Model", values=metric)
            .reindex(labels)
            .reindex(columns=MODEL_ORDER)
        )
        x = np.arange(len(labels))
        width = 0.15
        for i, model_name in enumerate(MODEL_ORDER):
            ax.bar(
                x + (i - (len(MODEL_ORDER) - 1) / 2) * width,
                pivot[model_name].values,
                width=width,
                color=COLORS[model_name],
                label=model_name,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Quartile-Based Load Bin")
        ax.grid(axis="y", alpha=0.2)
        sns.despine(ax=ax)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    paper_fig_dir = os.path.join(PROJECT_ROOT, "..", "paper-template", "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    fig.savefig(os.path.join(paper_fig_dir, "error_by_load_condition.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "error_by_load_condition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_dir)


if __name__ == "__main__":
    main()
