import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import apply_publication_style


def main():
    apply_publication_style()

    csv_path = os.path.join(
        PROJECT_ROOT,
        "Output",
        "revision_p0",
        "core_holdout_stats",
        "core_models_per_seed_kaggle_newproto.csv",
    )
    df = pd.read_csv(csv_path)
    order = [m for m in ["PGF-Net", "DLinear", "LSTM", "PatchTST", "S-Naive"] if m in set(df["Model"])]
    df = df[df["Model"].isin(order)].copy()
    if df.empty:
        raise ValueError("No same-protocol kaggle rows found for Figure 11 redraw.")

    palette = {
        "PGF-Net": "#e15759",
        "DLinear": "#4e79a7",
        "LSTM": "#f28e2b",
        "PatchTST": "#b07aa1",
        "S-Naive": "#59a14f",
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.stripplot(
        data=df,
        x="Model",
        y="MSE",
        hue="Model",
        order=order,
        palette=palette,
        jitter=0.12,
        size=8,
        alpha=0.85,
        legend=False,
        ax=ax,
    )

    grouped = df.groupby("Model")["MSE"].agg(["mean", "std"]).reindex(order)
    for idx, model_name in enumerate(order):
        mean = grouped.loc[model_name, "mean"]
        std = grouped.loc[model_name, "std"]
        ax.hlines(mean, idx - 0.22, idx + 0.22, colors="black", linewidth=2.0)
        if not np.isnan(std):
            ax.vlines(idx, mean - std, mean + std, colors="black", linewidth=1.5)
            ax.hlines([mean - std, mean + std], idx - 0.08, idx + 0.08, colors="black", linewidth=1.5)

    ax.set_xlabel("")
    ax.set_ylabel("Per-seed MSE")
    ax.set_title("nat_demand Same-protocol Seed Robustness")
    ax.grid(axis="y", alpha=0.35)
    sns.despine()

    figure_dir = os.path.join(PROJECT_ROOT, "..", "paper-template", "figures")
    os.makedirs(figure_dir, exist_ok=True)
    save_path = os.path.join(figure_dir, "robustness_boxplot_kaggle.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(save_path)


if __name__ == "__main__":
    main()
