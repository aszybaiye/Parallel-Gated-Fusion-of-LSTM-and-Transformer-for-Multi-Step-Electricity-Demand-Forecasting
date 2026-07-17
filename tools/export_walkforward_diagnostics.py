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


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def main():
    wf_dir = os.path.join(PROJECT_ROOT, "Output", "revision_p0", "walkforward")
    fig_dir = os.path.join(wf_dir, "figures")
    paper_fig_dir = os.path.join(PROJECT_ROOT, "..", "paper-template", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)

    origin_df = pd.read_csv(os.path.join(wf_dir, "walkforward_origin_summary_uci.csv"))
    horizon_df = pd.read_csv(os.path.join(wf_dir, "walkforward_horizon_metrics_uci.csv"))
    pred_df = pd.read_csv(os.path.join(wf_dir, "walkforward_predictions_uci.csv"))
    dm_df = pd.read_csv(os.path.join(wf_dir, "walkforward_dm_tests_uci.csv"))

    origin_df = origin_df.sort_values(["Model", "OriginIndex"]).reset_index(drop=True)
    horizon_df = horizon_df.sort_values(["Model", "OriginIndex", "Horizon"]).reset_index(drop=True)
    pred_df = pred_df.sort_values(["Model", "OriginIndex", "Horizon"]).reset_index(drop=True)

    origin_df.to_csv(os.path.join(wf_dir, "walkforward_origin_distribution_uci.csv"), index=False)

    horizon_summary = (
        horizon_df.groupby(["Model", "Horizon"])
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MSE_q25=("MSE", lambda s: float(np.quantile(s, 0.25))),
            MSE_median=("MSE", "median"),
            MSE_q75=("MSE", lambda s: float(np.quantile(s, 0.75))),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAE_q25=("MAE", lambda s: float(np.quantile(s, 0.25))),
            MAE_median=("MAE", "median"),
            MAE_q75=("MAE", lambda s: float(np.quantile(s, 0.75))),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_std=("sMAPE", "std"),
            SampleCount=("MAE", "size"),
        )
        .reset_index()
    )
    horizon_summary.to_csv(os.path.join(wf_dir, "walkforward_horizon_summary_uci.csv"), index=False)

    pivot = horizon_df.pivot_table(index=["OriginIndex", "Horizon"], columns="Model", values="MSE")
    loss_rows = []
    for baseline in [m for m in pivot.columns if m != "PGF-Net"]:
        tmp = pivot[["PGF-Net", baseline]].dropna().reset_index()
        tmp["Reference"] = "PGF-Net"
        tmp["Baseline"] = baseline
        tmp["LossDifferential"] = tmp["PGF-Net"] - tmp[baseline]
        loss_rows.append(tmp[["OriginIndex", "Horizon", "Reference", "Baseline", "LossDifferential"]])
    loss_df = pd.concat(loss_rows, ignore_index=True)
    loss_df.to_csv(os.path.join(wf_dir, "walkforward_loss_differentials_uci.csv"), index=False)

    detailed_dm_rows = []
    for baseline, group in dm_df.groupby("Baseline"):
        q_vals = _bh_adjust(group["DM_p_value_two_sided"].values)
        for (_, row), q_val in zip(group.iterrows(), q_vals):
            detailed_dm_rows.append(
                {
                    **row.to_dict(),
                    "LossDifferentialDefinition": "(y_hat_ref - y)^2 - (y_hat_baseline - y)^2",
                    "SampleSizeOrigins": 30,
                    "HACLag": 0,
                    "VarianceEstimator": "sample_variance",
                    "MultipleComparisonCorrection": "Benjamini-Hochberg within baseline",
                    "BH_q_value": float(q_val),
                }
            )
    detailed_dm_df = pd.DataFrame(detailed_dm_rows)
    detailed_dm_df.to_csv(os.path.join(wf_dir, "walkforward_dm_tests_detailed_uci.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    for ax, metric in zip(axes, ["MSE", "MAE", "PLE"]):
        sns.boxplot(data=origin_df, x="Model", y=metric, ax=ax, showfliers=True)
        ax.set_title(f"{metric} Across 30 Origins")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        sns.despine(ax=ax)
    plt.tight_layout()
    origin_fig = "walkforward_origin_distributions_uci.png"
    fig.savefig(os.path.join(fig_dir, origin_fig), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(paper_fig_dir, origin_fig), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for ax, metric in zip(axes, ["MAE", "MSE"]):
        for model_name, group in horizon_summary.groupby("Model"):
            ax.plot(group["Horizon"], group[f"{metric}_mean"], label=model_name)
            lower = group[f"{metric}_q25"] if f"{metric}_q25" in group else group[f"{metric}_mean"] - group[f"{metric}_std"].fillna(0)
            upper = group[f"{metric}_q75"] if f"{metric}_q75" in group else group[f"{metric}_mean"] + group[f"{metric}_std"].fillna(0)
            ax.fill_between(group["Horizon"], lower, upper, alpha=0.12)
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Horizon Across Origins")
        ax.grid(axis="y", alpha=0.2)
        sns.despine(ax=ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    horizon_fig = "walkforward_horizon_profiles_uci.png"
    fig.savefig(os.path.join(fig_dir, horizon_fig), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(paper_fig_dir, horizon_fig), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata_path = os.path.join(PROJECT_ROOT, "docs", "walkforward_evidence.md")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("# UCI Rolling-Origin Evidence\n\n")
        f.write("- Protocol: 30 weekly-spaced forecast origins under an expanding-window design.\n")
        f.write("- Retraining: each origin reruns model training on the origin-specific training split before evaluation.\n")
        f.write("- Horizon metrics: `walkforward_horizon_metrics_uci.csv` stores per-origin, per-horizon MSE/MAE/sMAPE.\n")
        f.write("- Raw predictions: `walkforward_predictions_uci.csv` stores `y_true` and `y_pred` for every origin and horizon.\n")
        f.write("- DM test: the released detailed file uses squared-error loss differentials, sample size 30, HAC lag 0, and Benjamini-Hochberg correction within each baseline family.\n")
        f.write("- Supplementary figures: `walkforward_origin_distributions_uci.png` and `walkforward_horizon_profiles_uci.png`.\n")

    print(wf_dir)


if __name__ == "__main__":
    main()
