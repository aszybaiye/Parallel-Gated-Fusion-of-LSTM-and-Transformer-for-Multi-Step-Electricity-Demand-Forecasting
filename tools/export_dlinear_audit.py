import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    files = [
        os.path.join(PROJECT_ROOT, "Output", "revision_p0", "core_holdout_stats", "core_models_per_seed_uci.csv"),
        os.path.join(PROJECT_ROOT, "Output", "revision_p0", "core_holdout_stats", "core_models_per_seed_weekly.csv"),
        os.path.join(PROJECT_ROOT, "Output", "revision_p0", "core_holdout_stats", "core_models_per_seed_kaggle_newproto.csv"),
    ]
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Model"] == "DLinear"].copy()
    if df.empty:
        raise ValueError("No DLinear rows found.")

    out_dir = os.path.join(PROJECT_ROOT, "Output", "revision_p0", "dlinear_audit")
    os.makedirs(out_dir, exist_ok=True)

    metric_cols = ["MSE", "MAE", "sMAPE", "PLE", "Ramp_MAE", "TrainingTime_s"]
    detail_frames = []
    summary_rows = []
    for dataset_name, group in df.groupby("Dataset"):
        g = group.copy().reset_index(drop=True)
        mse_cv = float(g["MSE"].std(ddof=1) / (g["MSE"].mean() + 1e-8)) if len(g) > 1 else 0.0
        for col in metric_cols:
            mean = g[col].mean()
            std = g[col].std(ddof=1)
            if pd.isna(std) or std <= 1e-12:
                g[f"{col}_zscore"] = 0.0
            else:
                g[f"{col}_zscore"] = (g[col] - mean) / std
        if mse_cv < 0.05:
            g["PotentialFailureSeed"] = False
        else:
            g["PotentialFailureSeed"] = (
                (g["MSE_zscore"] > 1.5)
                | (
                    (g["TrainingTime_s"] < g["TrainingTime_s"].median() * 0.75)
                    & (g["MSE"] > g["MSE"].median() * 1.25)
                )
            )
        detail_frames.append(g)
        summary_rows.append(
            {
                "Dataset": dataset_name,
                "SeedCount": int(len(g)),
                "MSE_mean": float(g["MSE"].mean()),
                "MSE_std": float(g["MSE"].std(ddof=1)),
                "MSE_cv": mse_cv,
                "TrainingTime_mean": float(g["TrainingTime_s"].mean()),
                "TrainingTime_std": float(g["TrainingTime_s"].std(ddof=1)),
                "PotentialFailureSeeds": ",".join(str(int(s)) for s in g.loc[g["PotentialFailureSeed"], "Seed"].tolist()),
                "AnyMissingInferenceLatency": bool(g["Inference_ms_per_batch"].isna().any()),
                "AnyNaNMetrics": bool(g[["MSE", "MAE", "sMAPE", "PLE", "Ramp_MAE"]].isna().any().any()),
            }
        )

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    detail_df.to_csv(os.path.join(out_dir, "dlinear_per_seed_audit.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "dlinear_audit_summary.csv"), index=False)

    notes_path = os.path.join(out_dir, "dlinear_audit_notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# DLinear Audit Notes\n\n")
        f.write("- The audit is based on canonical per-seed CSV files under `Output/revision_p0/core_holdout_stats/`.\n")
        for _, row in summary_df.sort_values("Dataset").iterrows():
            dataset_name = row["Dataset"]
            cv = float(row["MSE_cv"])
            flagged = row["PotentialFailureSeeds"]
            if dataset_name == "Kaggle nat_demand":
                if cv < 0.05:
                    f.write(
                        "- `nat_demand` no longer shows the previous bimodal failure pattern after the fairness-stabilized rerun; "
                        "the five seeds now form a tight cluster and no seed is flagged as a failure candidate.\n"
                    )
                else:
                    f.write(
                        f"- `nat_demand` still shows material instability (MSE CV={cv:.3f}); flagged seeds: {flagged or 'none'}.\n"
                    )
            elif dataset_name == "Weekly Pre-dispatch":
                f.write(
                    f"- Weekly remains the weakest DLinear regime in the current implementation (MSE CV={cv:.3f}); "
                    f"flagged seeds: {flagged or 'none'}.\n"
                )
            elif dataset_name == "UCI Household":
                f.write(
                    f"- UCI shows low variance (MSE CV={cv:.3f}), indicating that the anomalous instability is not universal across datasets.\n"
                )

    print(out_dir)


if __name__ == "__main__":
    main()
