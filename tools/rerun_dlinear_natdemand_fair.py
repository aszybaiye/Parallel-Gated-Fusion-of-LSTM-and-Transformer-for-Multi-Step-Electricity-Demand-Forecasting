import os
import sys

import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.revision_p0_experiments import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEEDS,
    ensure_dir,
    prepare_holdout_data,
    run_single_holdout_model,
    summarize_with_ci,
    compute_pairwise_effects,
)


def main():
    device = torch.device("cpu")
    output_root = ensure_dir(os.path.join(PROJECT_ROOT, "Output", "revision_p0"))
    stats_dir = ensure_dir(os.path.join(output_root, "core_holdout_stats"))
    data_bundle = prepare_holdout_data("kaggle")

    rerun_rows = []
    for seed in DEFAULT_SEEDS:
        metrics = run_single_holdout_model(
            model_name="DLinear",
            seed=seed,
            data_bundle=data_bundle,
            output_dir=stats_dir,
            device=device,
            epochs=30,
            batch_size=DEFAULT_BATCH_SIZE,
            patience=8,
            dataset_name="kaggle",
        )
        metrics["Dataset"] = "Kaggle nat_demand"
        rerun_rows.append(metrics)

    rerun_df = pd.DataFrame(rerun_rows)
    rerun_df.to_csv(os.path.join(stats_dir, "kaggle_dlinear_fair_rerun.csv"), index=False)

    canonical_per_seed = os.path.join(stats_dir, "core_models_per_seed_kaggle_newproto.csv")
    if os.path.exists(canonical_per_seed):
        base_df = pd.read_csv(canonical_per_seed)
        base_df = base_df[base_df["Model"] != "DLinear"].copy()
        merged_df = pd.concat([base_df, rerun_df], ignore_index=True)
    else:
        merged_df = rerun_df.copy()
    merged_df = merged_df.sort_values(["Dataset", "Model", "Seed"]).reset_index(drop=True)
    merged_df.to_csv(canonical_per_seed, index=False)

    summary_df = summarize_with_ci(
        merged_df[merged_df["Dataset"] == "Kaggle nat_demand"].copy(),
        "Model",
        ["MSE", "MAE", "PLE", "Ramp_MAE", "TrainingTime_s", "Inference_ms_per_batch"],
    )
    summary_df.insert(0, "Dataset", "Kaggle nat_demand")
    summary_path = os.path.join(stats_dir, "core_models_summary_kaggle_newproto.csv")
    summary_df.to_csv(summary_path, index=False)

    effect_df = compute_pairwise_effects(
        merged_df[merged_df["Dataset"] == "Kaggle nat_demand"].copy(),
        "PGF-Net",
        ["MSE", "MAE", "PLE", "Ramp_MAE"],
    )
    effect_df.insert(0, "Dataset", "Kaggle nat_demand")
    effect_path = os.path.join(stats_dir, "core_models_effect_sizes_kaggle_newproto.csv")
    effect_df.to_csv(effect_path, index=False)

    report_path = os.path.join(stats_dir, "kaggle_dlinear_fair_rerun_notes.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# nat_demand DLinear Fair Rerun\n\n")
        f.write("- Training device: CPU\n")
        f.write("- Optimizer: AdamW with model-specific configuration from `resolve_training_config()`\n")
        f.write("- DLinear nat_demand overrides: lr=3e-4, min_epochs=15, patience=12, ReduceLROnPlateau enabled\n")
        f.write("- Per-epoch training curves: `Output/revision_p0/core_holdout_stats/training_curves/kaggle_dlinear_seed_*.csv`\n")
        f.write("- Canonical per-seed table updated: `core_models_per_seed_kaggle_newproto.csv`\n")
        f.write("- Summary and effect-size tables updated accordingly.\n")


if __name__ == "__main__":
    main()
