# nat_demand DLinear Fair Rerun

- Training device: CPU
- Optimizer: AdamW with model-specific configuration from `resolve_training_config()`
- DLinear nat_demand overrides: lr=3e-4, min_epochs=15, patience=12, ReduceLROnPlateau enabled
- Per-epoch training curves: `Output/revision_p0/core_holdout_stats/training_curves/kaggle_dlinear_seed_*.csv`
- Canonical per-seed table updated: `core_models_per_seed_kaggle_newproto.csv`
- Summary and effect-size tables updated accordingly.
