# Results Index

## Canonical Tables

- `Table 2`
  - Source: `Output/revision_p0/core_holdout_stats/core_models_summary_uci.csv`
  - Per-seed source: `Output/revision_p0/core_holdout_stats/core_models_per_seed_uci.csv`
  - Note: this is the canonical full-model UCI source used by both the main results table and the full-model row in Table 5

- `Table 3`
  - Source: `Output/revision_p0/core_holdout_stats/core_models_summary_weekly.csv`
  - Per-seed source: `Output/revision_p0/core_holdout_stats/core_models_per_seed_weekly.csv`

- `Table 4`
  - Source: `Output/revision_p0/core_holdout_stats/core_models_summary_kaggle_newproto.csv`
  - Per-seed source: `Output/revision_p0/core_holdout_stats/core_models_per_seed_kaggle_newproto.csv`
  - Scope: same-protocol five-model comparison (`PGF-Net`, `DLinear`, `LSTM`, `PatchTST`, `S-Naive`)

- `Table 5`
  - Source: `Output/revision_p0/ablation_uci/ablation_uci_summary.csv`
  - Per-seed source: `Output/revision_p0/ablation_uci/ablation_uci_per_seed.csv`
  - Note: the full-model `PGF-Net` row is synchronized to the canonical Table 2 UCI result

## Canonical Figures

- `Figure 7`
  - Output: `paper-template/figures/overall_comparison_three_datasets.png`
  - Script: `paper-template/generate_revision_figures.py`
  - Sources:
    - `core_models_summary_uci.csv`
    - `core_models_summary_weekly.csv`
    - `core_models_summary_kaggle_newproto.csv`
  - Note: regenerated as dataset-specific subplots (not a shared linear axis)

- `Figure 8`
  - Output: `paper-template/figures/robustness_boxplot_kaggle.png`
  - Script: `tools/redraw_figure11_kaggle_newproto.py`
  - Source: `Output/revision_p0/core_holdout_stats/core_models_per_seed_kaggle_newproto.csv`

- `Figure 2`
  - Output: `paper-template/figures/forecast_comparison_grid.png`
  - Script: `tools/redraw_figure2_uci_forecast_grid.py`
  - Sources:
    - `Output_uci_rev/model_output/best_pgf_net_seed_0.pt`
    - `Output_uci_rev/model_output/best_dlinear_seed_0.pt`
    - `data/household_power_consumption.txt`

- `Figure 6`
  - Output: `paper-template/figures/error_by_load_condition.png`
  - Script: `tools/redraw_figure6_uci_load_bins.py`
  - Sources:
    - `Output_uci_rev/model_output/best_pgf_net_seed_0.pt`
    - `Output_uci_rev/model_output/best_dlinear_seed_0.pt`
    - `Output_uci_rev/model_output/best_lstm_seed_0.pt`
    - `Output_uci_rev/model_output/best_patchtst_seed_0.pt`
    - `Output/revision_p0/figure6_load_bins/error_by_load_condition.csv`
    - `Output/revision_p0/figure6_load_bins/error_by_load_condition_bin_metadata.csv`

## Rolling-Origin Outputs

- `UCI rolling-origin (current 30 origins)`
  - Directory: `Output/revision_p0/walkforward/`
  - Protocol file: `walkforward_protocol_uci.csv`
  - Origin summary: `walkforward_origin_summary_uci.csv`
  - Horizon summary: `walkforward_horizon_summary_uci.csv`
  - Raw predictions: `walkforward_predictions_uci.csv`
  - DM tests: `walkforward_dm_tests_uci.csv`, `walkforward_dm_tests_detailed_uci.csv`
  - Loss differentials: `walkforward_loss_differentials_uci.csv`
  - Diagnostics note: `docs/walkforward_evidence.md`

## Additional Audits

- `Gate quantitative analysis (UCI, five seeds)`
  - Directory: `Output/revision_p0/gate_analysis/uci/all_seeds/`
  - Key files:
    - `gate_statistics_summary.csv`
    - `gate_bin_proportions.csv`
    - `gate_signal_correlations.csv`
    - `gate_extended_signal_correlations_all_seeds.csv`

- `DLinear audit`
  - Directory: `Output/revision_p0/dlinear_audit/`
  - Key files:
    - `dlinear_audit_summary.csv`
    - `dlinear_per_seed_audit.csv`
    - `dlinear_audit_notes.md`

## Manifest

- Unified experiment mapping: `experiment_manifest.csv`
