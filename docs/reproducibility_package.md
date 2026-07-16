# Reproducibility Package

## Project Access

- Project homepage: `https://github.com/aszybaiye/Gated-Fusion-of-LSTM-and-Transformer-for-Robust-Multi-Step-Electricity-Demand-Forecasting`
- Current manuscript basis: local workspace package and generated outputs in this repository snapshot
- Commit policy for this revision: do not cite a fixed public commit unless the synchronized archival upload is actually accessible

## Local Bundle Contents

- Bundle archive: `Output/revision_p0/revision_package/pgfnet_revision_bundle.zip`
- Bundle manifest: `Output/revision_p0/revision_package/revision_package_manifest.csv`

- `src/`
- `tools/`
- `requirements.txt`
- `environment.yml`
- `PAPER_EN.md`
- `docs/results_index.md`
- `docs/data_availability.md`
- `docs/revision_package_contents.md`
- `docs/walkforward_evidence.md`
- `Output/revision_p0/core_holdout_stats/`
- `Output/revision_p0/ablation_uci/`
- `Output/revision_p0/walkforward/`
- `Output/revision_p0/gate_analysis/uci/all_seeds/`
- `Output/revision_p0/figure6_load_bins/`
- `Output/revision_p0/dlinear_audit/`

## Key Released Evidence

- Per-seed CSV tables for all three datasets
- Raw UCI rolling-origin predictions and loss differentials
- Detailed DM metadata with Benjamini-Hochberg correction
- Gate mean/std/quantile summaries and activation-bin proportions
- Quartile-based Figure 6 bin thresholds and sample counts
- DLinear per-seed audit tables for UCI, weekly, and `nat_demand`
- Fairness-stabilized `nat_demand` DLinear rerun tables and per-epoch training curves
