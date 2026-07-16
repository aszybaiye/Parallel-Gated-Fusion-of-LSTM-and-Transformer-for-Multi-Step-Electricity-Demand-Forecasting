# DLinear Audit Notes

- The audit is based on canonical per-seed CSV files under `Output/revision_p0/core_holdout_stats/`.
- `nat_demand` no longer shows the previous bimodal failure pattern after the fairness-stabilized rerun; the five seeds now form a tight cluster and no seed is flagged as a failure candidate.
- UCI shows low variance (MSE CV=0.009), indicating that the anomalous instability is not universal across datasets.
- Weekly remains the weakest DLinear regime in the current implementation (MSE CV=0.146); flagged seeds: none.
