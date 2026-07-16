# UCI Rolling-Origin Evidence

- Protocol: 30 weekly-spaced forecast origins under an expanding-window design.
- Retraining: each origin reruns model training on the origin-specific training split before evaluation.
- Horizon metrics: `walkforward_horizon_metrics_uci.csv` stores per-origin, per-horizon MSE/MAE/sMAPE.
- Raw predictions: `walkforward_predictions_uci.csv` stores `y_true` and `y_pred` for every origin and horizon.
- DM test: the released detailed file uses squared-error loss differentials, sample size 30, HAC lag 0, and Benjamini-Hochberg correction within each baseline family.
- Generated figures: `walkforward_origin_distributions_uci.png` and `walkforward_horizon_profiles_uci.png`.
