# Supplementary Materials

## Manuscript Information

**Journal:** Energy, Sustainability and Society  
**Manuscript title:** Gated Fusion of LSTM and Transformer for Robust Multi-Step Electricity Demand Forecasting  
**Article type:** Research Article

## Authors

Zhaoyi Sun, Ping Liu

## Supplementary Content Overview

This supplementary package provides additional materials that support the revised manuscript:

1. Additional benchmark visualizations for the third public dataset
2. Cross-dataset statistical visualization
3. Dataset split statistics visualization
4. UCI rolling-origin diagnostic figures and raw-result summaries
5. Gate quantitative analysis and DLinear audit tables
6. Reproducibility scripts for figure regeneration and similarity self-check
7. Response-to-reviewers document

## Supplementary Figures

### Figure S1. Dataset split statistics across three benchmarks

File: `figures/dataset_statistics_three_datasets.png`  
Description: Stacked split statistics (train/validation/test) for UCI Household, Weekly Pre-dispatch, and Kaggle Nat Demand datasets.

### Figure S2. Cross-dataset metric comparison

File: `figures/overall_comparison_three_datasets.png`  
Description: Comparative visualization of MSE, MAE, and PLE across three datasets and selected baseline models.

### Figure S3. Third-dataset trajectory comparison

File: `figures/forecast_comparison_grid_kaggle.png`  
Description: Multi-step trajectory comparison between PGF-Net and baselines on the third public dataset.

### Figure S4. Third-dataset peak-load scatter

File: `figures/peak_scatter_kaggle.png`  
Description: Predicted versus observed peak load comparison on the third public dataset.

### Figure S5. Third-dataset horizon-wise error

File: `figures/horizon_mse_comparison_kaggle.png`  
Description: Horizon-wise MSE profiles for the compared models.

### Figure S6. Third-dataset robustness distribution

File: `figures/robustness_boxplot_kaggle.png`  
Description: Distributional robustness comparison (MAE) across models.

### Figure S7. UCI rolling-origin origin-level distributions

File: `figures/walkforward_origin_distributions_uci.png`  
Description: Boxplot summary of MSE, MAE, and PLE across the released 30 UCI forecast origins.

### Figure S8. UCI rolling-origin horizon-wise profiles

File: `figures/walkforward_horizon_profiles_uci.png`  
Description: Mean and interquartile-band profiles of MAE and MSE across the 24 forecast horizons over all released origins.

## Supplementary Reproducibility Files

### Script S1. Figure regeneration script

File: `generate_revision_figures.py`  
Purpose: Regenerates the supplementary statistical and cross-dataset visualizations from processed experiment outputs.

### Script S2. Similarity self-check script

File: `generate_similarity_report.py`  
Purpose: Produces a reproducible local similarity self-check report for manuscript integrity screening.

### Report S1. Similarity self-check report

File: `similarity_selfcheck_report.md`  
Purpose: Documents local text-overlap self-check outputs and manuscript hash for traceability.

### Report S2. UCI rolling-origin evidence note

File: `../project_name/docs/walkforward_evidence.md`  
Purpose: Documents retraining-by-origin, released raw files, and DM-test metadata.

### Report S3. DLinear audit summary

File: `../project_name/Output/revision_p0/dlinear_audit/dlinear_audit_summary.csv`  
Purpose: Summarizes cross-dataset DLinear variance, training-time patterns, and flagged seeds.

### Report S4. Gate quantitative summary

File: `../project_name/Output/revision_p0/gate_analysis/uci/all_seeds/gate_statistics_summary.csv`  
Purpose: Summarizes mean, spread, quantiles, entropy, and activation-bin proportions of the UCI gate outputs.

## Supplementary Review Material

### Document R1. Point-by-point response to reviewers

File: `Response_Letter_Revision.md`  
Purpose: Detailed responses to reviewer and editor comments with revision locations.

## Data and Code Availability (Supplementary Confirmation)

Project homepage:  
https://github.com/aszybaiye/Gated-Fusion-of-LSTM-and-Transformer-for-Robust-Multi-Step-Electricity-Demand-Forecasting  

For this manuscript revision, the editorially reviewed code, plotting scripts, and result files are taken from the current local package rather than from a fixed public commit. The manuscript therefore does not cite a public revision hash and instead relies on the packaged local materials and manifest.

The supplementary files listed above are prepared to support transparency, reproducibility, and editorial assessment.
