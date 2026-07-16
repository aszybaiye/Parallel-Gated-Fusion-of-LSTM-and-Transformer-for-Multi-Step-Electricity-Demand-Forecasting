# Data Availability

## Overview

This project currently uses three local dataset files during revision experiments:

- `data/household_power_consumption.txt`
- `data/weekly pre-dispatch forecast.csv`
- `data/continuous dataset.csv`

## Dataset Mapping

- `UCI Household`
  - File: `data/household_power_consumption.txt`
  - Target column: `Global_active_power`
  - Public source: <https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption>
  - Usage in paper: Table 2, Table 5, Figures 2-6

- `Weekly Pre-dispatch`
  - File: `data/weekly pre-dispatch forecast.csv`
  - Target column: `load_forecast`
  - Interpretation: auxiliary forecast-trajectory task rather than metered realized demand
  - Usage in paper: Table 3, Figure 7

- `nat_demand`
  - File: `data/continuous dataset.csv`
  - Target column: `nat_demand`
  - Usage in paper: Table 4, Figure 7, Figure 8
  - Current scope: same-protocol five-model, five-seed results under the unified `newproto` protocol

## Revision Notes

- All datasets use chronological splitting and training-only normalization.
- The current manuscript uses the `nat_demand` dataset as a completed same-protocol benchmark rather than a baselines-only placeholder.
- The current local package now contains 30-origin UCI rolling-origin raw files, gate-analysis summaries, Figure 6 bin metadata, and DLinear audit tables.
- Before submission, the exact packaged data files or download instructions should still be checked against the final Code Availability wording and the exported package manifest.
