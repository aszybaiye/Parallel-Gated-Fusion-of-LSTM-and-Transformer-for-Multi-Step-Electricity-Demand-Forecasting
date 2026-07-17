[README.md](https://github.com/user-attachments/files/30108727/README.md)
# Electricity Demand Forecasting Project

This repository contains the local experimental implementation of PGF-Net and its main comparison baselines for multi-step electricity demand forecasting. The file structure, experiment entry points, and result directories described below reflect the content that currently exists in the release package.

## Current Directory Structure

```text
pgfnet_revision_release/
├── docs/                         # Reproducibility and package documentation
├── figures/                      # Manuscript figures
├── paper/                        # LaTeX source and compiled manuscript
├── results/
│   └── revision_p0/              # Canonical revision outputs
├── src/
│   ├── simulate_experiment.py    # Main experiment entry: train, evaluate, test, plot
│   ├── pgfnet_model.py           # PGF-Net
│   ├── lstm_model.py             # LSTM baseline
│   ├── dlinear_model.py          # DLinear baseline
│   ├── patchtst_model.py         # PatchTST baseline
│   ├── plotting.py               # Figure generation utilities
│   ├── paths.py                  # Output path management
│   ├── utils.py                  # Helper functions and plotting utilities
│   ├── train.py                  # Additional training script
│   └── evaluate.py               # Additional evaluation script
├── tools/                        # Revision utilities and export scripts
├── PAPER_EN.md                   # Current English manuscript source of truth
├── README.md
├── requirements.txt
├── environment.yml
├── experiment_manifest.csv
└── MANIFEST.csv
```

## Environment Setup

Python 3.9 or above is recommended.

```bash
pip install -r requirements.txt
```

Alternatively, use Conda:

```bash
conda env create -f environment.yml
conda activate pgfnet-revision
```

## Data Description

The main data files used in the workspace are:

- `data/household_power_consumption.txt`
- `data/weekly pre-dispatch forecast.csv`
- `data/continuous dataset.csv`

In the current manuscript setup:

- The target column for the UCI dataset is `Global_active_power`.
- The Weekly dataset contains only `load_forecast`, so it is treated as a pre-dispatch forecast trajectory rather than realized metered demand.
- The target column in `continuous dataset.csv` is `nat_demand`.

Preprocessing logic is implemented in `src/data_preprocessing.py`. The current release package does not include a standalone universal dataset download script; raw data should be obtained from the documented sources and placed in the expected `data/` location before running the experiments.

## Main Experiment Entry

The primary experiment script is `src/simulate_experiment.py`. It:

- uses chronological `70%/15%/15%` train/validation/test splitting,
- fits `MinMaxScaler` on the training split only,
- runs `PGF-Net`, `DLinear`, `LSTM`, `PatchTST`, and `S-Naive`,
- exports metric CSV files, per-seed outputs, statistical tests, and manuscript figures,
- resets the unified output directory at the beginning of a main run and overwrites old results.

### Run the UCI Benchmark

```bash
python src/simulate_experiment.py --dataset uci --device cpu --seeds 0,42,123,456,789 --models PGFNet,DLinear,LSTM,PatchTST,SeasonalNaive
```

### Run the Weekly Pre-dispatch Benchmark

```bash
python src/simulate_experiment.py --dataset weekly --device cpu --seeds 0,42,123,456,789 --models PGFNet,DLinear,LSTM,PatchTST,SeasonalNaive
```

## Notes on the Third `nat_demand` Benchmark

For the third dataset used in the manuscript, the canonical results are the `newproto` files under `results/revision_p0/core_holdout_stats/`:

- `core_models_per_seed_kaggle_newproto.csv`
- `core_models_summary_kaggle_newproto.csv`

These files contain the completed five-model results for:

- `PGF-Net`
- `DLinear`
- `LSTM`
- `PatchTST`
- `S-Naive`

Older revision snapshot directories are retained only as historical artifacts and should not be treated as the canonical source for the current manuscript.

## Current Revision Result Entry Points

The manuscript revision primarily relies on the following directories and files:

- `results/revision_p0/core_holdout_stats/`
- `results/revision_p0/ablation_uci/`
- `results/revision_p0/walkforward/`
- `experiment_manifest.csv`
- `docs/results_index.md`
- `docs/data_availability.md`
- `docs/revision_package_contents.md`

Key files include:

- `core_models_per_seed_uci.csv` / `core_models_summary_uci.csv`: main UCI results
- `core_models_per_seed_weekly.csv` / `core_models_summary_weekly.csv`: main Weekly results
- `core_models_per_seed_kaggle_newproto.csv` / `core_models_summary_kaggle_newproto.csv`: five-model `nat_demand` results under the current common protocol
- `ablation_uci_per_seed.csv` / `ablation_uci_summary.csv`: UCI ablation results
- `walkforward_protocol_uci.csv` and related files: current 30-origin UCI rolling-origin outputs
- `gate_analysis/uci/all_seeds/`: gate quantitative summaries and correlation exports
- `figure6_load_bins/`: bin thresholds, sample counts, and metric CSV files for Figure 6
- `dlinear_audit/`: per-seed DLinear audit results

## Testing

```bash
python -m unittest discover tests
```

If the `tests/` directory is not included in the release package, this command is only applicable to the full development workspace.

## Notes

- `PAPER_EN.md` is the current English manuscript source of truth and should remain synchronized with the LaTeX version.
- `experiment_manifest.csv` tracks the provenance of current tables and result files.
- `MANIFEST.csv` records the released package contents and file metadata.
- For public release or submission, consistency across the README, code, environment files, data description, manifest files, and result CSV files should be checked first.
