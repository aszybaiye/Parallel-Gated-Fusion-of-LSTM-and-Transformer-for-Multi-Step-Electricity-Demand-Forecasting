# Revision Package Contents

## Included in Workspace

- `src/`
  - model implementations
  - training and evaluation code
  - revision scripts

- `tools/`
  - chunk runners
  - merge/finalize scripts
  - redraw scripts
  - recovery utilities

- `Output/revision_p0/`
  - current revision result tables
  - per-seed CSV files
  - UCI ablation CSV files
  - 30-origin UCI walk-forward outputs
  - gate-analysis summaries
  - load-bin diagnostics
  - DLinear audit tables

- `paper-template/`
  - LaTeX manuscript
  - figure captions
  - regenerated figure scripts

- `PAPER_EN.md`
  - synchronized English manuscript source

- `requirements.txt`
- `environment.yml`
- `experiment_manifest.csv`
- `docs/data_availability.md`
- `docs/results_index.md`

## Important Scope Notes

- `nat_demand` now provides a same-protocol five-model, five-seed comparison.
- The current local UCI rolling-origin outputs contain 30 origins with raw predictions, loss differentials, and detailed DM metadata.
- The current local package also includes explicit gate-analysis CSV summaries, quartile-based Figure 6 bin metadata, and a DLinear audit.
- Any final submission package should still be checked against the latest manuscript wording before being sent to editors or reviewers.
