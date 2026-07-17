import os
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.plotting import plot_gate_analysis  # noqa: E402


def _extended_correlations(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = [
        "load_level",
        "abs_ramp",
        "future_peak",
        "future_range",
        "history_volatility",
        "future_volatility",
    ]
    gate_signal = sample_df["gate_mean"].astype(float).values
    for col in targets:
        vals = sample_df[col].astype(float).values
        if len(vals) > 1 and np.std(vals) > 0 and np.std(gate_signal) > 0:
            corr = float(np.corrcoef(gate_signal, vals)[0, 1])
        else:
            corr = float("nan")
        rows.append({"signal": col, "pearson_r": corr})
    return pd.DataFrame(rows)


def main():
    output_root = os.path.join(PROJECT_ROOT, "Output", "revision_p0", "gate_analysis", "uci")
    seed_dirs = sorted(
        [
            os.path.join(output_root, name)
            for name in os.listdir(output_root)
            if name.startswith("seed_") and os.path.isdir(os.path.join(output_root, name))
        ]
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {output_root}")

    sample_frames = []
    gate_arrays = []
    for seed_dir in seed_dirs:
        sample_frames.append(pd.read_csv(os.path.join(seed_dir, "gate_sample_summary.csv")))
        gate_arrays.append(pd.read_csv(os.path.join(seed_dir, "gate_raw_matrix.csv")).drop(columns=["seed", "sample_index"]).values)

    combined_dir = os.path.join(output_root, "all_seeds")
    os.makedirs(combined_dir, exist_ok=True)
    combined_sample_df = pd.concat(sample_frames, ignore_index=True)
    combined_gates = np.concatenate(gate_arrays, axis=0)

    combined_sample_df.to_csv(os.path.join(combined_dir, "gate_sample_summary_all_seeds.csv"), index=False)
    pd.DataFrame(combined_gates).to_csv(os.path.join(combined_dir, "gate_raw_matrix_all_seeds.csv"), index=False)
    _extended_correlations(combined_sample_df).to_csv(
        os.path.join(combined_dir, "gate_extended_signal_correlations_all_seeds.csv"),
        index=False,
    )
    (
        pd.DataFrame(combined_gates)
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "dimension"})
    ).to_csv(os.path.join(combined_dir, "gate_dimension_summary_all_seeds.csv"), index=False)

    plot_gate_analysis(
        [combined_gates],
        combined_dir,
        load_series=combined_sample_df["load_level"].tolist(),
        hour_series=combined_sample_df["hour"].tolist(),
    )
    print(combined_dir)


if __name__ == "__main__":
    main()
