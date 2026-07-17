import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.plotting import plot_forecast_comparison  # noqa: E402
from src.revision_p0_experiments import build_model, make_loaders, prepare_holdout_data, predict_model  # noqa: E402


OUTPUT_SEQ_LEN = 24


def _first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in candidates: {paths}")


def _load_predictions(model_name, checkpoint_path, test_loader, scaler, device):
    model = build_model(model_name, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    pred_scaled = predict_model(model, test_loader, device, model_name)
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)


def main():
    device = torch.device("cpu")
    data_bundle = prepare_holdout_data("uci")
    _, _, test_loader = make_loaders(
        data_bundle["X_train"],
        data_bundle["y_train"],
        data_bundle["X_val"],
        data_bundle["y_val"],
        data_bundle["X_test"],
        data_bundle["y_test"],
        batch_size=64,
    )
    scaler = data_bundle["scaler"]
    y_true_inv = scaler.inverse_transform(data_bundle["y_test"].reshape(-1, 1)).reshape(data_bundle["y_test"].shape)

    checkpoint_root = os.path.join(PROJECT_ROOT, "Output_uci_rev", "model_output")
    predictions = {
        "PGF-Net": np.squeeze(
            _load_predictions(
                "PGF-Net",
                _first_existing_path([os.path.join(checkpoint_root, "best_pgf_net_seed_0.pt")]),
                test_loader,
                scaler,
                device,
            )
        ),
        "DLinear": np.squeeze(
            _load_predictions(
                "DLinear",
                _first_existing_path([os.path.join(checkpoint_root, "best_dlinear_seed_0.pt")]),
                test_loader,
                scaler,
                device,
            )
        ),
        "S-Naive": np.squeeze(
            scaler.inverse_transform(data_bundle["X_test"][:, -OUTPUT_SEQ_LEN:, 0:1].reshape(-1, 1)).reshape(
                data_bundle["X_test"].shape[0], OUTPUT_SEQ_LEN, 1
            )
        ),
    }

    figure_dir = os.path.join(PROJECT_ROOT, "..", "paper-template", "figures")
    os.makedirs(figure_dir, exist_ok=True)
    plot_forecast_comparison(
        predictions=predictions,
        X_test=data_bundle["X_test"][:, :, 0:1],
        y_test_inv=y_true_inv,
        scaler=scaler,
        input_seq_len=96,
        output_seq_len=OUTPUT_SEQ_LEN,
        save_dir=figure_dir,
        prediction_std=None,
    )
    print(os.path.join(figure_dir, "forecast_comparison_grid.png"))


if __name__ == "__main__":
    main()
