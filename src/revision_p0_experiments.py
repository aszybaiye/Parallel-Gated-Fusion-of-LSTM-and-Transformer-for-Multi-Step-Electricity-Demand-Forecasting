import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.dlinear_model import DLinearModel
from src.lstm_model import LSTMModel
from src.patchtst_model import PatchTSTModel
from src.pgfnet_model import PGFNet
from src.plotting import COLORS
from src.utils import apply_publication_style


INPUT_SEQ_LEN = 96
OUTPUT_SEQ_LEN = 24
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 8
DEFAULT_LR = 0.001
DEFAULT_SEEDS = [0, 42, 123, 456, 789]


def _slugify(text: str) -> str:
    return "".join([c.lower() if c.isalnum() else "_" for c in str(text)]).strip("_")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def append_row_csv(file_path: str, row: Dict) -> None:
    df = pd.DataFrame([row])
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode="a", header=header, index=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_training_config(model_name: str, dataset_name: str, epochs: int, patience: int) -> Dict[str, float]:
    dataset_key = _slugify(dataset_name)
    config = {
        "epochs": int(epochs),
        "patience": int(patience),
        "min_epochs": 0,
        "lr": float(DEFAULT_LR),
        "weight_decay": 0.0,
        "scheduler_patience": 0,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-5,
    }
    if model_name == "DLinear":
        config.update({
            "lr": 5e-4,
            "patience": max(int(patience), 10),
            "min_epochs": 10,
            "weight_decay": 1e-5,
            "scheduler_patience": 3,
        })
        if dataset_key in ["kaggle", "nat_demand", "kaggle_nat_demand", "continuous"]:
            config.update({
                "lr": 3e-4,
                "patience": max(int(patience), 12),
                "min_epochs": 15,
                "weight_decay": 1e-4,
                "scheduler_patience": 4,
            })
    config["min_epochs"] = min(int(config["min_epochs"]), int(config["epochs"]))
    return config


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    epsilon = 1e-8
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100)
    true_peaks = np.max(y_true, axis=1)
    pred_peaks = np.max(y_pred, axis=1)
    ple = float(np.mean(np.abs(true_peaks - pred_peaks) / (true_peaks + epsilon)) * 100)
    true_ramp = np.diff(y_true, axis=1)
    pred_ramp = np.diff(y_pred, axis=1)
    ramp_mae = float(np.mean(np.abs(true_ramp - pred_ramp)))
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "sMAPE": smape,
        "PLE": ple,
        "Ramp_MAE": ramp_mae,
    }


def create_sequences(data: np.ndarray, input_len: int, output_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len, 0])
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, output_len, 1)
    return X, y


def load_resampled_dataframe(dataset: str, data_path: str = "") -> Tuple[pd.DataFrame, str]:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    dataset_key = _slugify(dataset)
    if dataset_key in ["uci", "household", "household_power_consumption"]:
        txt_path = data_path if data_path else os.path.join(data_dir, "household_power_consumption.txt")
        df = pd.read_csv(txt_path, sep=";", low_memory=False, na_values=["nan", "?"])
        df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
        df.set_index("dt", inplace=True)
        df.drop(["Date", "Time"], axis=1, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df_resampled = df.resample("h").mean().ffill()
        return df_resampled, "Global_active_power"
    if dataset_key in ["weekly_predispatch", "weekly_pre_dispatch", "predispatch", "weekly"]:
        csv_path = data_path if data_path else os.path.join(data_dir, "weekly pre-dispatch forecast.csv")
        df = pd.read_csv(csv_path)
        df["dt"] = pd.to_datetime(df["datetime"])
        df.set_index("dt", inplace=True)
        if "datetime" in df.columns:
            df = df.drop(columns=["datetime"])
        df["load_forecast"] = pd.to_numeric(df["load_forecast"], errors="coerce")
        df_resampled = df[["load_forecast"]].resample("h").mean().ffill()
        return df_resampled, "load_forecast"
    if dataset_key in ["kaggle", "nat_demand", "continuous"]:
        csv_path = data_path if data_path else os.path.join(data_dir, "continuous dataset.csv")
        df = pd.read_csv(csv_path)
        df["dt"] = pd.to_datetime(df["datetime"])
        df.set_index("dt", inplace=True)
        if "datetime" in df.columns:
            df = df.drop(columns=["datetime"])
        df["nat_demand"] = pd.to_numeric(df["nat_demand"], errors="coerce")
        df_resampled = df[["nat_demand"]].resample("h").mean().ffill()
        return df_resampled, "nat_demand"
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_feature_matrix(df_resampled: pd.DataFrame, target_col: str, scaler=None, fit_scaler: bool = False) -> Tuple[np.ndarray, MinMaxScaler]:
    raw_values = df_resampled[target_col].values.reshape(-1, 1)
    hours = df_resampled.index.hour.values.reshape(-1, 1)
    days = df_resampled.index.dayofweek.values.reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    if fit_scaler:
        scaler.fit(raw_values)
    scaled = scaler.transform(raw_values)
    data = np.concatenate([scaled, hours, days], axis=1)
    return data, scaler


def prepare_holdout_data(dataset: str, data_path: str = ""):
    df_resampled, target_col = load_resampled_dataframe(dataset, data_path)
    raw_values = df_resampled[target_col].values.reshape(-1, 1)
    total_len = len(raw_values)
    train_size = int(total_len * 0.7)
    val_size = int(total_len * 0.15)
    train_df = df_resampled.iloc[:train_size].copy()
    val_df = df_resampled.iloc[train_size : train_size + val_size].copy()
    test_df = df_resampled.iloc[train_size + val_size :].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[[target_col]].values)
    train_data, _ = build_feature_matrix(train_df, target_col, scaler=scaler, fit_scaler=False)
    val_data, _ = build_feature_matrix(val_df, target_col, scaler=scaler, fit_scaler=False)
    test_data, _ = build_feature_matrix(test_df, target_col, scaler=scaler, fit_scaler=False)
    X_train, y_train = create_sequences(train_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    X_val, y_val = create_sequences(val_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    X_test, y_test = create_sequences(test_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    return {
        "df_resampled": df_resampled,
        "target_col": target_col,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, dataloader, optimizer, criterion, device, model_name, progress_prefix: str = "", progress_every: int = 0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        model_input = X_batch if model_name.startswith("PGF-Net") else X_batch[:, :, 0:1]
        optimizer.zero_grad()
        output = model(model_input)
        if output.shape != y_batch.shape:
            output = output.view(y_batch.shape)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if progress_every > 0 and batch_idx % progress_every == 0:
            label = progress_prefix if progress_prefix else f"[Train] {model_name}"
            print(f"{label} batch={batch_idx} loss={loss.item():.6f}", flush=True)
    return total_loss / max(n_batches, 1)


def evaluate_model(model, dataloader, criterion, device, model_name):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            model_input = X_batch if model_name.startswith("PGF-Net") else X_batch[:, :, 0:1]
            output = model(model_input)
            if output.shape != y_batch.shape:
                output = output.view(y_batch.shape)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def predict_model(model, dataloader, device, model_name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            model_input = X_batch if model_name.startswith("PGF-Net") else X_batch[:, :, 0:1]
            output = model(model_input)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def measure_inference_ms_per_batch(model, dataloader, device, model_name, warmup_batches=2, measure_batches=5):
    model.eval()
    times = []
    with torch.no_grad():
        for batch_idx, (X_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            model_input = X_batch if model_name.startswith("PGF-Net") else X_batch[:, :, 0:1]
            if batch_idx < warmup_batches:
                _ = model(model_input)
                continue
            start = time.perf_counter()
            _ = model(model_input)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
            if len(times) >= measure_batches:
                break
    return float(np.mean(times)) if times else float("nan")


def build_model(model_name: str, device: torch.device):
    pgf_kwargs = {
        "input_dim": 3,
        "output_dim": OUTPUT_SEQ_LEN,
        "d_model": 32,
        "nhead": 2,
        "num_layers": 1,
        "lstm_hidden": 32,
        "dropout": 0.0,
    }
    if model_name == "PGF-Net":
        model = PGFNet(**pgf_kwargs, fusion_mode="gated").to(device)
    elif model_name == "PGF-Net Fixed 0.5":
        model = PGFNet(**pgf_kwargs, fusion_mode="fixed_average", fixed_alpha=0.5).to(device)
    elif model_name == "PGF-Net w/o Gating":
        model = PGFNet(**pgf_kwargs, fusion_mode="fixed_average", fixed_alpha=0.5).to(device)
    elif model_name == "PGF-Net Scalar Fusion":
        model = PGFNet(**pgf_kwargs, fusion_mode="scalar").to(device)
    elif model_name == "PGF-Net Concatenation":
        model = PGFNet(**pgf_kwargs, fusion_mode="concat").to(device)
    elif model_name == "PGF-Net w/o Transformer":
        model = PGFNet(**pgf_kwargs, use_transformer=False, fusion_mode="fixed_average").to(device)
    elif model_name == "PGF-Net w/o LSTM":
        model = PGFNet(**pgf_kwargs, use_lstm=False, fusion_mode="fixed_average").to(device)
    elif model_name == "PGF-Net w/o Time Embedding":
        model = PGFNet(**pgf_kwargs, use_time_embeddings=False, fusion_mode="gated").to(device)
    elif model_name == "PGF-Net w/o PosEnc":
        model = PGFNet(**pgf_kwargs, use_positional_encoding=False, fusion_mode="gated").to(device)
    elif model_name == "DLinear":
        model = DLinearModel(input_size=1, output_size=OUTPUT_SEQ_LEN, seq_len=INPUT_SEQ_LEN).to(device)
    elif model_name == "LSTM":
        model = LSTMModel(input_size=1, hidden_size=32, output_size=OUTPUT_SEQ_LEN, num_layers=2, dropout=0.1).to(device)
    elif model_name == "PatchTST":
        model = PatchTSTModel(input_size=1, output_size=OUTPUT_SEQ_LEN, seq_len=INPUT_SEQ_LEN).to(device)
    elif model_name == "S-Naive":
        return None
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def run_single_holdout_model(
    model_name: str,
    seed: int,
    data_bundle: Dict,
    output_dir: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    dataset_name: str = "",
):
    set_seed(seed)
    X_train = data_bundle["X_train"]
    y_train = data_bundle["y_train"]
    X_val = data_bundle["X_val"]
    y_val = data_bundle["y_val"]
    X_test = data_bundle["X_test"]
    y_test = data_bundle["y_test"]
    scaler = data_bundle["scaler"]
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)

    if model_name == "S-Naive":
        y_pred_scaled = X_test[:, -OUTPUT_SEQ_LEN:, 0:1].copy()
        training_time_s = 0.0
        inference_ms = float("nan")
        num_params = 0
    else:
        model = build_model(model_name, device)
        num_params = count_parameters(model)
        train_config = resolve_training_config(model_name, dataset_name, epochs, patience)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config["weight_decay"],
        )
        scheduler = None
        if train_config["scheduler_patience"] > 0:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=train_config["scheduler_factor"],
                patience=int(train_config["scheduler_patience"]),
                min_lr=train_config["scheduler_min_lr"],
            )
        criterion = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        model_path = os.path.join(output_dir, "models")
        ensure_dir(model_path)
        checkpoint_path = os.path.join(model_path, f"{_slugify(model_name)}_seed_{seed}.pt")
        patience_counter = 0
        best_epoch = 0
        history_rows = []
        history_dir = ensure_dir(os.path.join(output_dir, "training_curves"))
        history_path = os.path.join(
            history_dir,
            f"{_slugify(dataset_name or 'unknown_dataset')}_{_slugify(model_name)}_seed_{seed}.csv",
        )
        t0 = time.perf_counter()
        for epoch_idx in range(epochs):
            train_loss = train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                model_name,
                progress_prefix=f"[Train] {model_name} seed={seed} epoch={epoch_idx + 1}/{epochs}",
                progress_every=100,
            )
            val_loss = evaluate_model(model, val_loader, criterion, device, model_name)
            print(
                f"[Val] {model_name} seed={seed} epoch={epoch_idx + 1}/{epochs} val_loss={val_loss:.6f}",
                flush=True,
            )
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, checkpoint_path)
                patience_counter = 0
                best_epoch = epoch_idx + 1
            else:
                patience_counter += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            history_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Seed": seed,
                "Epoch": epoch_idx + 1,
                "TrainLoss": float(train_loss),
                "ValLoss": float(val_loss),
                "BestValLossSoFar": float(best_val),
                "PatienceCounter": int(patience_counter),
                "LearningRate": current_lr,
                "MinEpochs": int(train_config["min_epochs"]),
                "ConfiguredPatience": int(train_config["patience"]),
            })
            if epoch_idx + 1 >= train_config["min_epochs"] and patience_counter >= train_config["patience"]:
                print(
                    (
                        f"[EarlyStop] {model_name} seed={seed} epoch={epoch_idx + 1} "
                        f"best_epoch={best_epoch} min_epochs={train_config['min_epochs']} "
                        f"patience={train_config['patience']}"
                    ),
                    flush=True,
                )
                break
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        training_time_s = time.perf_counter() - t0
        if best_state is not None:
            model.load_state_dict(best_state)
        y_pred_scaled = predict_model(model, test_loader, device, model_name)
        inference_ms = measure_inference_ms_per_batch(model, test_loader, device, model_name)

    y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    metrics = calculate_metrics(y_true_inv, y_pred_inv)
    metrics.update({
        "Model": model_name,
        "Seed": seed,
        "Params": num_params,
        "TrainingTime_s": float(training_time_s),
        "Inference_ms_per_batch": float(inference_ms),
        "TrainLR": float(train_config["lr"]) if model_name != "S-Naive" else float("nan"),
        "TrainPatience": int(train_config["patience"]) if model_name != "S-Naive" else 0,
        "MinEpochs": int(train_config["min_epochs"]) if model_name != "S-Naive" else 0,
        "BestEpoch": int(best_epoch) if model_name != "S-Naive" else 0,
    })
    return metrics


def summarize_with_ci(df: pd.DataFrame, group_col: str, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for group_name, group_df in df.groupby(group_col):
        row = {group_col: group_name, "N": int(len(group_df))}
        for metric in metric_cols:
            vals = group_df[metric].astype(float).values
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_effects(df: pd.DataFrame, reference_model: str, metric_cols: List[str]) -> pd.DataFrame:
    ref_df = df[df["Model"] == reference_model].copy()
    rows = []
    for model_name in sorted(df["Model"].unique()):
        if model_name == reference_model:
            continue
        cmp_df = df[df["Model"] == model_name].copy()
        merged = pd.merge(ref_df[["Seed"] + metric_cols], cmp_df[["Seed"] + metric_cols], on="Seed", suffixes=("_ref", "_cmp"))
        if merged.empty:
            continue
        for metric in metric_cols:
            diff = merged[f"{metric}_cmp"] - merged[f"{metric}_ref"]
            denom = diff.std(ddof=1) if len(diff) > 1 else float("nan")
            effect_size = float(diff.mean() / (denom + 1e-12)) if denom == denom else float("nan")
            if len(diff) >= 3:
                w_stat, w_p = stats.wilcoxon(merged[f"{metric}_ref"], merged[f"{metric}_cmp"], alternative="two-sided")
            else:
                w_p = float("nan")
            rows.append({
                "Reference": reference_model,
                "Comparison": model_name,
                "Metric": metric,
                "N": int(len(diff)),
                "MeanDiff_cmp_minus_ref": float(diff.mean()),
                "EffectSize_d": effect_size,
                "Wilcoxon_p_two_sided": float(w_p),
            })
    return pd.DataFrame(rows)


def plot_ablation_summary(summary_df: pd.DataFrame, save_path: str, metric: str = "MSE"):
    plt.figure(figsize=(11, 5))
    order = summary_df.sort_values(f"{metric}_mean")[["Model"]]
    plot_df = summary_df.set_index("Model").loc[order["Model"]].reset_index()
    ax = sns.barplot(data=plot_df, x="Model", y=f"{metric}_mean", palette="Set2")
    ax.errorbar(
        x=np.arange(len(plot_df)),
        y=plot_df[f"{metric}_mean"].values,
        yerr=plot_df[f"{metric}_ci95"].values,
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
    )
    ax.set_title(f"UCI Ablation Summary ({metric})")
    ax.set_ylabel(f"{metric} mean")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=28)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_uci_ablation(output_root: str, device: torch.device, epochs: int, batch_size: int, patience: int):
    ablation_dir = ensure_dir(os.path.join(output_root, "ablation_uci"))
    data_bundle = prepare_holdout_data("uci")
    models = [
        "PGF-Net",
        "PGF-Net Fixed 0.5",
        "PGF-Net Scalar Fusion",
        "PGF-Net Concatenation",
        "PGF-Net w/o Transformer",
        "PGF-Net w/o LSTM",
        "PGF-Net w/o Time Embedding",
        "PGF-Net w/o PosEnc",
    ]
    rows = []
    for model_name in models:
        for seed in DEFAULT_SEEDS:
            print(f"[Ablation] {model_name} | seed={seed}", flush=True)
            rows.append(
                run_single_holdout_model(
                    model_name=model_name,
                    seed=seed,
                    data_bundle=data_bundle,
                    output_dir=ablation_dir,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ablation_dir, "ablation_uci_per_seed.csv"), index=False)
    metric_cols = ["MSE", "MAE", "PLE", "Ramp_MAE", "Params", "TrainingTime_s", "Inference_ms_per_batch"]
    summary_df = summarize_with_ci(df, "Model", metric_cols)
    summary_df.to_csv(os.path.join(ablation_dir, "ablation_uci_summary.csv"), index=False)
    effect_df = compute_pairwise_effects(df, "PGF-Net", ["MSE", "MAE", "PLE", "Ramp_MAE"])
    effect_df.to_csv(os.path.join(ablation_dir, "ablation_uci_effect_sizes.csv"), index=False)
    plot_ablation_summary(summary_df, os.path.join(ablation_dir, "ablation_uci_mse.png"), metric="MSE")
    return df, summary_df, effect_df


def run_core_holdout_stats(
    output_root: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    datasets: List[str] = None,
    models: List[str] = None,
    seeds: List[int] = None,
):
    stats_dir = ensure_dir(os.path.join(output_root, "core_holdout_stats"))
    dataset_list = [
        ("uci", "UCI Household"),
        ("weekly", "Weekly Pre-dispatch"),
        ("kaggle", "Kaggle nat_demand"),
    ]
    if datasets:
        allow = {_slugify(x) for x in datasets}
        dataset_list = [item for item in dataset_list if _slugify(item[0]) in allow or _slugify(item[1]) in allow]
    if models is None:
        models = ["PGF-Net", "DLinear", "LSTM", "PatchTST", "S-Naive"]
    if seeds is None:
        seeds = DEFAULT_SEEDS
    suffix = "all" if not datasets else "_".join([_slugify(x) for x in datasets])
    per_seed_path = os.path.join(stats_dir, f"core_models_per_seed_{suffix}.csv")
    rows = []
    for dataset_key, dataset_name in dataset_list:
        data_bundle = prepare_holdout_data(dataset_key)
        for model_name in models:
            for seed in seeds:
                print(f"[CoreStats] {dataset_name} | {model_name} | seed={seed}", flush=True)
                metrics = run_single_holdout_model(
                    model_name=model_name,
                    seed=seed,
                    data_bundle=data_bundle,
                    output_dir=stats_dir,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    dataset_name=dataset_key,
                )
                metrics["Dataset"] = dataset_name
                rows.append(metrics)
                append_row_csv(per_seed_path, metrics)
    df = pd.DataFrame(rows)
    if df.empty and os.path.exists(per_seed_path):
        df = pd.read_csv(per_seed_path)

    summary_rows = []
    for dataset_name, group_df in df.groupby("Dataset"):
        summary_df = summarize_with_ci(
            group_df,
            "Model",
            ["MSE", "MAE", "PLE", "Ramp_MAE", "TrainingTime_s", "Inference_ms_per_batch"],
        )
        summary_df.insert(0, "Dataset", dataset_name)
        summary_rows.append(summary_df)
    summary_all = pd.concat(summary_rows, ignore_index=True)
    summary_all.to_csv(os.path.join(stats_dir, f"core_models_summary_{suffix}.csv"), index=False)

    effect_rows = []
    for dataset_name, group_df in df.groupby("Dataset"):
        effect_df = compute_pairwise_effects(group_df, "PGF-Net", ["MSE", "MAE", "PLE", "Ramp_MAE"])
        if not effect_df.empty:
            effect_df.insert(0, "Dataset", dataset_name)
            effect_rows.append(effect_df)
    if effect_rows:
        pd.concat(effect_rows, ignore_index=True).to_csv(
            os.path.join(stats_dir, f"core_models_effect_sizes_{suffix}.csv"), index=False
        )
    return df


def build_origin_list(total_len: int, initial_val_size: int, output_len: int, origin_step: int, max_origins: int) -> List[int]:
    origin_start = int(total_len * 0.85)
    max_origin = total_len - output_len
    origins = list(range(origin_start, max_origin, origin_step))
    if max_origins > 0:
        origins = origins[:max_origins]
    return origins


def forecast_single_origin(
    model_name: str,
    seed: int,
    df_resampled: pd.DataFrame,
    target_col: str,
    origin_idx: int,
    val_window: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    checkpoint_dir: str = "",
):
    train_end = origin_idx - val_window
    train_df = df_resampled.iloc[:train_end].copy()
    val_df = df_resampled.iloc[train_end:origin_idx].copy()
    if len(train_df) < INPUT_SEQ_LEN + OUTPUT_SEQ_LEN or len(val_df) < INPUT_SEQ_LEN + OUTPUT_SEQ_LEN:
        return None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[[target_col]].values)
    train_data, _ = build_feature_matrix(train_df, target_col, scaler=scaler, fit_scaler=False)
    val_data, _ = build_feature_matrix(val_df, target_col, scaler=scaler, fit_scaler=False)
    X_train, y_train = create_sequences(train_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    X_val, y_val = create_sequences(val_data, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)

    hist_df = df_resampled.iloc[origin_idx - INPUT_SEQ_LEN : origin_idx].copy()
    future_df = df_resampled.iloc[origin_idx : origin_idx + OUTPUT_SEQ_LEN].copy()
    X_test_data, _ = build_feature_matrix(hist_df, target_col, scaler=scaler, fit_scaler=False)
    X_test = X_test_data.reshape(1, INPUT_SEQ_LEN, 3)
    y_true = future_df[target_col].values.reshape(1, OUTPUT_SEQ_LEN, 1)

    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_val, y_val, X_test, y_true, batch_size)

    set_seed(seed)
    if model_name == "S-Naive":
        y_pred = hist_df[target_col].values[-OUTPUT_SEQ_LEN:].reshape(1, OUTPUT_SEQ_LEN, 1)
        training_time_s = 0.0
    else:
        model = build_model(model_name, device)
        optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
        criterion = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        checkpoint_path = ""
        if checkpoint_dir:
            ensure_dir(checkpoint_dir)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{_slugify(model_name)}_origin_{origin_idx}_seed_{seed}.pt",
            )
        patience_counter = 0
        t0 = time.perf_counter()
        for epoch_idx in range(epochs):
            train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                model_name,
                progress_prefix=(
                    f"[WF-Train] origin_idx={origin_idx} model={model_name} "
                    f"seed={seed} epoch={epoch_idx + 1}/{epochs}"
                ),
                progress_every=100,
            )
            val_loss = evaluate_model(model, val_loader, criterion, device, model_name)
            print(
                (
                    f"[WF-Val] origin_idx={origin_idx} model={model_name} "
                    f"seed={seed} epoch={epoch_idx + 1}/{epochs} val_loss={val_loss:.6f}"
                ),
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_path:
                    torch.save(best_state, checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        training_time_s = time.perf_counter() - t0
        if best_state is not None:
            model.load_state_dict(best_state)
        y_pred_scaled = predict_model(model, test_loader, device, model_name)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(1, OUTPUT_SEQ_LEN, 1)

    y_true_2d = y_true.reshape(1, OUTPUT_SEQ_LEN)
    y_pred_2d = y_pred.reshape(1, OUTPUT_SEQ_LEN)
    overall_metrics = calculate_metrics(y_true_2d, y_pred_2d)
    horizon_rows = []
    eps = 1e-8
    for h in range(OUTPUT_SEQ_LEN):
        yt = float(y_true_2d[0, h])
        yp = float(y_pred_2d[0, h])
        horizon_rows.append({
            "Horizon": h + 1,
            "MSE": float((yp - yt) ** 2),
            "MAE": float(abs(yp - yt)),
            "sMAPE": float(2 * abs(yp - yt) / (abs(yp) + abs(yt) + eps) * 100),
        })
    return overall_metrics, horizon_rows, y_true_2d.flatten(), y_pred_2d.flatten(), training_time_s


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> float:
    d = np.asarray(loss_a) - np.asarray(loss_b)
    if len(d) < 2:
        return float("nan")
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d <= 1e-12:
        return float("nan")
    dm_stat = mean_d / math.sqrt(var_d / len(d))
    return float(2 * (1 - stats.norm.cdf(abs(dm_stat))))


def plot_walkforward_curves(df_horizon: pd.DataFrame, dataset_name: str, save_dir: str):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in df_horizon.groupby("Model"):
        agg = group.groupby("Horizon").agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")).reset_index()
        color = COLORS.get(model_name, None)
        ax.plot(agg["Horizon"], agg["MAE_mean"], label=model_name, color=color)
        std = agg["MAE_std"].fillna(0.0).values
        ax.fill_between(agg["Horizon"], agg["MAE_mean"] - std, agg["MAE_mean"] + std, alpha=0.18, color=color)
    ax.set_title(f"Walk-forward MAE by Horizon ({dataset_name})")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"walkforward_mae_{_slugify(dataset_name)}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_walkforward(
    output_root: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    origin_step: int,
    max_origins: int,
    datasets: List[str] = None,
    models: List[str] = None,
    origin_offset: int = 0,
):
    wf_dir = ensure_dir(os.path.join(output_root, "walkforward"))
    figures_dir = ensure_dir(os.path.join(wf_dir, "figures"))
    dataset_list = [
        ("uci", "UCI Household"),
        ("weekly", "Weekly Pre-dispatch"),
        ("kaggle", "Kaggle nat_demand"),
    ]
    if datasets:
        allow = {_slugify(x) for x in datasets}
        dataset_list = [item for item in dataset_list if _slugify(item[0]) in allow or _slugify(item[1]) in allow]
    if models is None:
        models = ["PGF-Net", "DLinear", "S-Naive"]
    overall_rows = []
    horizon_rows = []
    pred_rows = []
    dm_rows = []
    protocol_rows = []
    suffix = "all" if not datasets else "_".join([_slugify(x) for x in datasets])
    overall_path = os.path.join(wf_dir, f"walkforward_origin_summary_{suffix}.csv")
    horizon_path = os.path.join(wf_dir, f"walkforward_horizon_metrics_{suffix}.csv")
    pred_path = os.path.join(wf_dir, f"walkforward_predictions_{suffix}.csv")
    model_ckpt_dir = ensure_dir(os.path.join(wf_dir, "models"))

    for dataset_key, dataset_name in dataset_list:
        df_resampled, target_col = load_resampled_dataframe(dataset_key)
        total_len = len(df_resampled)
        initial_val_size = int(total_len * 0.15)
        origins = build_origin_list(total_len, initial_val_size, OUTPUT_SEQ_LEN, origin_step, 0)
        if origin_offset > 0 or max_origins > 0:
            end_idx = origin_offset + max_origins if max_origins > 0 else None
            origins = origins[origin_offset:end_idx]
        protocol_rows.append({
            "Dataset": dataset_name,
            "InitialTrainHours": int(total_len * 0.7),
            "ValidationHours": initial_val_size,
            "OriginStepHours": origin_step,
            "Origins": len(origins),
            "FirstOrigin": str(df_resampled.index[origins[0]]) if origins else "",
            "LastOrigin": str(df_resampled.index[origins[-1]]) if origins else "",
        })
        for origin_no, origin_idx in enumerate(origins):
            global_origin_no = origin_offset + origin_no
            for model_name in models:
                seed = DEFAULT_SEEDS[0] if model_name != "S-Naive" else -1
                print(
                    f"[WalkForward] {dataset_name} | origin={global_origin_no+1}/{origin_offset + len(origins)} | model={model_name}",
                    flush=True,
                )
                out = forecast_single_origin(
                    model_name=model_name,
                    seed=seed if seed >= 0 else 0,
                    df_resampled=df_resampled,
                    target_col=target_col,
                    origin_idx=origin_idx,
                    val_window=initial_val_size,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    checkpoint_dir=os.path.join(model_ckpt_dir, _slugify(dataset_name)),
                )
                if out is None:
                    continue
                overall_metrics, horizon_metric_rows, y_true, y_pred, training_time_s = out
                overall_metrics.update({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "OriginIndex": global_origin_no,
                    "OriginTimestamp": str(df_resampled.index[origin_idx]),
                    "TrainingTime_s": float(training_time_s),
                })
                overall_rows.append(overall_metrics)
                append_row_csv(overall_path, overall_metrics)
                for item in horizon_metric_rows:
                    item.update({
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "OriginIndex": global_origin_no,
                        "OriginTimestamp": str(df_resampled.index[origin_idx]),
                    })
                    horizon_rows.append(item)
                    append_row_csv(horizon_path, item)
                for h in range(OUTPUT_SEQ_LEN):
                    pred_row = {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "OriginIndex": global_origin_no,
                        "OriginTimestamp": str(df_resampled.index[origin_idx]),
                        "Horizon": h + 1,
                        "y_true": float(y_true[h]),
                        "y_pred": float(y_pred[h]),
                    }
                    pred_rows.append(pred_row)
                    append_row_csv(pred_path, pred_row)

        df_h_dataset = pd.DataFrame([r for r in horizon_rows if r["Dataset"] == dataset_name])
        if not df_h_dataset.empty:
            plot_walkforward_curves(df_h_dataset, dataset_name, figures_dir)
            pivot_losses = {}
            for model_name in models:
                group = df_h_dataset[df_h_dataset["Model"] == model_name]
                if group.empty:
                    continue
                pivot_losses[model_name] = group.pivot(index="OriginIndex", columns="Horizon", values="MSE").sort_index()
            if "PGF-Net" in pivot_losses:
                for baseline in [m for m in models if m != "PGF-Net" and m in pivot_losses]:
                    common_cols = sorted(set(pivot_losses["PGF-Net"].columns).intersection(set(pivot_losses[baseline].columns)))
                    for horizon in common_cols:
                        p_val = dm_test(
                            pivot_losses["PGF-Net"][horizon].values,
                            pivot_losses[baseline][horizon].values,
                        )
                        dm_rows.append({
                            "Dataset": dataset_name,
                            "Reference": "PGF-Net",
                            "Baseline": baseline,
                            "Horizon": int(horizon),
                            "DM_p_value_two_sided": p_val,
                        })

    df_overall = pd.DataFrame(overall_rows)
    df_horizon = pd.DataFrame(horizon_rows)
    df_preds = pd.DataFrame(pred_rows)
    df_protocol = pd.DataFrame(protocol_rows)
    df_dm = pd.DataFrame(dm_rows)
    df_protocol.to_csv(os.path.join(wf_dir, f"walkforward_protocol_{suffix}.csv"), index=False)
    if not df_overall.empty:
        df_overall.to_csv(os.path.join(wf_dir, f"walkforward_origin_summary_{suffix}.csv"), index=False)
    if not df_horizon.empty:
        df_horizon.to_csv(os.path.join(wf_dir, f"walkforward_horizon_metrics_{suffix}.csv"), index=False)
    if not df_preds.empty:
        df_preds.to_csv(os.path.join(wf_dir, f"walkforward_predictions_{suffix}.csv"), index=False)
    if not df_dm.empty:
        df_dm.to_csv(os.path.join(wf_dir, f"walkforward_dm_tests_{suffix}.csv"), index=False)
    if not df_overall.empty:
        summary = summarize_with_ci(df_overall, "Model", ["MSE", "MAE", "sMAPE", "PLE", "Ramp_MAE", "TrainingTime_s"])
        summary.to_csv(os.path.join(wf_dir, f"walkforward_overall_summary_{suffix}.csv"), index=False)
    return df_overall, df_horizon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--run_ablation", action="store_true")
    parser.add_argument("--run_core_stats", action="store_true")
    parser.add_argument("--run_walkforward", action="store_true")
    parser.add_argument("--finalize_core_stats", action="store_true")
    parser.add_argument("--origin_step", type=int, default=168)
    parser.add_argument("--max_origins", type=int, default=6)
    parser.add_argument("--origin_offset", type=int, default=0)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    apply_publication_style()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_root = ensure_dir(os.path.join(PROJECT_ROOT, "Output", "revision_p0"))
    stats_dir = ensure_dir(os.path.join(output_root, "core_holdout_stats"))
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    suffix = "all" if not datasets else "_".join([_slugify(x) for x in datasets])
    with open(os.path.join(output_root, "cli_trace.txt"), "a", encoding="utf-8") as f:
        f.write(
            f"run_ablation={args.run_ablation}; run_core_stats={args.run_core_stats}; "
            f"finalize_core_stats={args.finalize_core_stats}; run_walkforward={args.run_walkforward}; "
            f"datasets={datasets}; models={model_names}; epochs={args.epochs}; patience={args.patience}\n"
        )
    if args.run_ablation:
        run_uci_ablation(output_root, device, args.epochs, args.batch_size, args.patience)
    if args.run_core_stats:
        run_core_holdout_stats(
            output_root,
            device,
            args.epochs,
            args.batch_size,
            args.patience,
            datasets=datasets or None,
            models=model_names or None,
            seeds=seed_list or None,
        )
    if args.finalize_core_stats:
        per_seed_path = os.path.join(stats_dir, f"core_models_per_seed_{suffix}.csv")
        if os.path.exists(per_seed_path):
            df = pd.read_csv(per_seed_path)
            df = df.drop_duplicates(subset=["Dataset", "Model", "Seed"], keep="last")
            df.to_csv(per_seed_path, index=False)
            summary_rows = []
            for dataset_name, group_df in df.groupby("Dataset"):
                summary_df = summarize_with_ci(
                    group_df,
                    "Model",
                    ["MSE", "MAE", "PLE", "Ramp_MAE", "TrainingTime_s", "Inference_ms_per_batch"],
                )
                summary_df.insert(0, "Dataset", dataset_name)
                summary_rows.append(summary_df)
            if summary_rows:
                pd.concat(summary_rows, ignore_index=True).to_csv(
                    os.path.join(stats_dir, f"core_models_summary_{suffix}.csv"),
                    index=False,
                )
            effect_rows = []
            for dataset_name, group_df in df.groupby("Dataset"):
                effect_df = compute_pairwise_effects(group_df, "PGF-Net", ["MSE", "MAE", "PLE", "Ramp_MAE"])
                if not effect_df.empty:
                    effect_df.insert(0, "Dataset", dataset_name)
                    effect_rows.append(effect_df)
            if effect_rows:
                pd.concat(effect_rows, ignore_index=True).to_csv(
                    os.path.join(stats_dir, f"core_models_effect_sizes_{suffix}.csv"),
                    index=False,
                )
    if args.run_walkforward:
        run_walkforward(
            output_root,
            device,
            args.epochs,
            args.batch_size,
            args.patience,
            args.origin_step,
            args.max_origins,
            datasets=datasets or None,
            models=model_names or None,
            origin_offset=args.origin_offset,
        )


if __name__ == "__main__":
    main()
