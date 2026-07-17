
print("Script starting...", flush=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import argparse
import urllib.request
import zipfile
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Import models and utils
from src.paths import get_output_root, output_path, ensure_dir, reset_output_root
from src.dlinear_model import DLinearModel
from src.patchtst_model import PatchTSTModel
from src.pgfnet_model import PGFNet
from src.lstm_model import LSTMModel
from src.utils import apply_publication_style

# --- 1. Configuration & Setup ---
# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

base_output_root = get_output_root()
data_dir = os.path.join(project_root, 'data')

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(input_seq_len=96, output_seq_len=24, dataset="uci", data_path=None):
    """
    Loads, preprocesses, and splits the Household Power Consumption dataset.
    Includes Time Features (Hour, Day) for PGFNet.
    """
    dataset_key = _slugify(dataset)
    if dataset_key in ["uci", "household", "household_power_consumption"]:
        zip_path = os.path.join(data_dir, 'household_power_consumption.zip')
        txt_path = data_path if data_path else os.path.join(data_dir, 'household_power_consumption.txt')
        if not os.path.exists(txt_path):
            print("Downloading dataset...", flush=True)
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
            try:
                with urllib.request.urlopen(url) as response:
                    with open(zip_path, 'wb') as f:
                        f.write(response.read())
                print("Unzipping dataset...", flush=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print("Dataset downloaded and unzipped.", flush=True)
            except Exception as e:
                print(f"Failed to download/unzip data: {e}")
        print("Loading data (Full Dataset)...", flush=True)
        df = pd.read_csv(txt_path, sep=';', low_memory=False, na_values=['nan','?'])
        print("Processing datetime...", flush=True)
        df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df.set_index('dt', inplace=True)
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Resampling to hourly...", flush=True)
        df_resampled = df.resample('h').mean()
        df_resampled = df_resampled.ffill()
        target_col = "Global_active_power"
    elif dataset_key in ["weekly_predispatch", "weekly_pre_dispatch", "predispatch", "weekly"]:
        csv_path = data_path if data_path else os.path.join(data_dir, 'weekly pre-dispatch forecast.csv')
        print("Loading data (Weekly Pre-dispatch)...", flush=True)
        df = pd.read_csv(csv_path)
        df['dt'] = pd.to_datetime(df['datetime'])
        df.set_index('dt', inplace=True)
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])
        df = df.rename(columns={"load_forecast": "Global_active_power"})
        df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
        df = df.select_dtypes(include=[np.number])
        df_resampled = df.resample('h').mean()
        df_resampled = df_resampled.ffill()
        target_col = "Global_active_power"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # --- Feature Extraction ---
    # 1. Target: Global_active_power
    # 2. Time Features: Hour (0-23), DayOfWeek (0-6)
    
    raw_values = df_resampled[target_col].values.reshape(-1, 1)
    hours = df_resampled.index.hour.values.reshape(-1, 1)
    days = df_resampled.index.dayofweek.values.reshape(-1, 1)
    
    # --- Train/Val/Test Split ---
    total_len = len(raw_values)
    train_size = int(total_len * 0.7)
    val_size = int(total_len * 0.15)
    test_size = total_len - train_size - val_size
    
    # Split raw target
    train_raw = raw_values[:train_size]
    val_raw = raw_values[train_size:train_size + val_size]
    test_raw = raw_values[train_size + val_size:]
    
    # Split time features
    train_time = np.concatenate([hours[:train_size], days[:train_size]], axis=1)
    val_time = np.concatenate([hours[train_size:train_size + val_size], days[train_size:train_size + val_size]], axis=1)
    test_time = np.concatenate([hours[train_size + val_size:], days[train_size + val_size:]], axis=1)
    
    # --- Scaling ---
    # Only scale the target (Load)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)
    
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)
    
    # Concatenate: [ScaledLoad, Hour, Day] -> (N, 3)
    train_data = np.concatenate([train_scaled, train_time], axis=1)
    val_data = np.concatenate([val_scaled, val_time], axis=1)
    test_data = np.concatenate([test_scaled, test_time], axis=1)
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}", flush=True)

    # --- Create Sequences ---
    def create_sequences(data, input_len, output_len):
        X, y = [], []
        # data is (N, 3)
        # We predict Load (col 0)
        for i in range(len(data) - input_len - output_len + 1):
            X.append(data[i:(i + input_len)]) # (Seq, 3)
            y.append(data[(i + input_len):(i + input_len + output_len), 0]) # (PredLen, 1) - Only Load
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, input_seq_len, output_seq_len)
    X_val, y_val = create_sequences(val_data, input_seq_len, output_seq_len)
    X_test, y_test = create_sequences(test_data, input_seq_len, output_seq_len)
    
    # Reshape targets to (N, PredLen, 1)
    y_train = y_train.reshape(-1, output_seq_len, 1)
    y_val = y_val.reshape(-1, output_seq_len, 1)
    y_test = y_test.reshape(-1, output_seq_len, 1)
    
    print(f"Created sequences: X_train shape={X_train.shape}, y_train shape={y_train.shape}", flush=True)
    
    train_end = train_size - 1
    val_end = train_size + val_size - 1
    split_info = {
        "dataset": dataset,
        "total_len": int(total_len),
        "train_start": str(df_resampled.index[0]),
        "train_end": str(df_resampled.index[train_end]),
        "val_start": str(df_resampled.index[train_end + 1]),
        "val_end": str(df_resampled.index[val_end]),
        "test_start": str(df_resampled.index[val_end + 1]),
        "test_end": str(df_resampled.index[-1])
    }
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df_resampled, split_info

# --- 3. Metrics Calculation ---
def calculate_metrics(y_true, y_pred):
    """
    Calculates regression metrics including Peak Load Error (PLE) and Ramp Rate MAE.
    y_true, y_pred: (N, 24) or (N, 24, 1) unscaled
    """
    y_true = np.squeeze(np.asarray(y_true))
    y_pred = np.squeeze(np.asarray(y_pred))
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1))
    
    # sMAPE
    epsilon = 1e-8
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
    
    # Peak Load Error (PLE)
    # Calculate max load for each 24h horizon
    true_peaks = np.max(y_true, axis=1)
    pred_peaks = np.max(y_pred, axis=1)
    ple = np.mean(np.abs(true_peaks - pred_peaks) / (true_peaks + epsilon)) * 100
    
    # Ramp Rate MAE
    # First difference along the time axis (axis 1)
    true_ramp = np.diff(y_true, axis=1)
    pred_ramp = np.diff(y_pred, axis=1)
    ramp_mae = np.mean(np.abs(true_ramp - pred_ramp))
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "sMAPE": smape, "PLE": ple, "Ramp_MAE": ramp_mae}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 4. Model Training & Evaluation ---

def train_model(model, dataloader, optimizer, criterion, device, model_name, max_batches=None):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Handle inputs: PGFNet gets all 3 features. Others get only Load (idx 0).
        if model_name.startswith('PGF-Net'):
            model_input = X_batch
        else:
            model_input = X_batch[:, :, 0:1] # (Batch, Seq, 1)
        
        optimizer.zero_grad()
        output = model(model_input)
        
        # Ensure shapes match
        if output.shape != y_batch.shape:
            output = output.view(y_batch.shape)
            
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
        
    return total_loss / max(n_batches, 1)

def evaluate_model(model, dataloader, criterion, device, model_name, max_batches=None):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            if model_name.startswith('PGF-Net'):
                model_input = X_batch
            else:
                model_input = X_batch[:, :, 0:1]
                
            output = model(model_input)
            
            if output.shape != y_batch.shape:
                output = output.view(y_batch.shape)
                
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    return total_loss / max(n_batches, 1)

def predict(model, dataloader, device, model_name, max_batches=None, return_gate=False):
    model.eval()
    predictions = []
    gate_values = []
    with torch.no_grad():
        for batch_idx, (X_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            
            if model_name.startswith('PGF-Net'):
                model_input = X_batch
            else:
                model_input = X_batch[:, :, 0:1]
                
            if return_gate and hasattr(model, "forward"):
                output, gates = model(model_input, return_gate=True)
                if gates is not None:
                    gate_values.append(gates.detach().cpu().numpy())
            else:
                output = model(model_input)
            predictions.append(output.cpu().numpy())
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    preds = np.concatenate(predictions, axis=0)
    if return_gate:
        return preds, gate_values
    return preds

def measure_inference_ms_per_batch(model, dataloader, device, model_name, warmup_batches=3, measure_batches=10):
    model.eval()
    times = []
    with torch.no_grad():
        for batch_idx, (X_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            if model_name.startswith('PGF-Net'):
                model_input = X_batch
            else:
                model_input = X_batch[:, :, 0:1]
            if batch_idx < warmup_batches:
                _ = model(model_input)
                continue
            start = time.perf_counter()
            _ = model(model_input)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
            if len(times) >= measure_batches:
                break
    if not times:
        return float("nan")
    return float(np.mean(times))

def _slugify(text: str) -> str:
    return "".join([c.lower() if c.isalnum() else "_" for c in str(text)]).strip("_")

# --- 5. Main Experiment Runner ---
def run_experiment():
    print("========= Starting Full Experiment (Optimized) =========")
    
    # --- Reset Output ---
    reset_output_root()
    results_dir = ensure_dir(base_output_root)
    figures_dir = ensure_dir(output_path("figures"))
    model_output_dir = ensure_dir(output_path("model_output"))
    apply_publication_style()
    
    # --- Hyperparameters ---
    INPUT_SEQ_LEN = 96
    OUTPUT_SEQ_LEN = 24
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 8
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="uci")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--max_test_batches", type=int, default=0)
    parser.add_argument("--viz_samples", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="0,42,123,456,789")
    parser.add_argument("--models", type=str, default="PGFNet,DLinear,LSTM,PatchTST,SeasonalNaive")
    parser.add_argument("--run_ablation", action="store_true")
    args, _ = parser.parse_known_args()

    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
    max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None
    max_test_batches = args.max_test_batches if args.max_test_batches > 0 else None
    max_test_batches_for_timing = max_test_batches if max_test_batches else 10

    device = torch.device("cpu")
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- Load Data ---
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df_resampled, split_info = load_and_preprocess_data(
        input_seq_len=INPUT_SEQ_LEN, output_seq_len=OUTPUT_SEQ_LEN, dataset=args.dataset, data_path=args.data_path or None
    )
    pd.DataFrame([split_info]).to_csv(os.path.join(results_dir, "data_split_ranges.csv"), index=False)
    
    # PyTorch Tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size, shuffle=False)

    model_aliases = {
        "pgfnet": "PGF-Net",
        "patchtst": "PatchTST",
        "dlinear": "DLinear",
        "lstm": "LSTM",
        "seasonalnaive": "S-Naive"
    }

    models_to_run = []
    for m in [m.strip() for m in args.models.split(",") if m.strip()]:
        key = "".join([c.lower() for c in m if c.isalnum()])
        if key in model_aliases:
            models_to_run.append(model_aliases[key])
            
    random_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    all_results = []
    training_curves = []
    preds_for_horizon = {}
    gate_values_list = []
    gate_load_series = None
    gate_hour_series = None

    ablation_variants = []
    if args.run_ablation:
        ablation_variants = [
            "PGF-Net w/o Gating",
            "PGF-Net w/o Transformer",
            "PGF-Net w/o LSTM",
            "PGF-Net w/o Time Embedding",
            "PGF-Net w/o PosEnc"
        ]

    expanded_models_to_run = list(models_to_run)
    if "PGF-Net" in expanded_models_to_run and args.run_ablation:
        expanded_models_to_run.extend([m for m in ablation_variants if m not in expanded_models_to_run])

    for model_name in expanded_models_to_run:
        print(f"--- Evaluating Model: {model_name} ---")
        
        for seed in random_seeds:
            print(f"  Running with seed: {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            y_pred_scaled = None
            inference_ms_per_batch = float("nan")
            num_params = 0
            
            if model_name in ['PGF-Net', 'LSTM', 'PatchTST', 'DLinear'] or model_name.startswith("PGF-Net "):
                # Initialize Model
                if model_name.startswith('PGF-Net'):
                    pgf_kwargs = {
                        "input_dim": 3,
                        "output_dim": OUTPUT_SEQ_LEN,
                        "d_model": 32,
                        "nhead": 2,
                        "num_layers": 1,
                        "lstm_hidden": 32,
                        "dropout": 0.0
                    }
                    if model_name == "PGF-Net w/o Gating":
                        model = PGFNet(**pgf_kwargs, use_gating=False).to(device)
                    elif model_name == "PGF-Net w/o Transformer":
                        model = PGFNet(**pgf_kwargs, use_transformer=False, use_gating=False).to(device)
                    elif model_name == "PGF-Net w/o LSTM":
                        model = PGFNet(**pgf_kwargs, use_lstm=False, use_gating=False).to(device)
                    elif model_name == "PGF-Net w/o Time Embedding":
                        model = PGFNet(**pgf_kwargs, use_time_embeddings=False).to(device)
                    elif model_name == "PGF-Net w/o PosEnc":
                        model = PGFNet(**pgf_kwargs, use_positional_encoding=False).to(device)
                    else:
                        model = PGFNet(**pgf_kwargs).to(device)
                elif model_name == 'LSTM':
                    model = LSTMModel(input_size=1, hidden_size=32, output_size=OUTPUT_SEQ_LEN, num_layers=2, dropout=0.1).to(device)
                elif model_name == 'PatchTST':
                    model = PatchTSTModel(input_size=1, output_size=OUTPUT_SEQ_LEN, seq_len=INPUT_SEQ_LEN).to(device)
                elif model_name == 'DLinear':
                    model = DLinearModel(input_size=1, output_size=OUTPUT_SEQ_LEN, seq_len=INPUT_SEQ_LEN).to(device)
                
                num_params = count_parameters(model)
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                criterion = nn.MSELoss()
                
                best_val_loss = float('inf')
                patience_counter = 0
                model_tag = _slugify(model_name)
                best_model_path = os.path.join(model_output_dir, f'best_{model_tag}_seed_{seed}.pt')
                train_losses = []
                val_losses = []
                train_start_time = time.perf_counter()
                for epoch in range(args.epochs):
                    train_loss = train_model(model, train_loader, optimizer, criterion, device, model_name, max_batches=max_train_batches)
                    val_loss = evaluate_model(model, val_loader, criterion, device, model_name, max_batches=max_val_batches)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    training_curves.append({
                        "Model": model_name,
                        "Seed": seed,
                        "Epoch": epoch + 1,
                        "TrainLoss": float(train_loss),
                        "ValLoss": float(val_loss)
                    })
                    print(f"  Epoch {epoch+1}/{args.epochs}, Train: {train_loss:.5f}, Val: {val_loss:.5f}", flush=True)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), best_model_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            print("  Early stopping.", flush=True)
                            break
                training_time_s = time.perf_counter() - train_start_time
                            
                # Load Best
                model.load_state_dict(torch.load(best_model_path))
                if model_name == "PGF-Net" and seed == random_seeds[0]:
                    y_pred_scaled, gate_values_list = predict(model, test_loader, device, model_name, max_batches=max_test_batches, return_gate=True)
                else:
                    y_pred_scaled = predict(model, test_loader, device, model_name, max_batches=max_test_batches)
                inference_ms_per_batch = measure_inference_ms_per_batch(model, test_loader, device, model_name, warmup_batches=3, measure_batches=min(10, max_test_batches_for_timing))
                
            elif model_name == 'S-Naive':
                num_params = 0
                # Last 24h of input (Channel 0 only)
                y_pred_naive = []
                count = 0
                max_samples = len(X_test) if max_test_batches is None else max_test_batches * args.batch_size
                for x_seq in X_test:
                    pred_seq = x_seq[-OUTPUT_SEQ_LEN:, 0]
                    y_pred_naive.append(pred_seq)
                    count += 1
                    if count >= max_samples:
                        break
                y_pred_scaled = np.array(y_pred_naive)
                y_pred_scaled = y_pred_scaled.reshape(-1, OUTPUT_SEQ_LEN, 1)
                inference_ms_per_batch = float("nan")
                training_time_s = 0.0

            # --- Evaluation ---
            # Ensure shapes
            num_samples = min(len(y_test), len(y_pred_scaled))
            y_pred_scaled = y_pred_scaled[:num_samples]
            y_test_eval = y_test[:num_samples]
            
            # Inverse Transform
            # Both are (N, 24, 1)
            y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
            y_test_inv = scaler.inverse_transform(y_test_eval.reshape(-1, 1)).reshape(y_test_eval.shape)
            
            metrics = calculate_metrics(y_test_inv, y_pred_inv)
            
            if model_name not in preds_for_horizon:
                preds_for_horizon[model_name] = []
            preds_for_horizon[model_name].append(y_pred_inv[: min(args.viz_samples, len(y_pred_inv))])

            if model_name == "PGF-Net" and seed == random_seeds[0]:
                n_preds = len(y_pred_scaled)
                n_samples_for_plot = min(n_preds, len(X_test)) 
                gate_load_series = scaler.inverse_transform(X_test[:n_samples_for_plot, -1, 0].reshape(-1, 1)).flatten()
                gate_hour_series = X_test[:n_samples_for_plot, -1, 1].astype(int).flatten()
            
            all_results.append({
                'Model': model_name,
                'Seed': seed,
                'Parameters': num_params,
                'Inference_ms_per_batch': inference_ms_per_batch,
                'Training_time_s': training_time_s,
                **metrics
            })
            print(f"  Metrics: {metrics}")

    # --- Save & Plot ---
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(results_dir, 'evaluation_results_per_seed.csv'), index=False)
    
    # Aggregated
    df_agg = df_results.groupby(['Model', 'Parameters']).agg(['mean', 'std']).reset_index()
    df_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_agg.columns.values]
    df_agg.to_csv(os.path.join(results_dir, 'evaluation_results.csv'), index=False)

    try:
        if training_curves:
            df_curves = pd.DataFrame(training_curves)
            df_curves.to_csv(os.path.join(results_dir, "training_curves.csv"), index=False)

        stats_rows = []
        pgf_df = df_results[df_results["Model"] == "PGF-Net"].copy()
        baselines = [m for m in df_results["Model"].unique() if m != "PGF-Net"]
        for metric in ["MSE", "PLE"]:
            pgf_m = pgf_df[["Seed", metric]].sort_values("Seed")
            for baseline in baselines:
                b_df = df_results[df_results["Model"] == baseline][["Seed", metric]].sort_values("Seed")
                merged = pd.merge(pgf_m, b_df, on="Seed", suffixes=("_pgf", "_base"))
                if len(merged) >= 3:
                    diff = merged[f"{metric}_base"] - merged[f"{metric}_pgf"]
                    denom = diff.std(ddof=1) if len(diff) > 1 else float("nan")
                    effect_size = float(diff.mean() / (denom + 1e-12)) if denom == denom else float("nan")
                    w_stat, w_p = stats.wilcoxon(merged[f"{metric}_pgf"], merged[f"{metric}_base"], alternative="two-sided")
                    t_stat, t_p = stats.ttest_rel(merged[f"{metric}_pgf"], merged[f"{metric}_base"])
                    stats_rows.append({"Metric": metric, "Baseline": baseline, "Test": "Wilcoxon", "N": int(len(merged)), "p_value": float(w_p), "effect_size_d": effect_size})
                    stats_rows.append({"Metric": metric, "Baseline": baseline, "Test": "Paired_t", "N": int(len(merged)), "p_value": float(t_p), "effect_size_d": effect_size})
        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(results_dir, "stat_tests_vs_pgfnet.csv"), index=False)
    except Exception as e:
        print(f"Stats processing failed: {e}", flush=True)
    
    # Plotting Horizon MSE
    print("Generating Plots...")
    try:
        from src.plotting import (
            plot_horizon_mse_comparison,
            plot_gate_analysis,
            plot_forecast_comparison,
            plot_data_decomposition,
            plot_robustness_boxplot,
            plot_training_curves,
            save_figure
        )

        n_common = min([min([p.shape[0] for p in lst]) for lst in preds_for_horizon.values() if lst])
        y_true_inv_common = scaler.inverse_transform(y_test[:n_common].reshape(-1, 1)).reshape(n_common, OUTPUT_SEQ_LEN, 1)
        y_true_inv_common_2d = np.squeeze(y_true_inv_common)

        preds_for_horizon_common = {}
        for name, lst in preds_for_horizon.items():
            preds_for_horizon_common[name] = [np.squeeze(p[:n_common]) for p in lst]

        plot_horizon_mse_comparison(preds_for_horizon_common, y_true_inv_common_2d, OUTPUT_SEQ_LEN, figures_dir, use_log_scale=True)

        if gate_load_series is not None and gate_hour_series is not None:
            plot_gate_analysis(gate_values_list, figures_dir, load_series=gate_load_series, hour_series=gate_hour_series)
        else:
            plot_gate_analysis(gate_values_list, figures_dir)

        preds_mean = {}
        preds_std = {}
        for name, lst in preds_for_horizon_common.items():
            stack = np.stack(lst, axis=0)
            preds_mean[name] = stack.mean(axis=0)
            preds_std[name] = stack.std(axis=0)

        n_plot = min(n_common, max(1, int(args.viz_samples)))
        preds_mean_plot = {k: v[:n_plot] for k, v in preds_mean.items()}
        preds_std_plot = {k: v[:n_plot] for k, v in preds_std.items()}
        plot_forecast_comparison(preds_mean_plot, X_test[:n_plot, :, 0:1], y_true_inv_common[:n_plot], scaler, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, figures_dir, prediction_std=preds_std_plot)

        plot_data_decomposition(df_resampled, figures_dir)
        plot_robustness_boxplot(df_results, figures_dir)
        if training_curves:
            plot_training_curves(pd.DataFrame(training_curves), figures_dir)

        try:
            df_eff = df_results.groupby("Model").agg(MSE=("MSE", "mean"), Inference_ms_per_batch=("Inference_ms_per_batch", "mean")).reset_index()
            df_eff = df_eff.sort_values("MSE")
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax2 = ax1.twinx()
            ax1.bar(df_eff["Model"], df_eff["MSE"], color="#4c72b0", alpha=0.8)
            ax2.plot(df_eff["Model"], df_eff["Inference_ms_per_batch"], color="#dd8452", marker="o", linewidth=2)
            ax1.set_ylabel("MSE")
            ax2.set_ylabel("Inference (ms/batch)")
            ax1.set_title("Accuracy-Efficiency Trade-off (CPU)")
            ax1.tick_params(axis="x", rotation=30)
            sns.despine(right=False)
            save_figure(figures_dir, "efficiency_comparison.png")
            plt.close()
        except Exception as e:
            print(f"Efficiency plot failed: {e}", flush=True)

        try:
            df_inf = df_results.groupby("Model").agg(
                Inference_mean=("Inference_ms_per_batch", "mean"),
                Inference_std=("Inference_ms_per_batch", "std")
            ).reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_inf["Model"], df_inf["Inference_mean"], yerr=df_inf["Inference_std"], capsize=4, color="#4c72b0", alpha=0.85)
            ax.set_ylabel("Inference (ms/batch)")
            ax.set_title("Inference Latency Comparison (CPU)")
            ax.tick_params(axis="x", rotation=30)
            sns.despine()
            save_figure(figures_dir, "inference_latency.png")
            plt.close()
        except Exception as e:
            print(f"Inference plot failed: {e}", flush=True)

        try:
            ablation_models = ["PGF-Net"] + ablation_variants
            df_ab = df_results[df_results["Model"].isin(ablation_models)].copy()
            if not df_ab.empty:
                agg = df_ab.groupby("Model").agg(MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")).reset_index()
                order = [m for m in ablation_models if m in set(agg["Model"])]
                agg["Model"] = pd.Categorical(agg["Model"], categories=order, ordered=True)
                agg = agg.sort_values("Model")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.bar(agg["Model"], agg["MSE_mean"], yerr=agg["MSE_std"], capsize=4, color="#55a868", alpha=0.85)
                ax.set_ylabel("MSE")
                ax.set_title("PGF-Net Ablation Study (MSE, mean±std)")
                ax.tick_params(axis="x", rotation=30)
                sns.despine()
                save_figure(figures_dir, "ablation_mse.png")
                plt.close()
        except Exception as e:
            print(f"Ablation plot failed: {e}", flush=True)

        try:
            eps = 1e-8
            peaks_true = np.max(np.squeeze(y_true_inv_common[:n_plot]), axis=1)
            q25, q50, q75 = np.quantile(peaks_true, [0.25, 0.5, 0.75])
            bin_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=float)
            labels = ["Low", "Medium", "High", "Peak"]
            bin_idx = np.digitize(peaks_true, bin_edges[1:-1], right=True)
            bin_metadata_rows = []
            for b, lab in enumerate(labels):
                mask = bin_idx[:n_plot] == b
                lower = bin_edges[b]
                upper = bin_edges[b + 1]
                bin_metadata_rows.append({
                    "Bin": lab,
                    "LowerBoundInclusive": float(lower) if np.isfinite(lower) else None,
                    "UpperBoundExclusive": float(upper) if np.isfinite(upper) else None,
                    "SampleCount": int(mask.sum()),
                    "Definition": "quartile_of_true_future_peak",
                    "PLEDefinition": "samplewise_relative_peak_error_then_average",
                })
            rows = []
            y_true = np.squeeze(y_true_inv_common[:n_plot])
            for model_name, preds_list in preds_for_horizon_common.items():
                display_name = "S-Naive" if model_name == "Seasonal Naive" else model_name
                for run_idx, pred in enumerate(preds_list):
                    y_pred = pred[:n_plot]
                    mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)
                    pred_peaks = np.max(y_pred, axis=1)
                    ple_per_sample = np.abs(peaks_true[:n_plot] - pred_peaks) / (peaks_true[:n_plot] + eps) * 100
                    for b, lab in enumerate(labels):
                        mask = bin_idx[:n_plot] == b
                        if mask.sum() == 0:
                            continue
                        rows.append({
                            "Model": display_name,
                            "Run": run_idx,
                            "Bin": lab,
                            "MAE": float(mae_per_sample[mask].mean()),
                            "PLE": float(ple_per_sample[mask].mean())
                        })
            df_bins = pd.DataFrame(rows)
            df_bins.to_csv(os.path.join(results_dir, "error_by_load_condition.csv"), index=False)
            pd.DataFrame(bin_metadata_rows).to_csv(
                os.path.join(results_dir, "error_by_load_condition_bin_metadata.csv"),
                index=False,
            )
            df_bins_agg = df_bins.groupby(["Model", "Bin"]).agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), PLE_mean=("PLE", "mean"), PLE_std=("PLE", "std")).reset_index()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            for metric, ax in [("MAE", axes[0]), ("PLE", axes[1])]:
                mean_pivot = df_bins_agg.pivot(index="Bin", columns="Model", values=f"{metric}_mean").reindex(labels)
                std_pivot = df_bins_agg.pivot(index="Bin", columns="Model", values=f"{metric}_std").reindex(labels)
                mean_pivot.plot(kind="bar", ax=ax, yerr=std_pivot, capsize=4)
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} by Load Condition (Quartile Bins)")
                ax.tick_params(axis="x", rotation=0)
            sns.despine()
            save_figure(figures_dir, "error_by_load_condition.png")
            plt.close()
        except Exception as e:
            print(f"Error-by-load plot failed: {e}", flush=True)

        try:
            scatter_models = [m for m in ["PGF-Net", "DLinear", "S-Naive"] if m in preds_mean_plot]
            if len(scatter_models) >= 2:
                y_true = np.squeeze(y_true_inv_common[:n_plot])
                peaks_true = np.max(y_true, axis=1)
                fig, ax = plt.subplots(figsize=(6, 6))
                peak_arrays = [peaks_true]
                for model_name in scatter_models:
                    model_peaks = np.max(preds_mean_plot[model_name], axis=1)
                    peak_arrays.append(model_peaks)
                    ax.scatter(peaks_true, model_peaks, alpha=0.5, label=model_name, s=15)
                lims = [min(arr.min() for arr in peak_arrays), max(arr.max() for arr in peak_arrays)]
                ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel("Actual Peak Load")
                ax.set_ylabel("Predicted Peak Load")
                ax.set_title("Peak Prediction Scatter")
                ax.legend()
                sns.despine()
                save_figure(figures_dir, "peak_scatter.png")
                plt.close()
        except Exception as e:
            print(f"Peak scatter plot failed: {e}", flush=True)

        try:
            line_models = [m for m in ["PGF-Net", "DLinear", "S-Naive"] if m in preds_mean_plot]
            if line_models:
                y_true = np.squeeze(y_true_inv_common[:n_plot])
                peaks_true = np.max(y_true, axis=1)
                peak_idx = int(np.argmax(peaks_true))
                peak_h = int(np.argmax(y_true[peak_idx]))
                fig, ax = plt.subplots(figsize=(10, 4.5))
                t_axis = np.arange(OUTPUT_SEQ_LEN)
                ax.plot(t_axis, y_true[peak_idx], label="Ground Truth", color="black", linewidth=2.8)
                for model_name in line_models:
                    ax.plot(t_axis, np.squeeze(preds_mean_plot[model_name])[peak_idx], label=model_name, linewidth=2.2)
                ax.axvspan(max(peak_h - 1, 0), min(peak_h + 1, OUTPUT_SEQ_LEN - 1), color="#f2c14e", alpha=0.18)
                ax.set_xlabel("Forecast horizon")
                ax.set_ylabel("Load")
                ax.set_title("Peak-event Forecast Trajectory")
                ax.legend(ncol=max(2, len(line_models)), frameon=True)
                sns.despine()
                save_figure(figures_dir, "peak_event_trajectory.png")
                plt.close()
        except Exception as e:
            print(f"Peak trajectory plot failed: {e}", flush=True)

        try:
            if preds_for_horizon_common:
                y_true = np.squeeze(y_true_inv_common[:n_plot])
                peaks_true = np.max(y_true, axis=1)
                threshold = np.quantile(peaks_true, 0.9)
                peak_mask = peaks_true >= threshold
                rows = []
                for model_name, preds_list in preds_for_horizon_common.items():
                    for pred in preds_list:
                        y_pred = np.squeeze(pred[:n_plot])
                        residuals = (y_pred - y_true)
                        peak_res = residuals[peak_mask]
                        rows.append({
                            "Model": model_name,
                            "Residual_Mean": float(np.mean(peak_res)),
                            "Residual_Std": float(np.std(peak_res))
                        })
                df_peak = pd.DataFrame(rows)
                df_peak.to_csv(os.path.join(results_dir, "peak_residuals_summary.csv"), index=False)
                fig, ax = plt.subplots(figsize=(10, 5))
                df_peak.groupby("Model")["Residual_Mean"].mean().plot(kind="bar", yerr=df_peak.groupby("Model")["Residual_Std"].mean(), ax=ax, capsize=4)
                ax.set_ylabel("Residual at Peak Times")
                ax.set_title("Residual Bias at Peak Times (Top 10%)")
                ax.tick_params(axis="x", rotation=30)
                sns.despine()
                save_figure(figures_dir, "peak_residuals.png")
                plt.close()
        except Exception as e:
            print(f"Peak residual plot failed: {e}", flush=True)
        
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("========= Experiment Complete =========")

if __name__ == "__main__":
    run_experiment()
