import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import numpy as np
import sys
from data_preprocessing import DataProcessor
from lstm_model import LSTMModel
from transformer_model import TransformerModel
from hybrid_model import HybridModel, FusionType
from dlinear_model import DLinearModel
from patchtst_model import PatchTSTModel
from utils import plot_loss, set_seed, plot_predictions
from logger import setup_logger
from paths import output_path, ensure_dir

# Initialize logger
logger = setup_logger("train_logger", output_path("logs", "train.log"))

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        try:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {self.path}: {e}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', model_name='model', patience=5):
    train_losses = []
    val_losses = []
    
    # Path to save the best model
    save_dir = output_path("model_output")
    try:
        ensure_dir(save_dir)
    except OSError as e:
        logger.error(f"Failed to create directory {save_dir}: {e}")
        return model

    save_path = os.path.join(save_dir, f'{model_name}_best.pth')
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    
    logger.info(f"Starting training for {model_name} on {device}...")
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Check Early Stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
                
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state...")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Training process ended in {duration:.2f} seconds.")

    # Plot loss
    try:
        plot_dir = ensure_dir(output_path("figures"))
        plot_path = os.path.join(plot_dir, f'{model_name}_loss.png')
        plot_loss(train_losses, val_losses, save_path=plot_path)
        logger.info(f"Loss plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to plot loss: {e}")
    
    # Load the best model if it exists
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path))
            logger.info(f"Loaded best model from {save_path}")
        except Exception as e:
            logger.error(f"Failed to load best model from {save_path}: {e}")
    else:
        logger.warning(f"Best model file not found at {save_path}. Returning current model state.")
        
    return model

def save_gate_analysis(model, test_loader, device, save_dir=None):
    """
    Run inference on test set and save gate values for analysis.
    Only for HybridModel with ADAPTIVE fusion.
    """
    if not isinstance(model, HybridModel) or model.fusion_type != FusionType.ADAPTIVE:
        return
    
    if save_dir is None:
        save_dir = output_path("analysis")

    ensure_dir(save_dir)
    model.eval()
    gate_values = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            # Forward with return_gate=True
            _, gates = model(X_batch, return_gate=True)
            # gates shape: (batch_size, hidden_size) or (batch_size, 1) depending on implementation
            # In hybrid_model.py, z is (batch_size, hidden_size) if gate_fc outputs hidden_size
            # But wait, gate_fc outputs hidden_size? 
            # Check hybrid_model.py: 
            # self.gate_fc = nn.Linear(hidden_size * 2, hidden_size)
            # z = self.sigmoid(self.gate_fc(combined))
            # So z is per-feature weighting.
            
            # We can average z across hidden dimension to get a single scalar per time step "gate intensity"
            # or save the whole thing. Let's save the mean per sample.
            gates_mean = gates.mean(dim=1).cpu().numpy()
            gate_values.extend(gates_mean)
            
    np.save(os.path.join(save_dir, 'gate_values.npy'), np.array(gate_values))
    logger.info(f"Saved gate values to {os.path.join(save_dir, 'gate_values.npy')}")

def main():
    # Determine the absolute path to the data file
    # Assuming the script is in src/ and data is in data/ (sibling to src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(current_dir, '..', 'data', 'continuous dataset.csv')

    parser = argparse.ArgumentParser(description='Train Electricity Demand Forecasting Models')
    parser.add_argument('--model', type=str, default='hybrid', choices=['lstm', 'transformer', 'hybrid', 'dlinear', 'patchtst'], help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to data file')
    parser.add_argument('--seq_len', type=int, default=24, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction horizon')
    parser.add_argument('--fusion_type', type=str, default='adaptive', choices=['adaptive', 'concat', 'fixed'], help='Fusion type for Hybrid model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    try:
        set_seed(42)
        if args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize Data Processor
        logger.info("Initializing DataProcessor...")
        processor = DataProcessor(args.data_path, seq_len=args.seq_len, pred_len=args.pred_len)
        
        # Get Data Loaders and Input Size
        logger.info("Loading data...")
        (train_loader, val_loader, test_loader, input_size) = processor.get_data_loaders(batch_size=args.batch_size)
        
        logger.info(f"Input feature size: {input_size}, Prediction Horizon: {args.pred_len}")
        
        # Initialize Model
        logger.info(f"Initializing {args.model} model...")
        if args.model == 'lstm':
            model = LSTMModel(input_size=input_size, output_size=args.pred_len).to(device)
        elif args.model == 'transformer':
            model = TransformerModel(input_size=input_size, output_size=args.pred_len).to(device)
        elif args.model == 'hybrid':
            model = HybridModel(input_size=input_size, output_size=args.pred_len, fusion_type=args.fusion_type).to(device)
        elif args.model == 'dlinear':
            model = DLinearModel(input_size=input_size, output_size=args.pred_len, seq_len=args.seq_len).to(device)
        elif args.model == 'patchtst':
            model = PatchTSTModel(input_size=input_size, output_size=args.pred_len, seq_len=args.seq_len).to(device)
            
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Train
        train_model(model, train_loader, val_loader, criterion, optimizer, 
                    num_epochs=args.epochs, device=device, model_name=f"{args.model}_{args.fusion_type}" if args.model=='hybrid' else args.model, patience=args.patience)
        
        # Post-training Analysis (Gate Values)
        if args.model == 'hybrid' and args.fusion_type == 'adaptive':
            logger.info("Running Gate Analysis on Test Set...")
            save_gate_analysis(model, test_loader, device)

        # Generate Predictions Plot on Test Set
        logger.info("Generating predictions on Test Set...")
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.numpy())
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Take the first step prediction for plotting
        if args.pred_len > 1:
            pred_to_plot = all_preds[:, 0].reshape(-1, 1)
            target_to_plot = all_targets[:, 0].reshape(-1, 1)
        else:
            pred_to_plot = all_preds
            target_to_plot = all_targets
            
        # Inverse Transform
        try:
            pred_inv = processor.scaler_y.inverse_transform(pred_to_plot)
            target_inv = processor.scaler_y.inverse_transform(target_to_plot)
        except Exception as e:
            logger.warning(f"Could not inverse transform: {e}. Plotting scaled values.")
            pred_inv = pred_to_plot
            target_inv = target_to_plot
            
        # Plot
        plot_path = os.path.join(output_path("figures"), f'{args.model}_predictions.png')
        if args.model == 'hybrid':
             plot_path = os.path.join(output_path("figures"), f'{args.model}_{args.fusion_type}_predictions.png')
             
        plot_predictions(target_inv.flatten(), pred_inv.flatten(), 
                        title=f'{args.model} Predictions vs True (1-step ahead)', 
                        save_path=plot_path)
        logger.info(f"Prediction plot saved to {plot_path}")

    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
