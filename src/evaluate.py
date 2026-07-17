import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
import numpy as np
import sys
from data_preprocessing import DataProcessor
from lstm_model import LSTMModel
from transformer_model import TransformerModel
from hybrid_model import HybridModel
from dlinear_model import DLinearModel
from patchtst_model import PatchTSTModel
from utils import calculate_metrics, plot_predictions, save_metrics, set_seed
from logger import setup_logger
from paths import output_path, ensure_dir

# Initialize logger
logger = setup_logger("evaluate_logger", output_path("logs", "evaluate.log"))

def evaluate_model(model, test_loader, scaler_y, device='cpu', model_name='model'):
    model.eval()
    predictions = []
    targets = []
    
    logger.info(f"Evaluating {model_name}...")
    
    try:
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions.append(outputs.cpu().numpy())
                targets.append(y_batch.numpy())
                
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Inverse transform
        # The scaler expects shape (n_samples, 1) for target
        try:
            pred_shape = predictions.shape
            target_shape = targets.shape
            
            # Reshape to (-1, 1) for scaler, then back to original shape
            predictions_inv = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(pred_shape)
            targets_inv = scaler_y.inverse_transform(targets.reshape(-1, 1)).reshape(target_shape)
        except ValueError as e:
            logger.error(f"Error during inverse transformation: {e}")
            raise e
        
        # Calculate metrics
        metrics = calculate_metrics(targets_inv, predictions_inv)
        metrics['Model'] = model_name
        logger.info(f"Metrics for {model_name}: {metrics}")
        
        # Save metrics
        try:
            save_metrics(metrics, output_path("evaluation_results.csv"))
            logger.info(f"Metrics saved to {output_path('evaluation_results.csv')}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
        
        # Plot
        try:
            plot_dir = ensure_dir(output_path("figures"))
            plot_path = os.path.join(plot_dir, f'{model_name}_predictions.png')
            
            # For plotting, if multi-step, use the first step (t+1)
            y_true_plot = targets_inv[:, 0] if targets_inv.ndim > 1 else targets_inv
            y_pred_plot = predictions_inv[:, 0] if predictions_inv.ndim > 1 else predictions_inv
            
            plot_predictions(y_true_plot, y_pred_plot, title=f'{model_name} Predictions vs True Values (Step 1)', save_path=plot_path)
            logger.info(f"Prediction plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot predictions: {e}")
        
        return metrics

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        raise e

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(current_dir, '..', 'data', 'continuous dataset.csv')

    parser = argparse.ArgumentParser(description='Evaluate Electricity Demand Forecasting Models')
    parser.add_argument('--model', type=str, default='hybrid', choices=['lstm', 'transformer', 'hybrid', 'dlinear', 'patchtst'], help='Model type to evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
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
        # Note: We need test_loader and input_size (to initialize model)
        logger.info("Loading data...")
        _, _, test_loader, input_size = processor.get_data_loaders(batch_size=args.batch_size)
        
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
            
        # Load Weights
        model_tag = f"{args.model}_{args.fusion_type}" if args.model == 'hybrid' else args.model
        weights_path = os.path.join(output_path("model_output"), f'{model_tag}_best.pth')
        if os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.error(f"Failed to load weights from {weights_path}: {e}")
                sys.exit(1)
        else:
            logger.error(f"Weights file not found at {weights_path}. Cannot evaluate.")
            sys.exit(1)
        
        # Evaluate
        evaluate_model(model, test_loader, processor.scaler_y, device=device, model_name=model_tag)

    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
