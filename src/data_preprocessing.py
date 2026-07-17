import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os
from logger import setup_logger
from paths import output_path

# Setup logger
logger = setup_logger(__name__, log_file=output_path("logs", "data_processing.log"))

class ElectricityDataset(Dataset):
    """
    Custom Dataset for Electricity Load Forecasting.
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class DataProcessor:
    """
    Handles data loading, preprocessing, and splitting.
    """
    def __init__(self, file_path, target_col='nat_demand', seq_len=24, pred_len=24, train_split=0.7, val_split=0.15):
        self.file_path = file_path
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_split = train_split
        self.val_split = val_split
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols = [] 
        
    def load_data(self):
        """
        Loads data from CSV, parses dates, and handles missing values.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"Data file not found at {self.file_path}")
            raise FileNotFoundError(f"Data file not found at {self.file_path}")
            
        logger.info(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise e
        
        if df.empty:
            logger.error("Data file is empty.")
            raise ValueError("Data file is empty.")

        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Parse datetime
        if 'datetime' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
            except Exception as e:
                logger.error(f"Error parsing datetime column: {e}")
                raise e
        else:
            logger.warning("'datetime' column not found. Using index as time.")
            
        # Handle missing values
        if df.isnull().values.any():
            logger.info("Found missing values. Interpolating...")
            df = df.interpolate(method='time')
            df.fillna(method='bfill', inplace=True) 
            df.fillna(method='ffill', inplace=True)
            
        # Feature Engineering
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Select features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.target_col not in numeric_cols:
             logger.error(f"Target column '{self.target_col}' not found in dataset.")
             raise ValueError(f"Target column '{self.target_col}' not found.")

        # Ensure target column is first
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
            numeric_cols = [self.target_col] + numeric_cols

        self.feature_cols = numeric_cols
        logger.info(f"Selected features: {self.feature_cols}")
        
        return df[self.feature_cols]

    def preprocess(self, df):
        """
        Normalizes data and creates sequences.
        """
        try:
            data_values = df.values
            target_values = df[self.target_col].values.reshape(-1, 1)
            
            split_idx = int(len(df) * self.train_split)
            
            train_data = data_values[:split_idx]
            train_target = target_values[:split_idx]
            
            self.scaler_X.fit(train_data)
            self.scaler_y.fit(train_target)
            
            data_scaled = self.scaler_X.transform(data_values)
            target_scaled = self.scaler_y.transform(target_values)
            
            return data_scaled, target_scaled
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise e

    def create_sequences(self, data_scaled, target_scaled):
        """
        Creates (X, y) sequences for time series forecasting.
        X shape: (samples, seq_len, num_features)
        y shape: (samples, pred_len)
        """
        X, y = [], []
        # Ensure we have enough data for the last sequence
        for i in range(len(data_scaled) - self.seq_len - self.pred_len + 1):
            X.append(data_scaled[i:i+self.seq_len])
            # Handle multi-step prediction
            if self.pred_len == 1:
                y.append(target_scaled[i+self.seq_len])
            else:
                y.append(target_scaled[i+self.seq_len : i+self.seq_len+self.pred_len].flatten())
            
        return np.array(X), np.array(y)

    def get_data_loaders(self, batch_size=64):
        """
        Main method to get DataLoaders.
        """
        try:
            df = self.load_data()
            data_scaled, target_scaled = self.preprocess(df)
            X, y = self.create_sequences(data_scaled, target_scaled)
            
            total_len = len(X)
            if total_len == 0:
                logger.error("No sequences created. Check sequence length vs data size.")
                raise ValueError("Insufficient data to create sequences.")

            train_size = int(total_len * self.train_split)
            val_size = int(total_len * self.val_split)
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
            
            logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
            
            train_loader = DataLoader(ElectricityDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(ElectricityDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(ElectricityDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader, test_loader, X_train.shape[2] 
        except Exception as e:
            logger.error(f"Failed to get data loaders: {e}")
            raise e

def load_and_preprocess_data(file_path, batch_size=64, seq_len=24):
    """
    Wrapper function for backward compatibility.
    """
    processor = DataProcessor(file_path, seq_len=seq_len)
    return processor.get_data_loaders(batch_size), processor.scaler_y
