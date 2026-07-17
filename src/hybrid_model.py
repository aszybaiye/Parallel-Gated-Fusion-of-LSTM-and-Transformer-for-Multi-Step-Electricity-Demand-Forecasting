import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FusionType:
    ADAPTIVE = 'adaptive'
    CONCAT = 'concat'
    FIXED = 'fixed'

class HybridModel(nn.Module):
    """
    Parallel Hybrid Model with Gated Fusion.
    Branch 1: LSTM (Local Temporal Patterns)
    Branch 2: Transformer (Global Dependencies)
    Fusion: Gating Mechanism to weight contributions.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers_lstm=1, num_layers_transformer=1, output_size=1, dropout=0.1, fusion_type=FusionType.ADAPTIVE):
        super(HybridModel, self).__init__()
        
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # --- Branch 1: LSTM (Local) ---
        # Captures sequential evolution and short-term dependencies
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers_lstm, batch_first=True, dropout=dropout if num_layers_lstm > 1 else 0)
        
        # --- Branch 2: Transformer (Global) ---
        # Captures long-range dependencies via Self-Attention with Causal Masking
        # Enhanced with Time Embeddings if input_size >= 3 (Load, Hour, Day)
        if input_size >= 3:
            self.use_time_embeddings = True
            self.input_projection = nn.Linear(1, hidden_size) # Process only Load
            self.hour_embed = nn.Embedding(24, hidden_size)
            self.day_embed = nn.Embedding(7, hidden_size)
        else:
            self.use_time_embeddings = False
            self.input_projection = nn.Linear(input_size, hidden_size)
            
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_transformer)
        
        # --- Fusion: Gating Mechanism ---
        # Learnable gate z to weigh LSTM vs Transformer features
        # z is a vector of size (hidden_size,), applying element-wise (per feature dimension) gating.
        # z = sigmoid(W * [h_lstm; h_trans] + b)
        if self.fusion_type == FusionType.ADAPTIVE:
            # Maps 2*hidden -> hidden. 
            # Parameter count: (2*hidden * hidden) + hidden (bias)
            self.gate_fc = nn.Linear(hidden_size * 2, hidden_size)
            self.sigmoid = nn.Sigmoid()
        elif self.fusion_type == FusionType.CONCAT:
            # Concat: [h_lstm, h_trans] -> Linear -> hidden
            # This ensures parameter count is comparable to Adaptive (both have a 2h->h projection)
            self.concat_fc = nn.Linear(hidden_size * 2, hidden_size)
        
        # --- Output Layer ---
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, return_gate=False):
        # x shape: (batch_size, seq_len, input_size)
        
        # 1. LSTM Branch
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step feature
        lstm_feat = lstm_out[:, -1, :] # (batch_size, hidden_size)
        
        # 2. Transformer Branch
        if self.use_time_embeddings:
            # Assume x is (Load, Hour, Day, ...)
            load = x[..., 0:1]
            hour = x[..., 1].long()
            day = x[..., 2].long()
            
            # Embed and Sum
            trans_in = self.input_projection(load) + self.hour_embed(hour) + self.day_embed(day)
        else:
            trans_in = self.input_projection(x)
            
        trans_in = trans_in * math.sqrt(self.hidden_size)
        
        # Apply Positional Encoding (CRITICAL FIX)
        trans_in = self.pos_encoder(trans_in)
        
        # Generate Causal Mask to prevent information leakage
        # mask shape: (seq_len, seq_len)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        
        # Apply Transformer with Causal Mask
        trans_out = self.transformer_encoder(trans_in, mask=mask)
        # Take the last time step feature
        trans_feat = trans_out[:, -1, :] # (batch_size, hidden_size)
        
        # 3. Fusion
        z = None
        
        if self.fusion_type == FusionType.ADAPTIVE:
            # Concatenate features: (batch_size, 2 * hidden_size)
            combined = torch.cat((lstm_feat, trans_feat), dim=1)
            # Calculate Gate value z (0 to 1)
            z = self.sigmoid(self.gate_fc(combined))
            # Weighted sum: z * LSTM + (1-z) * Transformer
            fused_feat = z * lstm_feat + (1 - z) * trans_feat
            
        elif self.fusion_type == FusionType.FIXED:
            # Fixed 0.5 weight
            z = torch.full_like(lstm_feat, 0.5)
            fused_feat = 0.5 * lstm_feat + 0.5 * trans_feat
            
        elif self.fusion_type == FusionType.CONCAT:
            combined = torch.cat((lstm_feat, trans_feat), dim=1)
            fused_feat = torch.relu(self.concat_fc(combined))
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # 4. Final Prediction
        out = self.fc(fused_feat)
        
        if return_gate:
            return out, z
            
        return out
