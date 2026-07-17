import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant PE matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Add PE to input
        return x + self.pe[:, :x.size(1), :]

class PGFNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=32,
        nhead=2,
        num_layers=1,
        lstm_hidden=32,
        dropout=0.1,
        use_time_embeddings=True,
        use_transformer=True,
        use_lstm=True,
        use_gating=True,
        use_positional_encoding=True,
        fusion_mode=None,
        fixed_alpha=0.5,
    ):
        super(PGFNet, self).__init__()
        self.d_model = d_model
        self.use_transformer = use_transformer
        self.use_lstm = use_lstm
        self.use_positional_encoding = use_positional_encoding
        if fusion_mode is None:
            fusion_mode = "gated" if use_gating else "fixed_average"
        self.fusion_mode = fusion_mode
        self.fixed_alpha = fixed_alpha
        self.use_gating = fusion_mode == "gated" and use_transformer and use_lstm
        
        # Enhanced Input Processing with Time Embeddings
        # We expect input_dim to be 3 if time features are present: [Load, Hour, Day]
        if input_dim >= 3 and use_time_embeddings:
            self.use_time_embeddings = True
            # Project the scalar load value to d_model
            self.input_proj = nn.Linear(1, d_model) 
            # Embeddings for Hour (0-23) and Day (0-6)
            self.hour_embed = nn.Embedding(24, d_model)
            self.day_embed = nn.Embedding(7, d_model)
        else:
            self.use_time_embeddings = False
            self.input_proj = nn.Linear(input_dim, d_model)
        
        # Parallel branches
        # 1. Transformer Branch
        if self.use_transformer:
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 2. LSTM Branch
        if self.use_lstm:
            self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Gated Fusion
        # Learnable gate to weight LSTM vs Transformer features
        # Input to gate is concatenation of both branches' last hidden states
        if self.use_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(lstm_hidden + d_model, d_model),
                nn.Sigmoid()
            )
        elif self.use_transformer and self.use_lstm and self.fusion_mode == "scalar":
            self.scalar_gate_logit = nn.Parameter(torch.zeros(1))
        
        # Output layer
        if self.use_transformer and self.use_lstm and self.fusion_mode == "concat":
            output_input_dim = lstm_hidden + d_model
        elif self.use_transformer:
            output_input_dim = d_model
        else:
            output_input_dim = lstm_hidden
        self.output_layer = nn.Linear(output_input_dim, output_dim)

    def forward(self, src, return_gate=False):
        # src shape: (batch_size, seq_len, input_dim)
        
        # Project input
        if self.use_time_embeddings:
            # Slice inputs
            load = src[..., 0:1]       # (Batch, Seq, 1)
            hour = src[..., 1].long()  # (Batch, Seq)
            day = src[..., 2].long()   # (Batch, Seq)
            
            # Embed and Sum
            # Broadcasting embeddings to match load projection
            # hour_embed(hour) -> (Batch, Seq, d_model)
            src_proj = self.input_proj(load) + self.hour_embed(hour) + self.day_embed(day)
        else:
            src_proj = self.input_proj(src) # -> (batch_size, seq_len, d_model)
        
        transformer_last_hidden = None
        if self.use_transformer:
            if self.use_positional_encoding:
                src_t = self.pos_encoder(src_proj)
            else:
                src_t = src_proj
            transformer_output = self.transformer_encoder(src_t)
            transformer_last_hidden = transformer_output[:, -1, :]
        
        # LSTM Branch
        lstm_last_hidden = None
        if self.use_lstm:
            lstm_output, _ = self.lstm(src_proj)
            lstm_last_hidden = lstm_output[:, -1, :]
        
        gate_values = None
        if self.use_transformer and self.use_lstm:
            combined_hidden = torch.cat((transformer_last_hidden, lstm_last_hidden), dim=1)
            if self.fusion_mode == "gated":
                gate_values = self.gate_net(combined_hidden)
                fused_hidden = gate_values * transformer_last_hidden + (1 - gate_values) * lstm_last_hidden
            elif self.fusion_mode == "fixed_average":
                fused_hidden = (
                    self.fixed_alpha * transformer_last_hidden
                    + (1 - self.fixed_alpha) * lstm_last_hidden
                )
            elif self.fusion_mode == "scalar":
                scalar_gate = torch.sigmoid(self.scalar_gate_logit)
                gate_values = scalar_gate.expand_as(transformer_last_hidden)
                fused_hidden = scalar_gate * transformer_last_hidden + (1 - scalar_gate) * lstm_last_hidden
            elif self.fusion_mode == "concat":
                fused_hidden = combined_hidden
            else:
                raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        elif self.use_transformer:
            fused_hidden = transformer_last_hidden
        else:
            fused_hidden = lstm_last_hidden
        
        # Output layer
        output = self.output_layer(fused_hidden) # -> (batch_size, output_dim)
        
        if return_gate:
            return output.unsqueeze(-1), gate_values
            
        return output.unsqueeze(-1) # -> (batch_size, output_dim, 1)
