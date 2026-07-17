
# PatchTST model based on https://github.com/timeseriesAI/tsai/blob/main/tsai/models/PatchTST.py
# and the paper "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional

# helper modules
class _Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.transpose(-1, -2)
        return x

class _DataNorm(nn.Module):
    def __init__(self, c_in, affine=True, subtract_last=False):
        super().__init__()
        self.subtract_last = subtract_last
        self.norm = nn.LayerNorm(c_in, elementwise_affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        if self.subtract_last:
            x = x - x[..., -1:].detach()
        return self.norm(x)

class _PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class _TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout_1(src2)
        src = self.norm_1(src)
        src2 = self.ff(src)
        src = src + self.dropout_2(src2)
        src = self.norm_2(src)
        return src

class _TSTEncoder(nn.Module):
    def __init__(self, encoder_layer, n_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(n_layers)])
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class _PatchTST_backbone(nn.Module):
    def __init__(self, c_in, seq_len, pred_dim, patch_len, stride, n_layers, n_heads, d_model, d_ff, dropout, revin=True, affine=True):
        super().__init__()
        self.revin = revin
        if self.revin:
            self.revin_layer = _DataNorm(c_in, affine=affine, subtract_last=False)

        # Patching
        self.patching = _Patching(patch_len=patch_len, stride=stride)
        n_patches = int((seq_len - patch_len) / stride + 1)

        # Backbone
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.input_embedding = nn.Linear(patch_len, d_model)
        self.positional_encoding = _PositionalEncoder(d_model, max_len=n_patches)
        encoder_layer = _TSTEncoderLayer(d_model, n_heads, d_ff=d_ff, dropout=dropout)
        self.encoder = _TSTEncoder(encoder_layer, n_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(n_patches * d_model, pred_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs x n_vars x seq_len]
        if self.revin:
            x = self.revin_layer(x)
        
        # do patching
        x = self.padding_patch_layer(x)
        x = self.patching(x) # x: [bs x n_vars x n_patches x patch_len]
        
        # input embedding
        x = self.input_embedding(x) # x: [bs x n_vars x n_patches x d_model]
        
        # rearrange for encoder
        x = x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3]) # x: [bs*n_patches x n_vars x d_model]
        x = self.positional_encoding(x)
        
        # encoder
        z = self.encoder(x) # z: [bs*n_patches x n_vars x d_model]
        
        # rearrange for head
        z = z.reshape(-1, self.patching.patch_len, z.shape[1], z.shape[2]).permute(0, 2, 1, 3) # z: [bs x n_vars x n_patches x d_model]
        
        # head
        return self.head(z)


class PatchTST(nn.Module):
    def __init__(self, c_in, seq_len, pred_dim, patch_len=16, stride=8, n_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, revin=True, affine=True):
        super().__init__()
        self.model = _PatchTST_backbone(c_in=c_in, seq_len=seq_len, pred_dim=pred_dim,
                                      patch_len=patch_len, stride=stride, n_layers=n_layers, n_heads=n_heads,
                                      d_model=d_model, d_ff=d_ff, dropout=dropout, revin=revin, affine=affine)

    def forward(self, x):
        # x: [bs x seq_len x n_vars]
        x = x.permute(0, 2, 1)    # x: [bs x n_vars x seq_len]
        x = self.model(x)         # x: [bs x n_vars x pred_dim]
        x = x.permute(0, 2, 1)    # x: [bs x pred_dim x n_vars]
        return x
