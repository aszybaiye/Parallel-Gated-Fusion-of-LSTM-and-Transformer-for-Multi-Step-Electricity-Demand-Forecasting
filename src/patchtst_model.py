import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchTSTModel(nn.Module):
    """
    Simplified PatchTST implementation
    """
    def __init__(self, input_size, output_size, seq_len, patch_len=16, stride=8, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(PatchTSTModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = output_size
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size
        
        # Calculate number of patches
        # N = (L - P) / S + 1
        self.num_patches = int(math.ceil((seq_len - patch_len) / stride) + 1)
        self.total_len = (self.num_patches - 1) * stride + patch_len
        
        # Padding might be needed if exact fit isn't possible, but for now assume it fits or is handled
        # If not, we might lose some data or need padding logic. 
        # For simplicity, we ensure seq_len works with patch/stride or we pad.
        
        # Embedding: Project patch to d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Flatten Head
        self.head = nn.Linear(self.num_patches * d_model, output_size)
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        # For PatchTST, we often treat channels independently (Channel Independence).
        # Here, let's just use the first channel (Target) for simplicity, or process all and flatten.
        # Given the task is univariate target forecasting (mostly), let's focus on the first channel.
        
        x = x[:, :, 0:1]
        x = x.squeeze(-1)
        if x.shape[1] < self.total_len:
            pad_len = self.total_len - x.shape[1]
            x = F.pad(x, (0, pad_len), mode="replicate")
        
        # Patching
        # Unfold: [Batch, Num_Patches, Patch_Len]
        # We need to handle the case where patches don't perfectly cover seq_len
        # Using unfold: dimension 1, size patch_len, step stride
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: [Batch, Num_Patches, Patch_Len]
        
        # Embedding
        enc_in = self.patch_embedding(patches) # [Batch, Num_Patches, d_model]
        
        # Add Positional Embedding
        # pos_embedding is [1, Num_Patches, d_model]
        if enc_in.shape[1] > self.pos_embedding.shape[1]:
             # Handle size mismatch if any (dynamic padding)
             enc_in = enc_in[:, :self.pos_embedding.shape[1], :]
        
        enc_in = enc_in + self.pos_embedding[:, :enc_in.shape[1], :]
        
        # Transformer
        enc_out = self.transformer_encoder(enc_in) # [Batch, Num_Patches, d_model]
        
        # Flatten and Project
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        out = self.head(enc_out)
        
        return out
