import torch
import torch.nn as nn

class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    DLinear: Decomposition Linear Model
    """
    def __init__(self, input_size, output_size, seq_len, kernel_size=25):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = output_size
        
        # Decomposition Kernel Size
        self.decompsition = SeriesDecomp(kernel_size)
        
        # Linear layers for Seasonal and Trend
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # Handling multivariate input if needed (Independent strategy: shared weights)
        # For simplicity here, we assume input_size=1 (univariate) or we apply same linear to all channels
        # If input_size > 1, we might need a Linear layer per channel or treat channels independently.
        # Here we assume we project the feature dimension first if needed, but DLinear typically works on time axis.
        # Let's assume input_size=1 for the main target, or we apply to the last dim.
        
        self.channels = input_size
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        seasonal_init, trend_init = self.decompsition(x)
        # seasonal_init: [Batch, Input length, Channel]
        # trend_init: [Batch, Input length, Channel]
        
        # Permute to [Batch, Channel, Input length] for Linear layer over time
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        
        # Permute back to [Batch, Output length, Channel]
        x = x.permute(0, 2, 1)
        
        # If we need strictly [Batch, Output length] for single target, we might slice or project
        # Assuming we want to predict all channels or just the first one.
        # The existing models output [Batch, Pred_Len] (1D per sample).
        # So we should take the first channel if channels > 1
        
        if self.channels > 1:
            x = x[:, :, 0] # Take the first feature (Load) as target
        else:
            x = x.squeeze(-1)
            
        return x
