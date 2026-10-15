# models.py
# TCN for sequence-to-sequence error regression (predict delta for x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (B, C, T)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = F.relu(self.bn1(y))
        y = self.dropout1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = F.relu(self.bn2(y))
        y = self.dropout2(y)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(y + res)

class TCNRegressor(nn.Module):
    """
    Input:  (B, Cin, T)
    Output: (B, Cout, T)  delta sequence
    """
    def __init__(self, cin, cout, channels=(64, 64, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        n_in = cin
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(
                n_in, ch, kernel_size=kernel_size, dilation=2**i, dropout=dropout
            ))
            n_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(n_in, cout, kernel_size=1)

    def forward(self, x):
        h = self.tcn(x)
        out = self.head(h)
        return out