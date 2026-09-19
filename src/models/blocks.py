"""Convolutional building blocks shared by the PPG sleep staging models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class ResConvBlock(nn.Module):
    """Three 3-tap convolutions with batch norm, max-pooled by `stride`, plus a
    projected residual. Follows the SleepPPG-Net encoder block."""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

        # The residual path needs the same channel projection and downsampling in
        # time as the main path, so the two can be added
        if in_channels != out_channels or stride != 1:
            self.residual_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.MaxPool1d(kernel_size=stride, stride=stride),
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        return x + residual


class ScaledEncoder(nn.Sequential):
    """Nine stride-2 ResConvBlocks, a 512x reduction, so each 1,024-sample
    epoch becomes two d_model feature vectors. The models interpolate down to
    one per epoch after the temporal stage.

    Channel widths grow as fractions of d_model, with per-layer minimums so
    narrow models do not bottleneck in the early layers (following the
    SleepPPG-Net progression)."""

    def __init__(self, d_model):
        ratios = [1 / 16, 1 / 8, 1 / 8, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 1, 1]
        minimums = [16, 32, 32, 64, 64, 128, 128, 256, 256]

        channels = [1]
        for ratio, minimum in zip(ratios, minimums):
            channels.append(min(d_model, max(minimum, int(d_model * ratio))))
        channels[-1] = d_model

        blocks = [
            ResConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ]
        super().__init__(*blocks)


class DynamicSinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding computed for whatever sequence length
    arrives, so the same module serves full nights and short windows."""

    def __init__(self, d_model):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.d_model = d_model
        inv_freq = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """Add positional encoding to input (batch, d_model, seq_len)."""
        batch_size, d_model, seq_len = x.shape
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        sinusoid_inp = position.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        pos_emb = torch.zeros(seq_len, d_model, device=x.device, dtype=x.dtype)
        # Sin on even dims, cos on odd gives each position a unique signature
        pos_emb[:, 0::2] = torch.sin(sinusoid_inp)
        pos_emb[:, 1::2] = torch.cos(sinusoid_inp)
        pos_emb = pos_emb.transpose(0, 1).unsqueeze(0)
        return x + pos_emb


class AdaptiveModalityWeighting(nn.Module):
    """Learn a scalar importance weight for the clean and noisy PPG streams."""

    def __init__(self, d_model):
        super().__init__()
        self.clean_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, 1),
            nn.Sigmoid(),
        )
        self.noisy_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, clean_features, noisy_features):
        clean_weight = self.clean_gate(clean_features)
        noisy_weight = self.noisy_gate(noisy_features)

        # Normalise so clean + noisy sum to ~1.0, which stops the fused signal growing
        total_weight = clean_weight + noisy_weight
        clean_weight = clean_weight / (total_weight + 1e-8)
        noisy_weight = noisy_weight / (total_weight + 1e-8)

        return clean_weight, noisy_weight


class TemporalBlock(nn.Module):
    """Dilated temporal convolution block with weight-normalised convolutions,
    used by the TCN ablation of the dual-stream model."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        # "Same" padding for dilated convolutions keeps output length = input length
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )

        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvBlock(nn.Module):
    """Dilated temporal convolution block with a ReLU residual, used by the
    cross-attention baseline's temporal stage."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        # "Same" padding for dilated convolutions keeps output length = input length
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))

        if self.residual is not None:
            x = self.residual(x)

        return self.relu(out + x)


class TemporalTCNBlocks(nn.Module):
    """Stack of dilated temporal convolutional blocks."""

    def __init__(
        self, d_model, kernel_size=7, dropout=0.2, num_blocks=3, dilations=None
    ):
        super().__init__()
        # Exponential dilations (1, 2, 4, ...) double the receptive field per block
        if dilations is None:
            dilations = [2**i for i in range(num_blocks)]

        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x):
        return self.blocks(x)
