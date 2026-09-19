"""SleepPPG-Net, the efficiency baseline: a single-stream convolutional encoder
followed by two dilated TCN stacks.

Keeps its own ResConvBlock: unlike the one in models.blocks, it skips the residual
projection when the channel count is unchanged, so the parameters differ.

Sourced from: https://github.com/DavyWJW/sleep-staging-models
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
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

        residual = F.max_pool1d(residual, kernel_size=2, stride=2)

        return x + residual


class Chomp1d(nn.Module):
    """Trims extra samples from causal (left-only) padding on dilated convolutions,
    keeping output length = input length."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
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
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

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
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        # Kept from the reference TCN. weight_norm rebuilds conv1 and conv2's weight
        # from weight_g and weight_v on every forward pass, so this init only lasts
        # on the downsample conv, which SleepPPGNet's 128-to-128 blocks do not have.
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        # 6 levels with dilations 1 to 32. The convolutions are causal, and at kernel
        # size 7 one stack reaches back 756 epochs, so the two stacked in SleepPPGNet
        # see every earlier epoch of a 1,200-epoch night.
        num_levels = 6
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SleepPPGNet(nn.Module):
    NAME = "SleepPPG-Net"

    def __init__(self):
        super(SleepPPGNet, self).__init__()

        self.resconv_blocks = nn.Sequential(
            ResConvBlock(1, 16),
            ResConvBlock(16, 16),
            ResConvBlock(16, 32),
            ResConvBlock(32, 32),
            ResConvBlock(32, 64),
            ResConvBlock(64, 64),
            ResConvBlock(64, 128),
            ResConvBlock(128, 256),
        )

        # 1024 = channels(256) * sub-epoch windows(4), reduced to 128 per epoch
        self.dense = nn.Linear(1024, 128)
        self.tcnblock1 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2)
        self.tcnblock2 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2)
        self.final_conv = nn.Conv1d(128, 4, 1)

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_name(self):
        return self.NAME

    def get_short_name(self):
        return "SleepPPG-Net"

    def forward(self, x):
        x = self.resconv_blocks(x)

        batch_size, channels, length = x.shape

        # After 8 stride-2 ResConvBlocks the signal is downsampled 256x.
        # We group the result into 1200 epoch-aligned windows:
        # (B, 256, L) -> (B, 256, 1200, L//1200) split into epochs + sub-windows
        x = x.view(batch_size, channels, 1200, length // 1200)
        # -> (B, 256, L//1200, 1200) swap so epochs are the sequence dim
        x = x.permute(0, 1, 3, 2).contiguous()
        # -> (B, 256*L//1200, 1200) merge channels and sub-windows into features
        x = x.view(batch_size, -1, 1200)

        # Dense reduces (256 * sub-windows) features per epoch to 128
        x = x.transpose(1, 2)
        x = self.dense(x)
        x = x.transpose(1, 2)

        x = self.tcnblock1(x)
        x = self.tcnblock2(x)

        x = self.final_conv(x)
        x = F.softmax(x, dim=1)

        return x
