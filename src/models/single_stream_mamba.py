"""Single-stream Mamba model (SS-M) for sleep stage classification from PPG.

The dual-stream model with its second encoder and fusion stage removed: one
encoder, positional encoding, a stack of bidirectional Mamba blocks, and a
pointwise classifier."""

import torch.nn as nn
import torch.nn.functional as F

from common.constants import SAMPLES_PER_WINDOW
from models.blocks import DynamicSinusoidalEncoding, ScaledEncoder
from models.blocks_mamba import TemporalMambaBlock
from models.naming import format_model_name


class SingleStreamMamba(nn.Module):
    """Single-stream PPG model using pre-norm bidirectional Mamba blocks
    for temporal modelling."""

    NAME = "Single-stream Mamba"
    _DEPTH_LABELS = {6: "Shallow", 9: "Deep"}

    @classmethod
    def from_config(cls, config):
        mc = config.get("model", {})
        return cls(
            d_model=mc.get("d_model", 256),
            n_mamba_blocks=mc.get("n_mamba_blocks", 4),
            d_state=mc.get("d_state", 16),
            d_conv=mc.get("d_conv", 4),
            expand=mc.get("expand", 2),
            dropout=mc.get("dropout", 0.2),
        )

    def __init__(
        self,
        n_classes=4,
        d_model=256,
        n_mamba_blocks=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.2,
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for positional encoding, got {d_model}"
            )

        self.d_model = d_model
        self.n_classes = n_classes
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_mamba_blocks = n_mamba_blocks

        self.encoder = ScaledEncoder(d_model)

        self.positional_encoding = DynamicSinusoidalEncoding(d_model)

        temporal_blocks = []
        for i in range(n_mamba_blocks):
            temporal_blocks.append(
                TemporalMambaBlock(
                    d_model, d_state, d_conv, expand, dropout, layer_idx=i
                )
            )

        self.temporal_blocks = nn.Sequential(*temporal_blocks)

        self.feature_refinement = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, n_classes, kernel_size=1),
        )

    def get_name(self):
        params = {
            "d_model": self.d_model,
            "n_mamba_blocks": self.n_mamba_blocks,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
        }
        return format_model_name(self.NAME, params)

    def get_short_name(self):
        depth = self._DEPTH_LABELS.get(self.n_mamba_blocks, f"{self.n_mamba_blocks}B")
        return f"SS-M[{self.d_model}, {depth}]"

    def forward(self, ppg):
        n_epochs = ppg.size(2) // SAMPLES_PER_WINDOW

        features = self.encoder(ppg)
        features = self.positional_encoding(features)
        features = self.temporal_blocks(features)
        features = self.feature_refinement(features)

        # Map from encoder temporal resolution to one prediction per epoch
        features = F.interpolate(
            features, size=n_epochs, mode="linear", align_corners=False
        )

        output = self.classifier(features)
        return F.softmax(output, dim=1)
