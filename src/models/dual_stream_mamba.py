"""Dual-stream Mamba model for sleep stage classification from PPG signals.

Adapts the dual-stream cross-attention model (DS-CA) of Wang et al. (2025),
replacing the cross-attention fusion and TCN context stages with Mamba blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import SAMPLES_PER_WINDOW
from models.blocks import (
    AdaptiveModalityWeighting,
    DynamicSinusoidalEncoding,
    ScaledEncoder,
    TemporalTCNBlocks,
)
from models.blocks_mamba import CrossModalMambaFusion, TemporalMambaBlock
from models.naming import format_model_name
from models.noise import DEFAULT_NOISE_CONFIG, add_noise_to_ppg


class DualStreamMamba(nn.Module):
    """Dual-stream PPG model using Mamba SSM blocks for both per-stream
    temporal modelling and cross-modal fusion."""

    NAME = "Dual-stream Mamba"
    _DEPTH_LABELS = {3: "Shallow", 6: "Deep"}

    def __init__(
        self,
        n_classes=4,
        d_model=256,
        n_fusion_blocks=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.2,
        noise_config=None,
        use_tcn=False,
        tcn_kernel_size=7,
        tcn_dilations=None,
        fusion_ffn_expand=2,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_fusion_blocks = n_fusion_blocks
        self.use_tcn = use_tcn
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dilations = tcn_dilations or [1, 2, 4]
        self.fusion_ffn_expand = fusion_ffn_expand

        self.noise_config = noise_config or DEFAULT_NOISE_CONFIG

        self.clean_ppg_encoder = ScaledEncoder(d_model)
        self.noisy_ppg_encoder = ScaledEncoder(d_model)

        self.positional_encoding = DynamicSinusoidalEncoding(d_model)

        self.modality_weighting = AdaptiveModalityWeighting(d_model)

        self.fusion_blocks = nn.ModuleList(
            [
                CrossModalMambaFusion(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    dropout,
                    layer_idx=i,
                    fusion_ffn_expand=fusion_ffn_expand,
                )
                for i in range(n_fusion_blocks)
            ]
        )

        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
        )

        if use_tcn:
            self.temporal_blocks = TemporalTCNBlocks(
                d_model,
                kernel_size=tcn_kernel_size,
                dropout=dropout,
                num_blocks=len(self.tcn_dilations),
                dilations=self.tcn_dilations,
            )
        else:
            self.temporal_blocks = nn.Sequential(
                TemporalMambaBlock(
                    d_model, d_state, d_conv, expand, dropout, layer_idx=0
                ),
                TemporalMambaBlock(
                    d_model, d_state, d_conv, expand, dropout, layer_idx=1
                ),
                TemporalMambaBlock(
                    d_model, d_state, d_conv, expand, dropout, layer_idx=2
                ),
            )

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

    @classmethod
    def from_config(cls, config):
        mc = config.get("model", {})
        return cls(
            d_model=mc.get("d_model", 256),
            n_fusion_blocks=mc.get("n_fusion_blocks", 3),
            d_state=mc.get("d_state", 16),
            d_conv=mc.get("d_conv", 4),
            expand=mc.get("expand", 2),
            dropout=mc.get("dropout", 0.2),
            noise_config=mc.get("noise_config"),
            use_tcn=mc.get("use_tcn", False),
            tcn_kernel_size=mc.get("tcn_kernel_size", 7),
            tcn_dilations=mc.get("tcn_dilations", None),
            fusion_ffn_expand=mc.get("fusion_ffn_expand", 2),
        )

    def get_name(self):
        params = {
            "d_model": self.d_model,
            "n_fusion_blocks": self.n_fusion_blocks,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "use_tcn": self.use_tcn,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_dilations": self.tcn_dilations,
            "fusion_ffn_expand": self.fusion_ffn_expand,
        }
        return format_model_name(self.NAME, params)

    def get_short_name(self):
        depth = self._DEPTH_LABELS.get(self.n_fusion_blocks, f"{self.n_fusion_blocks}B")
        return f"DS-M[{self.d_model}, {depth}]"

    def forward(self, ppg):
        n_epochs = ppg.size(2) // SAMPLES_PER_WINDOW

        ppg_unfiltered = add_noise_to_ppg(ppg, self.noise_config)

        clean_features = self.clean_ppg_encoder(ppg)
        noisy_features = self.noisy_ppg_encoder(ppg_unfiltered)

        clean_features = self.positional_encoding(clean_features)
        noisy_features = self.positional_encoding(noisy_features)

        clean_weight, noisy_weight = self.modality_weighting(
            clean_features, noisy_features
        )

        clean_features_weighted = clean_features * clean_weight
        noisy_features_weighted = noisy_features * noisy_weight

        # (B, C, T) -> (B, T, C) for Mamba fusion blocks
        clean_features_t = clean_features_weighted.transpose(1, 2)
        noisy_features_t = noisy_features_weighted.transpose(1, 2)

        for fusion_block in self.fusion_blocks:
            clean_features_t, noisy_features_t = fusion_block(
                clean_features_t, noisy_features_t
            )

        clean_features = clean_features_t.transpose(1, 2)
        noisy_features = noisy_features_t.transpose(1, 2)

        combined_features = torch.cat([clean_features, noisy_features], dim=1)
        fused_features = self.feature_aggregation(combined_features)

        temporal_features = self.temporal_blocks(fused_features)

        refined_features = self.feature_refinement(temporal_features)

        # Map from encoder temporal resolution to one prediction per epoch
        output_features = F.interpolate(
            refined_features, size=n_epochs, mode="linear", align_corners=False
        )

        output = self.classifier(output_features)
        output = F.softmax(output, dim=1)

        return output
