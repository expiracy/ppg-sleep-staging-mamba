"""Mamba-based building blocks for PPG sleep staging models."""

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba with pre-norm, fusion projection, and residual."""

    def __init__(
        self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, layer_idx=None
    ):
        super().__init__()

        # layer_idx only matters for mamba_ssm's inference-time state cache,
        # which these models never use. The two directions are kept distinct anyway.
        forward_idx = layer_idx * 2 if layer_idx is not None else None
        backward_idx = layer_idx * 2 + 1 if layer_idx is not None else None

        self.norm = nn.LayerNorm(d_model)
        # d_state: SSM hidden state size (memory capacity)
        # d_conv: local conv width in Mamba's gating
        # expand: inner dimension multiplier
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=forward_idx,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=backward_idx,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm(x)
        fwd = self.forward_mamba(normed)
        # Mamba is causal, so flip input/output to get backward context
        bwd = self.backward_mamba(normed.flip(dims=[1])).flip(dims=[1])
        combined = torch.cat([fwd, bwd], dim=-1)
        return self.fusion(combined) + x


class TemporalMambaBlock(nn.Module):
    """BidirectionalMambaBlock + pre-norm FFN with residual.

    Input and output are (B, D, T), like the convolutional stages around it.
    """

    def __init__(
        self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, layer_idx=None
    ):
        super().__init__()

        self.mamba_block = BidirectionalMambaBlock(
            d_model, d_state, d_conv, expand, dropout, layer_idx
        )

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # (B, D, T) -> (B, T, D)
        x_t = x.transpose(1, 2)
        x_t = self.mamba_block(x_t)
        x_t = x_t + self.ffn(self.norm(x_t))
        # (B, T, D) -> (B, D, T)
        return x_t.transpose(1, 2)


class CrossModalMambaFusion(nn.Module):
    """Cross-modal fusion using BidirectionalMambaBlock with pre-norm refine FFNs."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        layer_idx=None,
        fusion_ffn_expand=2,
    ):
        super().__init__()

        self.clean_modal_embed = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )  # small init to avoid dominating early training
        self.noisy_modal_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.joint_mamba = BidirectionalMambaBlock(
            d_model, d_state, d_conv, expand, dropout, layer_idx
        )

        d_ffn = d_model * fusion_ffn_expand

        self.clean_norm = nn.LayerNorm(d_model)
        self.clean_refine = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

        self.noisy_norm = nn.LayerNorm(d_model)
        self.noisy_refine = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, clean_features, noisy_features):
        batch_size, seq_len, d_model = clean_features.shape

        clean_with_modal = clean_features + self.clean_modal_embed
        noisy_with_modal = noisy_features + self.noisy_modal_embed

        # Interleave clean/noisy tokens so the SSM sees alternating modalities,
        # learning cross-modal dependencies through its recurrent state
        interleaved = torch.stack([clean_with_modal, noisy_with_modal], dim=2)
        interleaved = interleaved.view(batch_size, seq_len * 2, d_model)
        processed = self.joint_mamba(interleaved)
        processed = processed.view(batch_size, seq_len, 2, d_model)  # de-interleave

        # Pre-norm refine with residual from the Mamba block's output
        clean_processed = processed[:, :, 0, :]
        clean_out = clean_processed + self.clean_refine(
            self.clean_norm(clean_processed)
        )

        noisy_processed = processed[:, :, 1, :]
        noisy_out = noisy_processed + self.noisy_refine(
            self.noisy_norm(noisy_processed)
        )

        return clean_out, noisy_out
