"""
FI-Mamba: Frequency-Domain Lossless Mamba with Prior Collaboration,
Isotropic Unbiased State-Space Learning and Multi-Scale CLIP for
Perinatal Brain Ultrasound Image Classification.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _pairwise_off_diagonal_mean(similarity: Tensor) -> Tensor:
    """Mean off-diagonal similarity for a [B, M, M] matrix."""
    m = similarity.shape[-1]
    if m < 2:
        return similarity.new_zeros(similarity.shape[0], 1)
    eye = torch.eye(m, device=similarity.device, dtype=torch.bool)
    values = similarity.masked_select(~eye.unsqueeze(0)).view(similarity.shape[0], m, m - 1)
    return values.mean(dim=-1)


def _tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
    """[B, N, C] -> [B, C, H, W]."""
    batch, length, channels = tokens.shape
    if length != height * width:
        raise ValueError(f"Token length {length} does not match H×W={height * width}.")
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def _map_to_tokens(feature: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, N, C]."""
    return feature.flatten(2).transpose(1, 2)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for a 2-D feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class FrequencyPriorCollaborativeModule(nn.Module):
    """
    Reconstructable Laplacian-residual decomposition plus local visual prior.

    low  = upsample(avg_pool(x))
    high = x - low
    x    = low + high                 (exact residual reconstruction)

    The prior branch uses the normalized maximum absolute difference in a
    3×3 neighborhood to modulate a local residual. No frequency band is
    discarded by this module.
    """

    def __init__(self, in_channels: int = 1, pool_size: int = 2, eps: float = 1e-6) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("The paper configuration expects a grayscale input.")
        self.pool_size = pool_size
        self.eps = eps
        self.local_projection = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        nn.init.constant_(self.local_projection.weight, 1.0 / 9.0)

    def _local_max_difference(self, x: Tensor) -> Tensor:
        patches = F.unfold(x, kernel_size=3, padding=1)
        batch, _, locations = patches.shape
        patches = patches.view(batch, x.shape[1], 9, locations)
        center = patches[:, :, 4:5, :]
        max_difference = (patches - center).abs().amax(dim=2)
        max_difference = max_difference.view_as(x)

        minimum = max_difference.amin(dim=(-2, -1), keepdim=True)
        maximum = max_difference.amax(dim=(-2, -1), keepdim=True)
        return (max_difference - minimum) / (maximum - minimum + self.eps)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        low = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        high = x - low

        visual_prior = self._local_max_difference(x)
        local_context = self.local_projection(x)
        prior = x + visual_prior * (x - local_context)

        # [B, branches=3, channels=1, H, W]
        branches = torch.stack((low, high, prior), dim=1)
        reconstruction = low + high
        return {
            "branches": branches,
            "low": low,
            "high": high,
            "prior": prior,
            "reconstruction": reconstruction,
        }



class IOFSSSM(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        conv_kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inner_dim = dim * expansion
        self.in_projection = nn.Linear(dim, self.inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.inner_dim,
        )
        self.delta_projection = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_projection = nn.Linear(self.inner_dim, self.inner_dim)
        self.c_projection = nn.Linear(self.inner_dim, self.inner_dim)
        self.log_a = nn.Parameter(torch.full((self.inner_dim,), -2.0))
        self.skip = nn.Parameter(torch.ones(self.inner_dim))
        self.out_projection = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _non_causal_accumulate(decay: Tensor, drive: Tensor) -> Tensor:
        log_decay = torch.log(decay.clamp(min=1e-5, max=1.0))
        full_log = torch.cumsum(log_decay, dim=1)
        forward_weight = torch.exp(full_log)
        backward_weight = torch.exp(torch.flip(full_log, dims=(1,)))
        forward_acc = torch.cumsum(drive / forward_weight.clamp(min=1e-8), dim=1) * forward_weight
        backward_acc = torch.cumsum(torch.flip(drive, dims=(1,)) / backward_weight.clamp(min=1e-8), dim=1) * backward_weight
        backward_acc = torch.flip(backward_acc, dims=(1,))
        global_h = (forward_acc + backward_acc) / 2.0
        return global_h

    def forward(self, x: Tensor) -> Tensor:
        value, input_gate = self.in_projection(x).chunk(2, dim=-1)
        value = self.depthwise_conv(value.transpose(1, 2))[..., : x.shape[1]]
        value = F.silu(value.transpose(1, 2))

        delta = F.softplus(self.delta_projection(value)) + 1e-4
        state_rate = F.softplus(self.log_a).view(1, 1, -1)
        decay = torch.exp(-delta * state_rate).clamp(min=1e-5, max=1.0)

        input_map = torch.sigmoid(self.b_projection(value))
        output_map = torch.sigmoid(self.c_projection(value))
        drive = delta * input_map * value

        state = self._non_causal_accumulate(decay, drive)
        output = output_map * state + self.skip.view(1, 1, -1) * value
        output = output * torch.sigmoid(input_gate)
        return self.dropout(self.out_projection(output))


class IsotropicUnbiasedScan(nn.Module):
    """
    IU-Scan: adaptively fuse forward and reverse selective state-space paths.

    The same SSM parameters are shared by both directions. This avoids assigning
    a privileged parameter set to either scan origin while the learned gate
    remains content-adaptive.
    """

def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.ssm = IOFSSSM(dim, expansion=expansion, dropout=dropout)
        self.direction_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
forward_seq = x
        forward_state = self.ssm(forward_seq)

        backward_seq = torch.flip(x, dims=(1))
        backward_raw = self.ssm(backward_seq)
        backward_state = torch.flip(backward_raw, dims=(1))

        gate = self.direction_gate(x)
        fused = gate * forward_state + (1.0 - gate) * backward_state

        out = self.output_norm(fused)
        return out, gate



class OrthogonalFrequencySpectrumGating(nn.Module):
    """
    Full-spectrum, three-band, content-adaptive gating.

    The radial masks form a soft partition of unity. Therefore the three raw
    band tensors sum back to the source feature (up to FFT numerical error)
    before nonlinear reweighting. A residual path retains the unmodified source.

    hard_token_mask=False is the scientifically conservative default. A dense
    zero mask alone does not lower real FLOPs; physical packing plus a sparse or
    variable-length downstream kernel is required for computational savings.
    """

    def __init__(
        self,
        dim: int,
        windows: Sequence[int] = (1, 3, 5, 7),
        num_bands: int = 3,
        temperature: float = 0.08,
        hard_token_mask: bool = False,
        threshold_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if num_bands != 3:
            raise ValueError("The FI-Mamba paper configuration uses three frequency bands.")
        if any(window % 2 == 0 or window < 1 for window in windows):
            raise ValueError("Local amplitude windows must be positive odd numbers.")

        self.dim = dim
        self.windows = tuple(windows)
        self.num_bands = num_bands
        self.temperature = temperature
        self.hard_token_mask = hard_token_mask
        self.threshold_scale = threshold_scale

        self.band_centers = nn.Parameter(torch.tensor([0.05, 0.45, 0.85]))
        amplitude_channels = num_bands * len(self.windows)
        hidden = max(dim // 2, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(amplitude_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.band_gate = nn.Sequential(
            nn.Linear(dim * num_bands, dim),
            nn.GELU(),
            nn.Linear(dim, num_bands),
        )
        self.output_projection = nn.Linear(dim, dim)

    def _soft_radial_masks(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.fftfreq(width, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.amax().clamp_min(1e-6)

        centers = self.band_centers.sigmoid().sort().values.to(dtype=dtype)
        logits = -((radius.unsqueeze(0) - centers[:, None, None]) ** 2) / self.temperature
        return torch.softmax(logits, dim=0)

    def _local_amplitudes(self, bands: Sequence[Tensor]) -> Tensor:
        amplitude_maps: List[Tensor] = []
        for band in bands:
            magnitude = band.abs().mean(dim=1, keepdim=True)
            for window in self.windows:
                amplitude_maps.append(
                    F.avg_pool2d(
                        magnitude,
                        kernel_size=window,
                        stride=1,
                        padding=window // 2,
                    )
                )
        return torch.cat(amplitude_maps, dim=1)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        source = _tokens_to_map(tokens, height, width)
        spectrum = torch.fft.fft2(source.float(), norm="ortho")
        masks = self._soft_radial_masks(height, width, source.device, source.dtype)

        bands = [
            torch.fft.ifft2(spectrum * masks[index], norm="ortho").real.to(source.dtype)
            for index in range(self.num_bands)
        ]

        # Content-adaptive band fusion. The source residual prevents full-band
        # information from being irreversibly deleted by the learned gates.
        pooled_bands = torch.cat([band.mean(dim=(-2, -1)) for band in bands], dim=-1)
        band_weights = torch.softmax(self.band_gate(pooled_bands), dim=-1)
        mixed = sum(
            band * band_weights[:, index, None, None, None]
            for index, band in enumerate(bands)
        )

        amplitude_features = self._local_amplitudes(bands)
        spatial_gate = self.spatial_gate(amplitude_features)
        token_scores = spatial_gate.mean(dim=1, keepdim=True)

        score_mean = token_scores.mean(dim=(-2, -1), keepdim=True)
        score_std = token_scores.std(dim=(-2, -1), keepdim=True, unbiased=False)
        threshold = score_mean - self.threshold_scale * score_std
        if self.hard_token_mask:
            token_mask = (token_scores >= threshold).to(spatial_gate.dtype)
        else:
            # Smooth approximation keeps gradients and does not claim sparse FLOPs.
            token_mask = torch.sigmoid((token_scores - threshold) / 0.1)

        gated = mixed * spatial_gate * token_mask
        output = tokens + self.output_projection(_map_to_tokens(gated))

        diagnostics = {
            "band_weights": band_weights,
            "token_gate": token_scores.flatten(2).transpose(1, 2),
            "token_mask": token_mask.flatten(2).transpose(1, 2),
            "frequency_partition_error": (masks.sum(dim=0) - 1.0).abs().mean(),
        }
        return output, diagnostics


class IOFSBlock(nn.Module):
    """IU-Scan followed by OFSG and a channel MLP, all with residual paths."""

    def __init__(
        self,
        dim: int,
        ssm_expansion: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        hard_token_mask: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.iu_scan = IsotropicUnbiasedScan(dim, expansion=ssm_expansion, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ofsg = OrthogonalFrequencySpectrumGating(
            dim,
            hard_token_mask=hard_token_mask,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        iu_output, direction_gate = self.iu_scan(self.norm1(tokens))
        tokens = tokens + iu_output
        tokens, frequency_diagnostics = self.ofsg(self.norm2(tokens), height, width)
        tokens = tokens + self.mlp(self.norm3(tokens))
        frequency_diagnostics["direction_gate"] = direction_gate
        return tokens, frequency_diagnostics


class IOFSStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        ssm_expansion: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        hard_token_mask: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IOFSBlock(
                    dim=dim,
                    ssm_expansion=ssm_expansion,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    hard_token_mask=hard_token_mask,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, feature: Tensor) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        height, width = feature.shape[-2:]
        tokens = _map_to_tokens(feature)
        diagnostics: List[Dict[str, Tensor]] = []
        for block in self.blocks:
            tokens, block_diagnostics = block(tokens, height, width)
            diagnostics.append(block_diagnostics)
        return _tokens_to_map(tokens, height, width), diagnostics


class SharedPatchEmbedding(nn.Module):
    def __init__(self, out_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            1,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = LayerNorm2d(out_dim)

    def forward(self, branches: Tensor) -> Tensor:
        batch, branch_count, channels, height, width = branches.shape
        x = branches.reshape(batch * branch_count, channels, height, width)
        x = self.norm(self.projection(x))
        return x.view(batch, branch_count, *x.shape[1:])


class SharedPatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.projection = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, branches: Tensor) -> Tensor:
        batch, branch_count, channels, height, width = branches.shape
        x = branches.reshape(batch * branch_count, channels, height, width)
        x = self.projection(self.norm(x))
        return x.view(batch, branch_count, *x.shape[1:])


class BranchMixer(nn.Module):
    """Lightweight residual collaboration across low/high/prior branches."""

    def __init__(self, branch_count: int = 3) -> None:
        super().__init__()
        self.mix = nn.Linear(branch_count, branch_count, bias=False)
        nn.init.eye_(self.mix.weight)

    def forward(self, branches: Tensor) -> Tensor:
        # [B, branches, C, H, W] -> apply a shared branch transform.
        transposed = branches.permute(0, 2, 3, 4, 1)
        mixed = self.mix(transposed).permute(0, 4, 1, 2, 3)
        return branches + 0.1 * mixed


class IOFSMambaEncoder(nn.Module):
    """Four-stage shared-weight encoder for the three FPC branches."""

    def __init__(
        self,
        dims: Sequence[int] = (48, 96, 192, 384),
        depths: Sequence[int] = (2, 2, 4, 2),
        ssm_expansion: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        hard_token_mask: bool = False,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("FI-Mamba uses a four-stage hierarchy.")
        self.dims = tuple(dims)
        self.patch_embedding = SharedPatchEmbedding(dims[0], patch_size=4)
        self.stages = nn.ModuleList(
            [
                IOFSStage(
                    dim=dim,
                    depth=depth,
                    ssm_expansion=ssm_expansion,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    hard_token_mask=hard_token_mask,
                )
                for dim, depth in zip(dims, depths)
            ]
        )
        self.branch_mixers = nn.ModuleList([BranchMixer(3) for _ in dims])
        self.downsamples = nn.ModuleList(
            [SharedPatchMerging(dims[index], dims[index + 1]) for index in range(3)]
        )

    def forward(
        self,
        branches: Tensor,
    ) -> Tuple[List[Tensor], List[List[Dict[str, Tensor]]]]:
        x = self.patch_embedding(branches)
        multi_scale_features: List[Tensor] = []
        all_diagnostics: List[List[Dict[str, Tensor]]] = []

        for stage_index, stage in enumerate(self.stages):
            batch, branch_count, channels, height, width = x.shape
            merged_batch = x.reshape(batch * branch_count, channels, height, width)
            merged_batch, stage_diagnostics = stage(merged_batch)
            x = merged_batch.view(batch, branch_count, channels, height, width)
            x = self.branch_mixers[stage_index](x)

            # Save [B, branches, tokens, channels] for visual CLIP interaction.
            prototypes_input = x.flatten(3).permute(0, 1, 3, 2)
            multi_scale_features.append(prototypes_input)
            all_diagnostics.append(stage_diagnostics)

            if stage_index < len(self.downsamples):
                x = self.downsamples[stage_index](x)

        return multi_scale_features, all_diagnostics



class MultiScaleCLIPInteraction(nn.Module):
    """
    CLIP-style normalized cosine interaction without a text encoder.

    First, three visual branches are aligned and fused within each scale.
    Then the four scale-level vectors are aligned and fused globally.
    """

    def __init__(
        self,
        stage_dims: Sequence[int],
        fusion_dim: int,
        num_classes: int,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(stage_dim),
                    nn.Linear(stage_dim, fusion_dim),
                )
                for stage_dim in stage_dims
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, features: Sequence[Tensor]) -> Dict[str, Tensor | List[Tensor]]:
        if len(features) != len(self.projections):
            raise ValueError("One feature tensor is required for every encoder stage.")

        projected_branch_prototypes: List[Tensor] = []
        scale_vectors: List[Tensor] = []
        branch_weights: List[Tensor] = []

        for feature, projection in zip(features, self.projections):
            # Global pooling: [B, 3, N, C] -> [B, 3, C].
            prototypes = feature.mean(dim=2)
            prototypes = F.normalize(projection(prototypes), dim=-1)
            similarity = prototypes @ prototypes.transpose(-1, -2)
            branch_scores = _pairwise_off_diagonal_mean(similarity)
            weights = torch.softmax(branch_scores / self.temperature, dim=-1)
            scale_vector = (weights.unsqueeze(-1) * prototypes).sum(dim=1)
            scale_vector = F.normalize(scale_vector, dim=-1)

            projected_branch_prototypes.append(prototypes)
            branch_weights.append(weights)
            scale_vectors.append(scale_vector)

        stacked_scales = torch.stack(scale_vectors, dim=1)
        scale_similarity = stacked_scales @ stacked_scales.transpose(-1, -2)
        scale_scores = _pairwise_off_diagonal_mean(scale_similarity)
        scale_weights = torch.softmax(scale_scores / self.temperature, dim=-1)
        global_vector = (scale_weights.unsqueeze(-1) * stacked_scales).sum(dim=1)
        global_vector = F.normalize(global_vector, dim=-1)
        logits = self.classifier(global_vector)

        return {
            "logits": logits,
            "global_feature": global_vector,
            "branch_prototypes": projected_branch_prototypes,
            "scale_features": stacked_scales,
            "branch_weights": torch.stack(branch_weights, dim=1),
            "scale_weights": scale_weights,
        }



class ClassBalancedCrossEntropy(nn.Module):
    """Effective-number class weighting for imbalanced four-class datasets."""

    def __init__(
        self,
        class_counts: Optional[Sequence[int]] = None,
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_counts is None:
            weights = torch.empty(0)
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1)
            effective_number = 1.0 - torch.pow(torch.tensor(beta), counts)
            weights = (1.0 - beta) / effective_number
            weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weights", weights)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        weight = self.weights if self.weights.numel() else None
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


class MultiScaleAlignmentLoss(nn.Module):
    """Cosine alignment across FPC branches and encoder scales."""

    def __init__(self, branch_weight: float = 1.0, scale_weight: float = 1.0) -> None:
        super().__init__()
        self.branch_weight = branch_weight
        self.scale_weight = scale_weight

    def forward(self, branch_prototypes: Sequence[Tensor], scale_features: Tensor) -> Tensor:
        branch_losses: List[Tensor] = []
        for prototypes in branch_prototypes:
            similarity = prototypes @ prototypes.transpose(-1, -2)
            branch_losses.append(1.0 - _pairwise_off_diagonal_mean(similarity).mean())

        scale_similarity = scale_features @ scale_features.transpose(-1, -2)
        scale_loss = 1.0 - _pairwise_off_diagonal_mean(scale_similarity).mean()
        branch_loss = torch.stack(branch_losses).mean()
        return self.branch_weight * branch_loss + self.scale_weight * scale_loss


def alignment_coefficient(
    epoch: Optional[int],
    max_epochs: int = 300,
    maximum: float = 0.2,
    warmup_fraction: float = 0.2,
) -> float:
    """Cosine ramp: classification first, alignment strengthened later."""
    if epoch is None:
        return maximum
    warmup_epochs = max(1, int(max_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs))
    return maximum * 0.5 * (1.0 - math.cos(math.pi * progress))



@dataclass
class FIMambaConfig:
    num_classes: int = 4
    input_channels: int = 1
    dims: Tuple[int, int, int, int] = (48, 96, 192, 384)
    depths: Tuple[int, int, int, int] = (2, 2, 4, 2)
    fusion_dim: int = 256
    ssm_expansion: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    hard_token_mask: bool = False
    alignment_maximum: float = 0.2


class FIMamba(nn.Module):
    """End-to-end FI-Mamba classifier."""

    def __init__(
        self,
        config: FIMambaConfig,
        class_counts: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.fpc = FrequencyPriorCollaborativeModule(config.input_channels)
        self.encoder = IOFSMambaEncoder(
            dims=config.dims,
            depths=config.depths,
            ssm_expansion=config.ssm_expansion,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            hard_token_mask=config.hard_token_mask,
        )
        self.interaction = MultiScaleCLIPInteraction(
            stage_dims=config.dims,
            fusion_dim=config.fusion_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
        self.classification_loss = ClassBalancedCrossEntropy(class_counts=class_counts)
        self.alignment_loss = MultiScaleAlignmentLoss()

    def forward(
        self,
        images: Tensor,
        targets: Optional[Tensor] = None,
        epoch: Optional[int] = None,
        max_epochs: int = 300,
    ) -> Dict[str, Tensor | List[Tensor] | List[List[Dict[str, Tensor]]]]:
        fpc_output = self.fpc(images)
        features, encoder_diagnostics = self.encoder(fpc_output["branches"])
        interaction_output = self.interaction(features)
        logits = interaction_output["logits"]

        output: Dict[str, Tensor | List[Tensor] | List[List[Dict[str, Tensor]]]] = {
            **interaction_output,
            "probabilities": torch.softmax(logits, dim=-1),
            "fpc_low": fpc_output["low"],
            "fpc_high": fpc_output["high"],
            "fpc_prior": fpc_output["prior"],
            "fpc_reconstruction": fpc_output["reconstruction"],
            "encoder_diagnostics": encoder_diagnostics,
        }

        if targets is not None:
            classification = self.classification_loss(logits, targets)
            alignment = self.alignment_loss(
                interaction_output["branch_prototypes"],
                interaction_output["scale_features"],
            )
            coefficient = alignment_coefficient(
                epoch=epoch,
                max_epochs=max_epochs,
                maximum=self.config.alignment_maximum,
            )
            output.update(
                {
                    "classification_loss": classification,
                    "alignment_loss": alignment,
                    "alignment_coefficient": logits.new_tensor(coefficient),
                    "loss": classification + coefficient * alignment,
                }
            )
        return output


def build_fimamba(
    num_classes: int = 4,
    class_counts: Optional[Sequence[int]] = None,
    **config_overrides: object,
) -> FIMamba:
    """
    Build the paper-oriented default network.

    Example:
        model = build_fimamba(
            num_classes=4,
            class_counts=[120, 80, 130, 75],
            hard_token_mask=False,
        )
    """
    config = FIMambaConfig(num_classes=num_classes, **config_overrides)
    return FIMamba(config=config, class_counts=class_counts)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="FI-Mamba architecture shape demonstration")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--paper-configuration",
        action="store_true",
        help="Use the full (48,96,192,384)/(2,2,4,2) reference configuration.",
    )
    args = parser.parse_args()

    if args.paper_configuration:
        model = build_fimamba(num_classes=args.num_classes)
    else:
        # Smaller shape demonstration; the paper-oriented defaults remain in
        # FIMambaConfig and build_fimamba().
        model = build_fimamba(
            num_classes=args.num_classes,
            dims=(16, 32, 64, 128),
            depths=(1, 1, 1, 1),
            fusion_dim=64,
        )

    images = torch.randn(args.batch_size, 1, args.image_size, args.image_size)
    targets = torch.randint(0, args.num_classes, (args.batch_size,))
    model.eval()
    with torch.no_grad():
        output = model(images, targets=targets, epoch=120, max_epochs=300)

    reconstruction_error = (output["fpc_reconstruction"] - images).abs().max().item()
    print(f"logits shape: {tuple(output['logits'].shape)}")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    print(f"FPC max reconstruction error: {reconstruction_error:.3e}")
    print(f"total reference loss: {output['loss'].item():.6f}")


if __name__ == "__main__":
    main()
