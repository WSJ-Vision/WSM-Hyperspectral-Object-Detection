"""Spectral mappers used by the WSM representation family.

The WSM mapper implements factorized shared spectral bases, a shared
band-distribution factor, bounded sample-conditioned modulation, and residual
spatial refinement. It returns both the three-channel representation and the
auxiliary regularization terms needed by a training wrapper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Fixed3Mapper(nn.Module):
    """Select three predefined zero-based spectral channels."""

    def __init__(self, indices: Sequence[int]) -> None:
        super().__init__()
        if len(indices) != 3:
            raise ValueError("Fixed3Mapper requires exactly three indices")
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        return x.float().index_select(1, self.indices)


class Conv1x1Mapper(nn.Module):
    """Learnable unconstrained 1x1 full-band projection."""

    def __init__(self, in_channels: int, fixed3_indices: Sequence[int] | None = None) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, 3, kernel_size=1)
        if fixed3_indices is not None:
            if len(fixed3_indices) != 3:
                raise ValueError("fixed3_indices must contain three indices")
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            with torch.no_grad():
                for output_channel, source_channel in enumerate(fixed3_indices):
                    self.projection.weight[output_channel, source_channel, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x.float())


class NNPMapper(nn.Module):
    """Direct nonnegative row-normalized full-band projection."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, in_channels))

    def spectral_weights(self) -> Tensor:
        return torch.softmax(self.logits, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("oc,bchw->bohw", self.spectral_weights(), x.float())


class WSMMapper(nn.Module):
    """Weighted structured spectral adaptation mapper.

    The orthogonality penalty acts on the shared-factor-weighted bases. It is a
    soft structural bias and does not impose a hard rank constraint on the
    composed sample-dependent effective mapping.
    """

    def __init__(
        self,
        in_channels: int,
        lambda_orth: float = 0.01,
        lambda_band_entropy: float = 0.001,
        tau: float = 1.0,
        eps: float = 1e-8,
        gate_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError("WSMMapper requires at least three input channels")
        hidden = gate_hidden_channels or max(8, min(64, in_channels // 3))
        self.in_channels = int(in_channels)
        self.S = nn.Parameter(torch.zeros(3, in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * in_channels),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self.refine = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.lambda_orth = float(lambda_orth)
        self.lambda_band_entropy = float(lambda_band_entropy)
        self.band_entropy_target = math.log(float(in_channels))
        self.tau = float(tau)
        self.eps = float(eps)

    def shared_weights(self) -> tuple[Tensor, Tensor]:
        """Return the normalized shared bases and the shared distribution."""
        q = torch.softmax(self.S / self.tau, dim=1)
        pi = torch.softmax(self.beta, dim=0)
        base = q * pi.unsqueeze(0)
        return base / (base.sum(dim=1, keepdim=True) + self.eps), pi

    @staticmethod
    def _orthogonality_loss(weights: Tensor) -> Tensor:
        gram = weights @ weights.transpose(0, 1)
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).square().mean()

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], received {tuple(x.shape)}"
            )
        x = x.float()
        base, pi = self.shared_weights()
        pooled = x.mean(dim=(2, 3))
        gate = 1.0 + 0.5 * torch.tanh(
            self.gate_mlp(pooled).reshape(x.shape[0], 3, self.in_channels)
        )
        effective = base.unsqueeze(0) * gate
        effective = effective / (effective.sum(dim=2, keepdim=True) + self.eps)
        mapped = torch.einsum("boc,bchw->bohw", effective, x)
        mapped = mapped + self.alpha * self.refine(mapped)
        entropy = -(pi * (pi + self.eps).log()).sum()
        auxiliary = {
            "orth_loss": self._orthogonality_loss(base),
            "band_entropy_loss": (entropy - self.band_entropy_target).square(),
        }
        return mapped, auxiliary
