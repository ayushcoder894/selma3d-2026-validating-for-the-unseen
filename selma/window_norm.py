"""Per-window intensity normalisation, matching how the SSL encoder was pretrained.

pretrain_ssl.py normalised every 96³ crop to its own 0.5-99.5 percentiles. A frozen SSL
encoder therefore has to see inputs normalised the same way: at training time each patch,
at inference time each sliding window. Stretching each window to its own range also makes
the network indifferent to regional brightness (dim gut-nerve fibres far from a bright
region were missed under per-volume normalisation).
"""
import torch
from torch import nn


def normalize_per_window(x, lo_q=0.005, hi_q=0.995, eps=1e-6):
    """(B, C, D, H, W) -> each sample stretched to its own percentile range, clipped."""
    flat = x.detach().float().flatten(1)
    lo = torch.quantile(flat, lo_q, dim=1)
    hi = torch.quantile(flat, hi_q, dim=1)
    shape = (-1,) + (1,) * (x.ndim - 1)
    span = (hi - lo).clamp_min(eps).view(shape)
    out = ((x.float() - lo.view(shape)) / span).clamp_(0, 1)
    # A flat window (e.g. all-zero padding) carries no contrast to stretch.
    flat_window = (hi - lo) < eps
    if flat_window.any():
        out[flat_window] = 0.0
    return out.to(x.dtype)


class PerWindowNorm(nn.Module):
    """Wrap a segmentation network so every window it sees is normalised first."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(normalize_per_window(x))


def encoder_modules(model):
    """DynUNet's encoder: input block, downsampling blocks and bottleneck."""
    return [model.input_block, *model.downsamples, model.bottleneck]
