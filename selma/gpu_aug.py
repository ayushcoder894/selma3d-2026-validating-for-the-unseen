"""Batch augmentations that run on the GPU, after the CPU intensity augmentations.

The preliminary test set is mostly structure types never seen labelled (microglia, organ
proteins, gut nerves, lymphatic vessels). Two gaps in the CPU pipeline matter for them:

* no spatial scaling, so the network memorised the object sizes of the six training
  families - on microglia it paints blobs several times the size of the cell bodies;
* no illumination model, so dim structures far from a bright region are dropped - on gut
  nerves only the fibres near the brightest band are segmented.

Random zoom/rotation and smooth multiplicative bias fields address those. They run here
because a trilinear resample of a 96³ patch on the CPU would starve the GPU.
"""
import math

import torch
import torch.nn.functional as F


def _rotation(angles):
    """(B, 3) radians -> (B, 3, 3) rotation matrices, R = Rz @ Ry @ Rx."""
    cx, cy, cz = torch.cos(angles).unbind(1)
    sx, sy, sz = torch.sin(angles).unbind(1)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).view(-1, 3, 3)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).view(-1, 3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).view(-1, 3, 3)
    return rz @ ry @ rx


def random_affine(img, lab, p_scale=0.4, scale=(0.75, 1.35), p_rot=0.3, max_deg=20.0,
                  generator=None):
    """Isotropic zoom plus small rotation, identical for image and label.

    A zoom factor s makes structures appear s times larger. affine_grid maps output
    coordinates to input coordinates, so the sampling matrix is R / s.
    """
    b, dev = img.shape[0], img.device
    do_scale = torch.rand(b, device=dev, generator=generator) < p_scale
    do_rot = torch.rand(b, device=dev, generator=generator) < p_rot
    if not (do_scale.any() or do_rot.any()):
        return img, lab
    # Log-uniform zoom so shrinking and enlarging are equally likely.
    lo, hi = math.log(scale[0]), math.log(scale[1])
    s = torch.exp(torch.empty(b, device=dev).uniform_(lo, hi, generator=generator))
    s = torch.where(do_scale, s, torch.ones_like(s))
    ang = torch.empty(b, 3, device=dev).uniform_(-1.0, 1.0, generator=generator)
    ang = ang * math.radians(max_deg) * do_rot[:, None].float()
    theta = torch.zeros(b, 3, 4, device=dev)
    theta[:, :, :3] = _rotation(ang) / s[:, None, None]
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection",
                        align_corners=False)
    # Resample the label as a soft map and re-binarise: nearest-neighbour would alias
    # the boundaries of small objects (plaques are a few voxels across).
    lab = F.grid_sample(lab, grid, mode="bilinear", padding_mode="reflection",
                        align_corners=False)
    return img, (lab > 0.5).to(img.dtype)


def random_bias_field(img, p=0.3, log_std=(0.15, 0.45), control=3, generator=None):
    """Smooth multiplicative illumination field, exp of an upsampled coarse Gaussian."""
    b, dev = img.shape[0], img.device
    do = torch.rand(b, device=dev, generator=generator) < p
    if not do.any():
        return img
    std = torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(*log_std, generator=generator)
    coarse = torch.randn(b, 1, control, control, control, device=dev,
                         generator=generator) * std
    field = torch.exp(F.interpolate(coarse, size=img.shape[2:], mode="trilinear",
                                    align_corners=True))
    return torch.where(do[:, None, None, None, None], img * field, img)


def random_lowres(img, p=0.15, zoom=(0.5, 0.85), generator=None):
    """Simulate a coarser acquisition: downsample, then upsample back."""
    b, dev = img.shape[0], img.device
    do = torch.rand(b, device=dev, generator=generator) < p
    if not do.any():
        return img
    out = img.clone()
    for i in torch.nonzero(do).flatten().tolist():
        z = float(torch.empty(1).uniform_(*zoom))
        small = [max(8, int(round(d * z))) for d in img.shape[2:]]
        low = F.interpolate(img[i:i + 1], size=small, mode="area")
        out[i:i + 1] = F.interpolate(low, size=img.shape[2:], mode="trilinear",
                                     align_corners=False)
    return out


def augment_batch(img, lab, generator=None, clamp=(-0.5, 2.0)):
    img, lab = random_affine(img, lab, generator=generator)
    img = random_bias_field(img, generator=generator)
    img = random_lowres(img, generator=generator)
    return img.clamp_(*clamp), lab
