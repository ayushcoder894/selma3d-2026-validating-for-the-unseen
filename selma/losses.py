"""Losses aligned with the challenge metrics.

Isolated structures are scored with Dice + Panoptic Quality; contiguous structures with
Dice + centreline Dice. Dice+BCE covers the overlap term everywhere. soft-clDice is added
only on contiguous samples, where a single broken voxel costs far more centreline overlap
than it costs volumetric Dice.
"""
import torch
import torch.nn.functional as F
from torch import nn


def soft_erode(x):
    a = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    b = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    c = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(a, b), c)


def soft_dilate(x):
    return F.max_pool3d(x, 3, 1, 1)


def soft_skeleton(x, iters=5):
    """Differentiable skeleton via iterated morphological opening (Shit et al., clDice)."""
    opened = soft_dilate(soft_erode(x))
    skel = F.relu(x - opened)
    for _ in range(iters):
        x = soft_erode(x)
        opened = soft_dilate(soft_erode(x))
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(prob, target, iters=5, eps=1e-5):
    sk_p, sk_t = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    tprec = (torch.sum(sk_p * target) + eps) / (torch.sum(sk_p) + eps)
    tsens = (torch.sum(sk_t * prob) + eps) / (torch.sum(sk_t) + eps)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


def dice_loss(prob, target, eps=1e-5):
    dims = tuple(range(2, prob.ndim))
    num = 2 * (prob * target).sum(dims) + eps
    den = prob.sum(dims) + target.sum(dims) + eps
    return (1 - num / den).mean()


class SelmaLoss(nn.Module):
    """Dice + BCE everywhere, plus soft-clDice on tubular samples.

    BCE is weighted towards foreground because plaque patches can be 99.99% background,
    where an unweighted BCE is minimised by predicting empty.
    """

    def __init__(self, w_dice=1.0, w_bce=1.0, w_cldice=0.5, pos_weight=5.0, cldice_iters=5):
        super().__init__()
        self.w_dice, self.w_bce, self.w_cldice = w_dice, w_bce, w_cldice
        self.cldice_iters = cldice_iters
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits, target, tubular=None):
        prob = torch.sigmoid(logits)
        loss = self.w_dice * dice_loss(prob, target)
        loss = loss + self.w_bce * F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight)
        if self.w_cldice > 0 and tubular is not None:
            m = tubular.view(-1) > 0.5
            if m.any():
                loss = loss + self.w_cldice * soft_cldice(
                    prob[m].float(), target[m].float(), self.cldice_iters)
        return loss


def deep_supervision_loss(criterion, outputs, target, tubular, weights=None):
    """DynUNet emits (B, heads, C, ...) when deep supervision is on."""
    if outputs.ndim == target.ndim:
        return criterion(outputs, target, tubular)
    heads = outputs.unbind(1)
    if weights is None:
        weights = [1 / 2 ** i for i in range(len(heads))]
    total = sum(weights)
    loss = 0.0
    for w, head in zip(weights, heads):
        t = target
        if head.shape[2:] != target.shape[2:]:
            t = F.interpolate(target, head.shape[2:], mode="nearest")
        loss = loss + (w / total) * criterion(head, t, tubular)
    return loss
