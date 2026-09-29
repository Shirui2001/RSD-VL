from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FEATURE_LAYERS = (6, 12, 18, 24)
SCORING_MODES = ("rsd", "normalized_relative", "abnormal_only", "negative_normal")
FUSION_MODES = ("trimmed_mean", "mean", "max", "single")


@dataclass(frozen=True)
class PaperConfig:
    tau: float = 0.07
    gamma: float = 5.0
    scoring: str = "rsd"
    fusion: str = "trimmed_mean"
    fp_ratios: tuple[float, ...] = (0.001, 0.01, 0.05)
    pos_ratio: float = 0.1
    fp_weight: float = 0.25
    pos_weight: float = 0.10
    fp_dynamic_start: float = 0.22
    fp_dynamic_decay: float = 0.02
    fp_dynamic_min: float = 0.10
    fp_fixed_5pct: float = 0.08
    fp_fixed_0p1pct: float = 0.25
    pos_threshold: float = 0.55
    cls_weight: float = 0.25  
    use_erc: bool = True
    use_tao: bool = True


def check_config(cfg: PaperConfig) -> None:
    if cfg.tau <= 0 or cfg.gamma <= 0:
        raise ValueError("tau and gamma must be positive")
    if cfg.scoring not in SCORING_MODES or cfg.fusion not in FUSION_MODES:
        raise ValueError("Unsupported scoring or fusion")
    if any(q <= 0 or q > 1 for q in (*cfg.fp_ratios, cfg.pos_ratio)):
        raise ValueError("Tail ratios must lie in (0, 1]")
    if cfg.fp_dynamic_decay < 0 or not (0 < cfg.fp_dynamic_min <= cfg.fp_dynamic_start < 1):
        raise ValueError("Invalid FP dynamic threshold parameters")
    if any(not 0 < t < 1 for t in (cfg.fp_fixed_5pct, cfg.fp_fixed_0p1pct, cfg.pos_threshold)):
        raise ValueError("Fixed FP and positive tail thresholds must lie in (0,1)")
    if any(t < 0 for t in (cfg.fp_weight, cfg.pos_weight, cfg.cls_weight)):
        raise ValueError("Loss weights must be non-negative")


def aggregate_responses(
    features: torch.Tensor, normal: torch.Tensor, abnormal: torch.Tensor, tau: float = 0.07
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.shape[-1] != normal.shape[-1] or normal.shape[-1] != abnormal.shape[-1]:
        raise ValueError("Text and vision embedding dimensions differ")
    if normal.ndim != 2 or abnormal.ndim != 2 or not normal.shape[0] or not abnormal.shape[0]:
        raise ValueError("Both embedding banks must be nonempty [M,D]")
    f = F.normalize(features, dim=-1)
    n = F.normalize(normal, dim=-1)
    a = F.normalize(abnormal, dim=-1)
    s_n = f @ n.t()
    s_a = f @ a.t()
    r_n = tau * (torch.logsumexp(s_n / tau, dim=-1) - math.log(n.shape[0]))
    r_a = tau * (torch.logsumexp(s_a / tau, dim=-1) - math.log(a.shape[0]))
    return r_n, r_a


def semantic_score(r_n: torch.Tensor, r_a: torch.Tensor, cfg: PaperConfig) -> torch.Tensor:
    if cfg.scoring == "rsd":
        return cfg.gamma * (r_a - r_n)
    if cfg.scoring == "normalized_relative":
        return cfg.gamma * (r_a - r_n) / (r_a.abs() + r_n.abs() + 1e-6)
    if cfg.scoring == "abnormal_only":
        return cfg.gamma * r_a
    if cfg.scoring == "negative_normal":
        return -cfg.gamma * r_n
    raise ValueError(cfg.scoring)


def _feature_grid(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("Expected [batch, number_of_patches, channels]")
    b, p, d = features.shape
    side = math.isqrt(p)
    if side * side != p:
        raise ValueError(f"Patch count {p} is not square; verify visual preprocessing")
    return features.transpose(1, 2).reshape(b, d, side, side)


def scale_score(
    features: torch.Tensor, normal: torch.Tensor, abnormal: torch.Tensor,
    size: tuple[int, int], cfg: PaperConfig, *, feature_space: str = "pixel",
    pixel_chunk_rows: int = 32,
) -> torch.Tensor:
    feature_grid = _feature_grid(features)
    if feature_space == "pixel":
        if pixel_chunk_rows <= 0:
            raise ValueError("pixel_chunk_rows must be positive")
        b, _, _, _ = feature_grid.shape
        h, w = size
        x = 2.0 * (torch.arange(w, device=features.device, dtype=torch.float32) + 0.5) / w - 1.0
        rows = []

        def compute_chunk(grid_features, norm_bank, abn_bank, grid_xy):
            chunk_features = F.grid_sample(grid_features, grid_xy, mode="bilinear",
                                           padding_mode="border", align_corners=False)
            chunk_features = F.normalize(chunk_features.flatten(2).transpose(1,2), dim=-1)
            rn, ra = aggregate_responses(chunk_features, norm_bank, abn_bank, cfg.tau)
            return semantic_score(rn, ra, cfg).reshape(b, grid_xy.shape[1], w)

        for top in range(0, h, pixel_chunk_rows):
            bottom = min(h, top + pixel_chunk_rows)
            y = 2.0 * (torch.arange(top, bottom, device=features.device,
                                     dtype=torch.float32) + 0.5) / h - 1.0
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            coords = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b,-1,-1,-1)
            if torch.is_grad_enabled() and (features.requires_grad or normal.requires_grad or abnormal.requires_grad):
                row_scores = checkpoint(compute_chunk, feature_grid, normal, abnormal,
                                        coords, use_reentrant=False)
            else:
                row_scores = compute_chunk(feature_grid, normal, abnormal, coords)
            rows.append(row_scores)
        return torch.cat(rows, dim=1)
    if feature_space == "patch":
        rn, ra = aggregate_responses(features, normal, abnormal, cfg.tau)
        margin = semantic_score(rn, ra, cfg).reshape(features.shape[0], 1, *feature_grid.shape[-2:])
        return F.interpolate(margin, size=size, mode="bilinear", align_corners=False)[:, 0]
    raise ValueError("feature_space must be pixel (manuscript) or patch (diagnostic only)")


def fuse_scores(maps: Sequence[torch.Tensor], mode: str = "trimmed_mean") -> torch.Tensor:
    stack = torch.stack(tuple(maps), dim=1)
    if mode == "single":
        return stack[:, -1]
    if mode == "max":
        return stack.max(dim=1).values
    if mode == "mean":
        return stack.mean(dim=1)
    if mode == "trimmed_mean":
        if stack.shape[1] <= 2:
            raise ValueError("Trimmed mean requires at least three scales")
        return stack.sort(dim=1).values[:, 1:-1].mean(dim=1)
    raise ValueError(mode)


def seg_loss(raw_scores: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    y = labels.to(dtype=raw_scores.dtype)
    v = valid.bool()
    if not bool(v.any()):
        return raw_scores.sum() * 0.0
    log_prob = F.logsigmoid(raw_scores)
    log_one_minus = F.logsigmoid(-raw_scores)
    p = torch.sigmoid(raw_scores)
    pt = torch.where(y > 0.5, p, 1.0 - p)
    logpt = torch.where(y > 0.5, log_prob, log_one_minus)
    focal = (-0.5 * (1.0 - pt).pow(gamma) * logpt)[v].mean()
    vf = v.to(p.dtype)
    dims = (-2, -1)
    pred_pos, target_pos = p * vf, y * vf
    dice_pos = 1.0 - (2.0 * (pred_pos * target_pos).sum(dims) + 1.0) / (
        pred_pos.sum(dims) + target_pos.sum(dims) + 1.0)
    pred_neg, target_neg = (1.0 - p) * vf, (1.0 - y) * vf
    dice_neg = 1.0 - (2.0 * (pred_neg * target_neg).sum(dims) + 1.0) / (
        pred_neg.sum(dims) + target_neg.sum(dims) + 1.0)
    valid_imgs = v.flatten(1).any(dim=1)
    return focal + (dice_pos[valid_imgs] + dice_neg[valid_imgs]).mean()


def effective_region(
    y: torch.Tensor, ignore_mask: torch.Tensor | None, *, use_erc: bool
) -> torch.Tensor:
    if not use_erc:
        return torch.ones_like(y, dtype=torch.bool)
    if ignore_mask is None:
        raise ValueError("ERC requires Cityscapes training road mask: missing ignore_mask")
    return (ignore_mask < 0.5) | (y > 0.5)


def _tail_mean(values: torch.Tensor, fraction: float, *, largest: bool) -> torch.Tensor:
    k = max(1, int(math.ceil(values.numel() * fraction)))
    return values.topk(k, largest=largest).values.mean()


def tail_losses(
    raw: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
    img_labels: torch.Tensor, epoch: int, cfg: PaperConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = raw.sum() * 0.0
    if not cfg.use_tao:
        return zero, zero
    prob = torch.sigmoid(raw)
    normal_images = (img_labels == 0)[:, None, None]
    anomaly_images = (img_labels == 1)[:, None, None]
    negatives = prob[normal_images & valid & (y < 0.5)]
    positives = prob[anomaly_images & valid & (y > 0.5)]
    fp_loss = zero
    if negatives.numel():
        q001, q01, q05 = cfg.fp_ratios
        delta = max(cfg.fp_dynamic_min, cfg.fp_dynamic_start - cfg.fp_dynamic_decay * epoch)  
        mu01 = _tail_mean(negatives, q01, largest=True)
        mu05 = _tail_mean(negatives, q05, largest=True)
        mu001 = _tail_mean(negatives, q001, largest=True)
        fp_loss = (F.relu(mu01 - delta).square()
                   + 0.5 * F.relu(mu05 - cfg.fp_fixed_5pct).square()
                   + 0.5 * F.relu(mu001 - cfg.fp_fixed_0p1pct).square())
    pos_loss = zero
    if positives.numel():
        mu = _tail_mean(positives, cfg.pos_ratio, largest=False)
        pos_loss = F.relu(cfg.pos_threshold - mu)
    return fp_loss, pos_loss


def image_classification_loss(
    cls_features: torch.Tensor, normal: torch.Tensor, abnormal: torch.Tensor,
    labels: torch.Tensor, cfg: PaperConfig,
) -> torch.Tensor:
    norm_anchor = F.normalize(normal.mean(dim=0), dim=-1)
    abn_anchor = F.normalize(abnormal.mean(dim=0), dim=-1)
    f = F.normalize(cls_features, dim=-1)
    logits = cfg.gamma * torch.stack((f @ norm_anchor, f @ abn_anchor), dim=-1)
    return F.cross_entropy(logits, labels.long())
