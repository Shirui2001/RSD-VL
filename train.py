"""
Clean two-stage RSD-VL training script (road-focused).

Stage 1: text_adapter  — multi-scale seg loss + orthogonal text regularization
Stage 2: image_adapter — seg + cls + FP tail + POS tail

Example (recommended road setup):
    python train_clean.py \\
        --dataset RoadSynth \\
        --save_path ckpt/road \\
        --use_max_sim --logit_scale 10.0 \\
        --text_epoch 5 --image_epoch 10 --image_batch_size 4

See README / train.py for full experimental options removed here.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import warnings
from glob import glob
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset import get_dataset
from forward_utils import (
    calculate_seg_loss,
    calculate_similarity_map,
    get_adapted_multi_text_embeddings,
    get_adapted_single_class_text_embedding,
    get_adapted_text_embedding,
)
from model.adapter import AdaptedCLIP
from model.clip import create_model
from utils import setup_seed

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (fixed in original train.py; expose here for clarity)
# ---------------------------------------------------------------------------
FEATURE_LAYERS = [6, 12, 18, 24]
W_CLS = 0.25
W_FP = 0.25
W_POS = 0.10

ROAD_DATASETS = {
    "Road",
    "RoadSynth",
    "RoadAnomaly",
    "RoadAnomaly21",
    "RoadObsticle21",
    "FS_LostFound_full",
    "fs_static",
}


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def pos_tail_lift_loss(
    pa_pos: torch.Tensor, t: float = 0.55, q: float = 0.10
) -> torch.Tensor:
    if pa_pos.numel() == 0:
        return pa_pos.sum() * 0.0
    k = max(1, int(pa_pos.numel() * q))
    bottomk = torch.topk(pa_pos, k, largest=False).values.mean()
    return torch.relu(t - bottomk)


def fp_tail_loss(pa_valid: torch.Tensor, t: float = 0.15) -> torch.Tensor:
    if pa_valid.numel() == 0:
        return pa_valid.sum() * 0.0

    n = pa_valid.numel()
    top1 = torch.topk(pa_valid, max(1, int(n * 0.01))).values.mean()
    top5 = torch.topk(pa_valid, max(1, int(n * 0.05))).values.mean()
    top01 = torch.topk(pa_valid, max(1, int(n * 0.001))).values.mean()

    def hinge2(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2

    return (
        hinge2(top1 - t)
        + 0.5 * hinge2(top5 - 0.08)
        + 0.5 * hinge2(top01 - 0.25)
    )


def assert_model_finite(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param is not None and not torch.isfinite(param).all():
            raise RuntimeError(f"NaN/Inf in parameter: {name}")


# ---------------------------------------------------------------------------
# DataLoader helper for RoadSynth (idx % 2 == abnormal)
# ---------------------------------------------------------------------------
class EvenOddBatchSampler(Sampler):
    """Each batch: half even indices (normal), half odd indices (abnormal)."""

    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = True):
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for 1:1 normal/abnormal batches")
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.evens = [i for i in range(dataset_len) if i % 2 == 0]
        self.odds = [i for i in range(dataset_len) if i % 2 == 1]

    def __iter__(self):
        ev, od = self.evens[:], self.odds[:]
        random.shuffle(ev)
        random.shuffle(od)
        num_batches = min(len(ev), len(od)) // self.half
        for b in range(num_batches):
            batch = ev[b * self.half : (b + 1) * self.half] + od[
                b * self.half : (b + 1) * self.half
            ]
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return min(len(self.evens), len(self.odds)) // self.half


def build_image_dataloader(
    dataset,
    batch_size: int,
    dataset_name: str,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if dataset_name in ROAD_DATASETS:
        sampler = EvenOddBatchSampler(len(dataset), batch_size, drop_last=True)
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


# ---------------------------------------------------------------------------
# Frozen CLIP-Surgery features (Stage 1 only)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_frozen_patch_features(
    clip_surgery: nn.Module,
    clip_model: nn.Module,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    _, patch_features = clip_surgery.encode_image(images, FEATURE_LAYERS)
    cls_token, _ = clip_model.encode_image(images, [])
    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

    outputs = []
    for tokens in patch_features:
        t = clip_surgery.visual.ln_post(tokens[:, 1:, :])
        t = t @ clip_surgery.visual.proj
        t = t / t.norm(dim=-1, keepdim=True)
        outputs.append(t + cls_token.unsqueeze(1))
    return outputs


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def train_text_adapter(
    model: AdaptedCLIP,
    clip_surgery: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    dataset_name: str,
    img_size: int,
    text_epoch: int,
    start_epoch: int,
    save_path: str,
    text_norm_weight: float,
    use_max_sim: bool,
    logit_scale: Optional[float],
    logger: logging.Logger,
) -> AdaptedCLIP:
    model.train()
    for epoch in range(start_epoch, text_epoch):
        losses = []
        for batch in tqdm(loader, desc=f"text {epoch + 1}/{text_epoch}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            class_names = batch["class_name"]

            text_by_class = {
                c: get_adapted_single_class_text_embedding(model, dataset_name, c, device)
                for c in set(class_names)
            }
            text_feat = torch.stack([text_by_class[c] for c in class_names], dim=0)

            patch_features = encode_frozen_patch_features(
                clip_surgery, model.clipmodel, images
            )

            e_norm, e_abn = None, None
            if use_max_sim:
                e_norm, e_abn = get_adapted_multi_text_embeddings(
                    model, dataset_name, device
                )

            seg_loss = 0.0
            for feat in patch_features:
                preds = calculate_similarity_map(
                    feat,
                    text_feat,
                    img_size,
                    use_max_sim=use_max_sim,
                    logit_scale=logit_scale,
                    E_norm=e_norm,
                    E_abn=e_abn,
                )
                seg_loss += calculate_seg_loss(preds, masks)
            seg_loss /= len(patch_features)

            ortho = (
                (text_feat[:, :, 0] * text_feat[:, :, 1]).sum(1).mean() ** 2
            )
            loss = seg_loss + text_norm_weight * ortho

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        logger.info("text epoch %d loss=%.4f", epoch + 1, float(np.mean(losses)))
        assert_model_finite(model)
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": model.text_adapter.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            os.path.join(save_path, "text_adapter.pth"),
        )
    return model


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
def _valid_mask(
    ignore_mask: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    if ignore_mask is None:
        return torch.ones_like(mask.squeeze(1), dtype=torch.bool)
    tgt = mask.squeeze(1)
    road_valid = ignore_mask.squeeze(1) < 0.5
    return road_valid | (tgt > 0.5)


def train_image_adapter(
    model: AdaptedCLIP,
    text_embeddings: Dict[str, torch.Tensor],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    *,
    device: torch.device,
    img_size: int,
    image_epoch: int,
    start_epoch: int,
    save_path: str,
    use_max_sim: bool,
    logit_scale: Optional[float],
    e_norm: Optional[torch.Tensor],
    e_abn: Optional[torch.Tensor],
    logger: logging.Logger,
) -> AdaptedCLIP:
    model.train()
    for epoch in range(start_epoch, image_epoch):
        losses = []
        for it, batch in enumerate(
            tqdm(loader, desc=f"image {epoch + 1}/{image_epoch}")
        ):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            ignore_mask = batch.get("ignore_mask")
            if ignore_mask is not None:
                ignore_mask = ignore_mask.to(device)

            class_names = batch["class_name"]
            text_feat = torch.stack(
                [text_embeddings[c] for c in class_names], dim=0
            )

            patch_features, det_feature = model(images)
            cls_preds = torch.matmul(det_feature.unsqueeze(1), text_feat)[:, 0]
            cls_loss = F.cross_entropy(cls_preds, labels)

            seg_loss = torch.zeros([], device=device)
            fp_loss = torch.zeros([], device=device)
            pos_loss = torch.zeros([], device=device)

            t_fp = max(0.10, 0.22 - 0.02 * epoch)

            for feat in patch_features:
                preds = calculate_similarity_map(
                    feat,
                    text_feat,
                    img_size,
                    use_max_sim=use_max_sim,
                    logit_scale=logit_scale,
                    E_norm=e_norm,
                    E_abn=e_abn,
                )
                seg_loss = seg_loss + calculate_seg_loss(preds, masks, ignore_mask)

                prob_anom = torch.softmax(preds, dim=1)[:, 1]
                valid = _valid_mask(ignore_mask, masks)

                normal = labels == 0
                if normal.any():
                    pa = prob_anom[normal]
                    normal_pixels = masks[normal].squeeze(1) < 0.5
                    valid_fp = valid[normal] & normal_pixels
                    pa_valid = pa[valid_fp]
                    if pa_valid.numel() > 0:
                        fp_loss = fp_loss + fp_tail_loss(pa_valid, t=t_fp)

                abnormal = labels == 1
                if abnormal.any():
                    pa = prob_anom[abnormal]
                    abn_mask = masks[abnormal].squeeze(1) > 0.5
                    valid_abn = valid[abnormal]
                    pa_pos = pa[abn_mask & valid_abn]
                    pos_loss = pos_loss + pos_tail_lift_loss(pa_pos)

            n_scales = len(patch_features)
            seg_loss = seg_loss / n_scales
            fp_loss = fp_loss / n_scales
            pos_loss = pos_loss / n_scales

            loss = seg_loss + W_CLS * cls_loss + W_FP * fp_loss + W_POS * pos_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch + 1}, iter={it}"
                )

            if it % 50 == 0:
                logger.info(
                    "epoch=%d iter=%d total=%.4f seg=%.4f cls=%.4f fp=%.4f pos=%.4f",
                    epoch + 1,
                    it,
                    loss.item(),
                    seg_loss.item(),
                    cls_loss.item(),
                    fp_loss.item(),
                    pos_loss.item(),
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        logger.info(
            "image epoch %d avg_loss=%.4f lr=%.2e",
            epoch + 1,
            float(np.mean(losses)),
            optimizer.param_groups[0]["lr"],
        )

        assert_model_finite(model)
        state = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }
        torch.save(state, os.path.join(save_path, "image_adapter.pth"))
        torch.save(state, os.path.join(save_path, f"image_adapter_{epoch + 1}.pth"))

    return model


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def setup_logger(save_path: str) -> logging.Logger:
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger("train_clean")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(save_path, "train.log"), mode="a")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def load_max_sim_embeddings(
    model: nn.Module, dataset_name: str, device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if dataset_name not in ROAD_DATASETS:
        return None, None
    e_norm, e_abn = get_adapted_multi_text_embeddings(model, dataset_name, device)
    if e_norm is None or e_abn is None:
        return None, None
    return e_norm, e_abn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean AA-CLIP two-stage training")
    # model
    p.add_argument("--model_name", default="ViT-L-14-336")
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--surgery_until_layer", type=int, default=20)
    p.add_argument("--relu", action="store_true")
    # data / training
    p.add_argument("--dataset", default="RoadSynth")
    p.add_argument(
        "--training_mode",
        default="full_shot",
        choices=["few_shot", "full_shot"],
    )
    p.add_argument("--shot", type=int, default=-1, help="few-shot count; ignored in full_shot")
    p.add_argument("--text_batch_size", type=int, default=16)
    p.add_argument("--image_batch_size", type=int, default=4)
    p.add_argument("--text_epoch", type=int, default=5)
    p.add_argument("--image_epoch", type=int, default=20)
    p.add_argument("--text_lr", type=float, default=1e-5)
    p.add_argument("--image_lr", type=float, default=5e-4)
    # adapters
    p.add_argument("--text_norm_weight", type=float, default=0.1)
    p.add_argument("--text_adapt_weight", type=float, default=0.1)
    p.add_argument("--image_adapt_weight", type=float, default=0.1)
    p.add_argument("--text_adapt_until", type=int, default=3)
    p.add_argument("--image_adapt_until", type=int, default=6)
    # road extensions
    p.add_argument("--use_max_sim", action="store_true")
    p.add_argument("--logit_scale", type=float, default=None)
    p.add_argument("--hard_fp_root", type=str, default=None)
    p.add_argument("--hard_fp_prob", type=float, default=0.05)
    p.add_argument("--hard_fp_topk", type=int, default=20)
    # misc
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--save_path", default="ckpt/clean")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    logger = setup_logger(args.save_path)
    logger.info("args: %s", vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    pin_memory = use_cuda

    clip_surgery = create_model(
        args.model_name,
        args.img_size,
        device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        args.model_name,
        args.img_size,
        device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    for p in model.clipmodel.parameters():
        p.requires_grad = False

    text_optimizer = torch.optim.Adam(
        model.text_adapter.parameters(), lr=args.text_lr, betas=(0.5, 0.999)
    )
    image_optimizer = torch.optim.Adam(
        model.image_adapter.parameters(), lr=args.image_lr, betas=(0.5, 0.999)
    )
    image_scheduler = StepLR(image_optimizer, step_size=3, gamma=0.5)

    # resume
    text_start, adapt_text = 0, args.text_epoch > 0
    text_ckpt = glob(os.path.join(args.save_path, "text_adapter.pth"))
    if text_ckpt:
        ckpt = torch.load(text_ckpt[0], map_location=device)
        model.text_adapter.load_state_dict(ckpt["text_adapter"])
        text_optimizer.load_state_dict(ckpt["text_optimizer"])
        text_start = ckpt["epoch"]
        adapt_text = text_start < args.text_epoch

    image_start = 0
    image_ckpt = glob(os.path.join(args.save_path, "image_adapter.pth"))
    if image_ckpt:
        ckpt = torch.load(image_ckpt[0], map_location=device)
        model.image_adapter.load_state_dict(ckpt["image_adapter"])
        image_optimizer.load_state_dict(ckpt["image_optimizer"])
        image_start = ckpt["epoch"]

    shot = -1 if args.training_mode == "full_shot" else args.shot
    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        shot,
        "train",
        logger,
        hard_fp_root=args.hard_fp_root,
        hard_fp_prob=args.hard_fp_prob,
        hard_fp_topk=args.hard_fp_topk,
    )

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    text_loader = DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **loader_kwargs
    )
    image_loader = build_image_dataloader(
        image_dataset,
        args.image_batch_size,
        args.dataset,
        args.num_workers,
        pin_memory,
    )

    use_max_sim = args.use_max_sim
    if use_max_sim and args.dataset not in ROAD_DATASETS:
        logger.warning("use_max_sim disabled: dataset is not a road dataset")
        use_max_sim = False

    e_norm, e_abn = None, None
    if use_max_sim:
        e_norm, e_abn = load_max_sim_embeddings(model, args.dataset, device)
        if e_norm is None:
            logger.warning("use_max_sim disabled: failed to load multi-text embeddings")
            use_max_sim = False
        else:
            logger.info(
                "max_sim: %d normal / %d abnormal templates",
                e_norm.shape[0],
                e_abn.shape[0],
            )

    if adapt_text:
        model = train_text_adapter(
            model,
            clip_surgery,
            text_loader,
            text_optimizer,
            device=device,
            dataset_name=args.dataset,
            img_size=args.img_size,
            text_epoch=args.text_epoch,
            start_epoch=text_start,
            save_path=args.save_path,
            text_norm_weight=args.text_norm_weight,
            use_max_sim=use_max_sim,
            logit_scale=args.logit_scale,
            logger=logger,
        )

    del text_loader, text_dataset, clip_surgery, text_optimizer
    if use_cuda:
        torch.cuda.empty_cache()

    with torch.no_grad():
        embed_model = model if args.text_epoch > 0 else clip_model
        text_embeddings = get_adapted_text_embedding(embed_model, args.dataset, device)

    if use_max_sim:
        e_norm, e_abn = load_max_sim_embeddings(model, args.dataset, device)
        if e_norm is not None:
            e_norm = e_norm.detach().requires_grad_(False)
            e_abn = e_abn.detach().requires_grad_(False)

    train_image_adapter(
        model,
        text_embeddings,
        image_loader,
        image_optimizer,
        image_scheduler,
        device=device,
        img_size=args.img_size,
        image_epoch=args.image_epoch,
        start_epoch=image_start,
        save_path=args.save_path,
        use_max_sim=use_max_sim,
        logit_scale=args.logit_scale,
        e_norm=e_norm,
        e_abn=e_abn,
        logger=logger,
    )
    logger.info("Training finished. Checkpoints in %s", args.save_path)


if __name__ == "__main__":
    main()
