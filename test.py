"""
Clean RSD-VL evaluation script (road-focused).

Example (recommended road setup, aligned with train_clean.py):
    python test_clean.py \\
        --save_path ckpt/road \\
        --dataset RoadAnomaly21 \\
        --use_max_sim --logit_scale 10.0 \\
        --score_mode raw --fusion_mode trimmed_mean

See test.py for debug logging, sentinel traces, and experimental options.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset import DOMAINS, get_dataset
from forward_utils import (
    get_adapted_multi_text_embeddings,
    get_adapted_text_embedding,
    metrics_eval,
    visualize,
)
from model.adapter import AdaptedCLIP
from model.clip import create_model
from utils import setup_seed

warnings.filterwarnings("ignore")

ROAD_DATASETS = {
    "Road",
    "RoadSynth",
    "RoadAnomaly",
    "RoadAnomaly21",
    "RoadObsticle21",
    "FS_LostFound_full",
    "fs_static",
}

MAXSIM_TAU = 0.07
IMG_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _reshape_logits_to_map(logits: torch.Tensor, img_size: int) -> torch.Tensor:
    """logits: (B, 2, H, W) -> upsampled anomaly channel or raw diff."""
    logits = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=True)
    return logits


def score_scale_maxsim(
    features: torch.Tensor,
    e_norm: torch.Tensor,
    e_abn: torch.Tensor,
    img_size: int,
    *,
    logit_scale: Optional[float],
    score_mode: str,
) -> torch.Tensor:
    """Return per-scale anomaly map (B, H, W)."""
    f = F.normalize(features, dim=-1)
    e_norm = F.normalize(e_norm, dim=-1)
    e_abn = F.normalize(e_abn, dim=-1)

    sim_norm_all = torch.matmul(f, e_norm.t())
    sim_abn_all = torch.matmul(f, e_abn.t())
    sim_norm = (sim_norm_all / MAXSIM_TAU).logsumexp(dim=-1) * MAXSIM_TAU
    sim_abn = (sim_abn_all / MAXSIM_TAU).logsumexp(dim=-1) * MAXSIM_TAU
    score_diff = sim_abn - sim_norm
    if logit_scale is not None:
        score_diff = score_diff * logit_scale

    b, l = score_diff.shape
    h = int(np.sqrt(l))
    score_map = score_diff.view(b, h, h)

    if score_mode == "raw":
        return F.interpolate(
            score_map.unsqueeze(1), size=img_size, mode="bilinear", align_corners=True
        ).squeeze(1)

    logits = torch.stack([-score_map, score_map], dim=1)
    logits = _reshape_logits_to_map(logits, img_size)
    return torch.softmax(logits, dim=1)[:, 1]


def score_scale_single(
    features: torch.Tensor,
    text_emb: torch.Tensor,
    img_size: int,
    *,
    logit_scale: Optional[float],
    score_mode: str,
) -> torch.Tensor:
    """Return per-scale anomaly map (B, H, W) using one normal/abnormal text pair."""
    s = torch.matmul(features, text_emb)  # (B, L, 2)
    if logit_scale is not None:
        s = s * logit_scale

    b, l, _ = s.shape
    h = int(np.sqrt(l))
    logits = s.permute(0, 2, 1).view(b, 2, h, h)
    logits = _reshape_logits_to_map(logits, img_size)

    if score_mode == "raw":
        return logits[:, 1] - logits[:, 0]
    return torch.softmax(logits, dim=1)[:, 1]


def fuse_multiscale(
    maps: List[torch.Tensor],
    fusion_mode: str,
    score_mode: str,
    disable_zscore: bool,
) -> torch.Tensor:
    """maps: list of (B, H, W) -> fused (B, H, W)."""
    stacked = torch.stack(maps, dim=1)  # (B, S, H, W)
    if score_mode != "raw" and not disable_zscore:
        mu = stacked.mean(dim=(2, 3), keepdim=True)
        sd = stacked.std(dim=(2, 3), keepdim=True) + 1e-6
        stacked = (stacked - mu) / sd

    if fusion_mode == "median":
        sorted_maps, _ = torch.sort(stacked, dim=1)
        return sorted_maps[:, stacked.shape[1] // 2]
    if fusion_mode == "trimmed_mean":
        if stacked.shape[1] > 2:
            sorted_maps, _ = torch.sort(stacked, dim=1)
            return sorted_maps[:, 1:-1].mean(dim=1)
        return stacked.mean(dim=1)
    if fusion_mode == "max":
        return stacked.max(dim=1).values
    return stacked.mean(dim=1)


def apply_eval_mask(
    scores: torch.Tensor,
    road_mask: Optional[torch.Tensor],
    ignore_mask: Optional[torch.Tensor],
    eval_mode: str,
) -> torch.Tensor:
    if eval_mode != "road_only":
        return scores

    mask = road_mask if road_mask is not None else ignore_mask
    if mask is None:
        return scores

    mask_2d = mask.squeeze(1) if mask.dim() == 4 else mask
    if mask_2d.shape != scores.shape:
        mask_2d = F.interpolate(
            mask_2d.unsqueeze(1),
            size=scores.shape[1:],
            mode="nearest",
        ).squeeze(1)
    return scores * (1.0 - mask_2d)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    text_emb: torch.Tensor,
    device: torch.device,
    img_size: int,
    *,
    use_max_sim: bool,
    e_norm: Optional[torch.Tensor],
    e_abn: Optional[torch.Tensor],
    logit_scale: Optional[float],
    score_mode: str,
    fusion_mode: str,
    disable_zscore: bool,
    eval_mode: str,
) -> Tuple[np.ndarray, ...]:
    masks, labels, preds, preds_image, file_names = [], [], [], [], []
    road_masks, ignore_masks = [], []

    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device)
        masks.append(batch["mask"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        file_names.extend(batch["file_name"])

        road_mask = batch.get("road_mask")
        ignore_mask = batch.get("ignore_mask")
        road_mask_dev = road_mask.to(device) if road_mask is not None else None
        ignore_mask_dev = ignore_mask.to(device) if ignore_mask is not None else None

        road_masks.append(
            road_mask.cpu().numpy() if road_mask is not None else np.zeros_like(masks[-1])
        )
        if ignore_mask is not None:
            ignore_masks.append(ignore_mask.cpu().numpy())

        patch_features, det_feature = model(images)
        s_img = det_feature @ text_emb
        pred_img = torch.sigmoid((s_img[:, 1] - s_img[:, 0]) / IMG_TEMPERATURE)
        preds_image.append(pred_img.cpu().numpy())

        scale_maps = []
        for feat in patch_features:
            if use_max_sim and e_norm is not None and e_abn is not None:
                anom = score_scale_maxsim(
                    feat,
                    e_norm,
                    e_abn,
                    img_size,
                    logit_scale=logit_scale,
                    score_mode=score_mode,
                )
            else:
                anom = score_scale_single(
                    feat,
                    text_emb,
                    img_size,
                    logit_scale=logit_scale,
                    score_mode=score_mode,
                )
            scale_maps.append(anom)

        fused = fuse_multiscale(scale_maps, fusion_mode, score_mode, disable_zscore)
        fused = apply_eval_mask(fused, road_mask_dev, ignore_mask_dev, eval_mode)
        preds.append(fused.cpu().numpy())

    ignore_out = np.concatenate(ignore_masks, axis=0) if ignore_masks else None
    return (
        np.concatenate(masks, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(preds_image, axis=0),
        file_names,
        np.concatenate(road_masks, axis=0),
        ignore_out,
    )


# ---------------------------------------------------------------------------
# Checkpoint / setup
# ---------------------------------------------------------------------------
def load_image_adapter(model: AdaptedCLIP, save_path: str, logger: logging.Logger) -> None:
    files = glob(os.path.join(save_path, "image_adapter_*.pth"))
    if not files:
        alt = glob(os.path.join(save_path, "image_adapter.pth"))
        files = alt if alt else []

    if not files:
        logger.warning("No image adapter checkpoint found.")
        return

    def epoch_num(path: str) -> int:
        m = re.search(r"image_adapter_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    latest = max(files, key=epoch_num)
    ckpt = torch.load(latest, map_location="cpu")
    model.image_adapter.load_state_dict(ckpt["image_adapter"])
    logger.info("Loaded image adapter: %s (epoch %s)", latest, ckpt.get("epoch", "?"))


def load_text_adapter(model: AdaptedCLIP, save_path: str) -> bool:
    files = glob(os.path.join(save_path, "text_adapter.pth"))
    if not files:
        return False
    ckpt = torch.load(files[0], map_location="cpu")
    model.text_adapter.load_state_dict(ckpt["text_adapter"])
    return True


def build_loader(datasets: Dict, batch_size: int, num_workers: int, pin_memory: bool):
    if len(datasets) == 1:
        ds = next(iter(datasets.values()))
    else:
        ds = ConcatDataset([d for d in datasets.values() if len(d) > 0])
    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    return DataLoader(ds, batch_size=batch_size, shuffle=False, **kwargs)


def evaluate_dataset(
    model: AdaptedCLIP,
    clip_model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> pd.DataFrame:
    datasets = get_dataset(
        args.dataset, args.img_size, None, args.shot, "test", logger=logger
    )

    has_text = load_text_adapter(model, args.save_path)
    embed_model = model if has_text else clip_model
    text_embeddings = get_adapted_text_embedding(embed_model, args.dataset, device)

    use_max_sim = args.use_max_sim and args.dataset in ROAD_DATASETS
    e_norm = e_abn = None
    if use_max_sim:
        e_norm, e_abn = get_adapted_multi_text_embeddings(embed_model, args.dataset, device)
        if e_norm is None:
            logger.warning("use_max_sim disabled: failed to load multi-text embeddings")
            use_max_sim = False
        else:
            logger.info(
                "max_sim templates: %d normal, %d abnormal",
                e_norm.shape[0],
                e_abn.shape[0],
            )

    is_road = args.dataset in ROAD_DATASETS
    if is_road:
        columns = ["class name", "pixel AUC", "pixel AP", "pixel FPR95"]
    else:
        columns = ["class name", "pixel AUC", "pixel AP", "image AUC", "image AP"]

    df = pd.DataFrame(columns=columns)
    pin_memory = device.type == "cuda"

    if is_road:
        loader = build_loader(datasets, args.batch_size, args.num_workers, pin_memory)
        text_emb = text_embeddings[next(iter(text_embeddings))]
        with torch.no_grad():
            outputs = run_inference(
                model,
                loader,
                text_emb,
                device,
                args.img_size,
                use_max_sim=use_max_sim,
                e_norm=e_norm,
                e_abn=e_abn,
                logit_scale=args.logit_scale,
                score_mode=args.score_mode,
                fusion_mode=args.fusion_mode,
                disable_zscore=args.disable_zscore,
                eval_mode=args.eval_mode,
            )
        rows = _metrics_row(args, outputs, class_name=args.dataset)
        df.loc[len(df)] = rows
        if args.visualize:
            visualize(
                outputs[0], outputs[2], outputs[4],
                args.save_path, args.dataset, class_name=args.dataset,
            )
    else:
        for class_name, ds in datasets.items():
            if len(ds) == 0:
                continue
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            with torch.no_grad():
                outputs = run_inference(
                    model,
                    loader,
                    text_embeddings[class_name],
                    device,
                    args.img_size,
                    use_max_sim=use_max_sim,
                    e_norm=e_norm,
                    e_abn=e_abn,
                    logit_scale=args.logit_scale,
                    score_mode=args.score_mode,
                    fusion_mode=args.fusion_mode,
                    disable_zscore=args.disable_zscore,
                    eval_mode=args.eval_mode,
                )
            rows = _metrics_row(args, outputs, class_name=class_name)
            df.loc[len(df)] = rows
            if args.visualize:
                visualize(
                    outputs[0], outputs[2], outputs[4],
                    args.save_path, args.dataset, class_name=class_name,
                )

    numeric = df.select_dtypes(include=[np.number]).mean(numeric_only=True)
    avg = {col: numeric.get(col, np.nan) for col in columns if col != "class name"}
    avg["class name"] = "Average"
    df.loc[len(df)] = avg
    return df


def _metrics_row(args, outputs, class_name: str) -> dict:
    masks, labels, preds, preds_image, file_names, road_masks, ignore_masks = outputs
    return metrics_eval(
        masks,
        labels,
        preds,
        preds_image,
        class_name,
        domain=DOMAINS[args.dataset],
        dataset_name=args.dataset,
        road_mask=road_masks,
        ignore_mask=ignore_masks,
        eval_mode=args.eval_mode,
        file_names=file_names,
        save_path=args.save_path,
        disable_normalization=args.disable_normalization,
        score_mode=args.score_mode,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean AA-CLIP evaluation")
    p.add_argument("--model_name", default="ViT-L-14-336")
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--relu", action="store_true")
    p.add_argument("--dataset", default="RoadAnomaly21")
    p.add_argument("--shot", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--save_path", default="ckpt/clean")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--text_adapt_weight", type=float, default=0.1)
    p.add_argument("--image_adapt_weight", type=float, default=0.1)
    p.add_argument("--text_adapt_until", type=int, default=3)
    p.add_argument("--image_adapt_until", type=int, default=6)
    p.add_argument(
        "--eval_mode",
        default="road_or_anom",
        choices=["road_or_anom", "road_only"],
    )
    p.add_argument("--disable_normalization", action="store_true")
    p.add_argument("--disable_zscore", action="store_true")
    p.add_argument("--score_mode", default="prob", choices=["prob", "raw"])
    p.add_argument(
        "--fusion_mode",
        default="mean",
        choices=["mean", "median", "trimmed_mean", "max"],
    )
    p.add_argument("--logit_scale", type=float, default=None)
    p.add_argument("--use_max_sim", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger("test_clean")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(args.save_path, "test.log"), mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    logger.info("args: %s", vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_model = create_model(
        args.model_name, args.img_size, device,
        pretrained="openai", require_pretrained=True,
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
    model.eval()

    load_image_adapter(model, args.save_path, logger)

    df = evaluate_dataset(model, clip_model, args, device, logger)
    logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
    print(df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()
