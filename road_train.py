from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from paper_core import (FEATURE_LAYERS, PaperConfig, check_config, effective_region,
                        image_classification_loss, scale_score, seg_loss, tail_losses)

VARIANTS = {
    "B0": ("G", "abnormal_only", False, False),
    "B1": ("G", "rsd", False, False),
    "B2": ("T3", "abnormal_only", False, False),
    "B3": ("T3", "rsd", False, False),
    "B4": ("T3", "rsd", True, False),
    "B5": ("T3", "rsd", True, True),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_dataset(road_root: str) -> None:
    from dataset.constants import DATA_PATH
    root = Path(road_root).expanduser().resolve()
    if not (root / "cityscapes/leftImg8bit/train").is_dir():
        raise FileNotFoundError(f"Missing Cityscapes images: {root}/cityscapes/leftImg8bit/train")
    if not (root / "cityscapes/gtFine/train").is_dir():
        raise FileNotFoundError("Cityscapes gtFine/train is required for the training-only road mask")
    if not (root / "coco/train2017").is_dir() or not (root / "coco/annotations/ood_seg_train2017").is_dir():
        raise FileNotFoundError("COCO/train2017 + annotations/ood_seg_train2017 are required")
    DATA_PATH["Road"] = DATA_PATH["RoadSynth"] = str(root)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_clip(path: str, size: int, device: torch.device):
    if not Path(path).is_file():
        raise FileNotFoundError("Supply an authorized OpenAI CLIP ViT-L/14@336 local checkpoint via --clip_checkpoint")
    import model.clip as clip_factory
    clip_factory._MODEL_CKPT_PATHS["ViT-L-14-336"] = Path(path).expanduser().resolve()
    clip = clip_factory.create_model("ViT-L-14-336", size, pretrained="openai", device=device, require_pretrained=True)
    for p in clip.parameters():
        p.requires_grad_(False)
    clip.eval()
    from model.adapter import AdaptedCLIP
    model = AdaptedCLIP(clip, relu=True, text_adapt_until=3, image_adapt_until=6,
                        text_adapt_weight=0.1, image_adapt_weight=0.1, levels=list(FEATURE_LAYERS)).to(device)
    return model


def get_bank(model, device, *, grad: bool):
    from forward_utils import get_adapted_multi_text_embeddings
    normal, abnormal = get_adapted_multi_text_embeddings(model, "RoadSynth", device, requires_grad=grad)
    if normal is None or abnormal is None:
        raise RuntimeError("Road prompt bank unavailable")
    return normal, abnormal


class BalancedRoadSampler(Sampler):
    def __init__(self, dataset_length: int, batch_size: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Training batches must be even and >=2")
        self.evens = list(range(0, dataset_length, 2))
        self.odds = list(range(1, dataset_length, 2))
        self.half = batch_size // 2

    def __iter__(self):
        even, odd = random.sample(self.evens, len(self.evens)), random.sample(self.odds, len(self.odds))
        for j in range(self.__len__()):
            batch = even[j * self.half:(j + 1) * self.half] + odd[j * self.half:(j + 1) * self.half]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(self.evens), len(self.odds)) // self.half


@torch.no_grad()
def frozen_clip_patches(model, images):
    _, all_tokens = model.clipmodel.encode_image(images, list(FEATURE_LAYERS))
    out = []
    for t in all_tokens:
        t = model.clipmodel.visual.ln_post(t[:, 1:, :])
        t = t @ model.clipmodel.visual.proj
        out.append(F.normalize(t, dim=-1))
    if len(out) != 4:
        raise RuntimeError("Expected 4 intermediate CLIP feature layers")
    return out


def save_checkpoint(path: Path, model, optimizer, *, epoch: int, kind: str, cfg: dict) -> None:
    component = model.text_adapter if kind == "text" else model.image_adapter
    ckpt = {"epoch": epoch, "kind": kind, "component": component.state_dict(),
            "optimizer": optimizer.state_dict(), "config": cfg,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()}
    if torch.cuda.is_available():
        ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def restore_checkpoint(path: Path, model, optimizer, *, kind: str, config: dict) -> int:
    if not path.is_file():
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != kind or ckpt.get("config") != config:
        raise RuntimeError(f"Checkpoint config mismatch: {path}. Use a new output directory.")
    (model.text_adapter if kind == "text" else model.image_adapter).load_state_dict(ckpt["component"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in ckpt:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return int(ckpt["epoch"])


def jsonl(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def train_text(model, loader, optimizer, cfg, args, device, out, metadata):
    model.clipmodel.eval()
    model.image_adapter.eval()
    model.text_adapter.train()
    for p in model.image_adapter.parameters():
        p.requires_grad_(False)
    for p in model.text_adapter.parameters():
        p.requires_grad_(True)
    start = restore_checkpoint(out / "text_adapter.pth", model, optimizer, kind="text", config=metadata) if args.resume else 0
    for epoch in range(start, args.text_epoch):
        totals = []
        for j, batch in enumerate(loader):
            images, y = batch["image"].to(device), batch["mask"].to(device)[:, 0]
            region = batch.get("ignore_mask")
            region = region.to(device)[:, 0] if region is not None else None
            valid = effective_region(y, region, use_erc=cfg.use_erc)
            features = frozen_clip_patches(model, images)
            normal, abnormal = get_bank(model, device, grad=True)
            scale_losses = [seg_loss(scale_score(f, normal, abnormal, tuple(y.shape[-2:]), cfg,
                         feature_space=args.feature_space), y, valid) for f in features]
            loss = torch.stack(scale_losses).mean()  # Stage 1 ONLY pixel-level segmentation loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite text-stage loss epoch={epoch} batch={j}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms = [p.grad.abs().sum().item() for p in model.text_adapter.parameters() if p.grad is not None]
            if not norms or sum(norms) <= 0:
                raise RuntimeError("No text-adapter gradient: check prompt embeddings and stage-1 loss")
            optimizer.step()
            totals.append(loss.item())
            if j % args.log_every == 0:
                logging.info("STAGE1 epoch=%d/%d batch=%d seg=%.6f", epoch+1, args.text_epoch, j, loss.item())
        value = float(np.mean(totals)) if totals else float("nan")
        jsonl(out / "train_history.jsonl", {"stage": "text", "epoch": epoch+1, "loss": value})
        save_checkpoint(out / "text_adapter.pth", model, optimizer, epoch=epoch+1, kind="text", cfg=metadata)
    return model


def train_vision(model, loader, optimizer, cfg, args, device, out, metadata):
    if args.text_epoch and not (out / "text_adapter.pth").is_file():
        raise RuntimeError("Stage 1 checkpoint missing")
    for p in model.text_adapter.parameters():
        p.requires_grad_(False)
    for p in model.image_adapter.parameters():
        p.requires_grad_(True)
    model.clipmodel.eval()
    model.text_adapter.eval()
    model.image_adapter.train()
    with torch.no_grad():
        normal, abnormal = get_bank(model, device, grad=False)
        normal, abnormal = normal.detach(), abnormal.detach()
    start = restore_checkpoint(out / "image_adapter.pth", model, optimizer, kind="image", config=metadata) if args.resume else 0
    for epoch in range(start, args.image_epoch):
        totals = []
        for j, batch in enumerate(loader):
            images = batch["image"].to(device)
            y = batch["mask"].to(device)[:, 0]
            road_ignore = batch.get("ignore_mask")
            road_ignore = road_ignore.to(device)[:, 0] if road_ignore is not None else None
            valid = effective_region(y, road_ignore, use_erc=cfg.use_erc)
            labels = (y * valid).flatten(1).any(1).long()  # Eq. 26: any anomaly within Omega
            patch_feats, cls_feat = model(images)
            pixel, fp, pos = [], [], []
            for f in patch_feats:
                margin = scale_score(f, normal, abnormal, tuple(y.shape[-2:]), cfg, feature_space=args.feature_space)
                pixel.append(seg_loss(margin, y, valid))
                minus, plus = tail_losses(margin, y, valid, labels, epoch, cfg)
                fp.append(minus)
                pos.append(plus)
            loss_seg = torch.stack(pixel).mean()
            loss_cls = image_classification_loss(cls_feat, normal, abnormal, labels, cfg)
            loss_fp = torch.stack(fp).mean()
            loss_pos = torch.stack(pos).mean()
            total = loss_seg + cfg.cls_weight*loss_cls + cfg.fp_weight*loss_fp + cfg.pos_weight*loss_pos
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite vision-stage loss epoch={epoch} batch={j}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), max_norm=1.0)
            optimizer.step()
            totals.append(total.item())
            if j % args.log_every == 0:
                logging.info("STAGE2 epoch=%d/%d batch=%d total=%.5f seg=%.5f cls=%.5f fp=%.5f pos=%.5f",
                             epoch+1, args.image_epoch, j, total.item(), loss_seg.item(), loss_cls.item(),
                             loss_fp.item(), loss_pos.item())
        val = float(np.mean(totals)) if totals else float("nan")
        jsonl(out / "train_history.jsonl", {"stage": "image", "epoch": epoch+1, "loss": val})
        save_checkpoint(out / "image_adapter.pth", model, optimizer, epoch=epoch+1, kind="image", cfg=metadata)
        save_checkpoint(out / f"image_adapter_{epoch+1}.pth", model, optimizer, epoch=epoch+1, kind="image", cfg=metadata)
    return model


def make_config(args):
    if args.variant:
        args.prompt_setting, args.scoring, args.use_erc, args.use_tao = VARIANTS[args.variant]
    cfg = PaperConfig(tau=args.tau, gamma=args.gamma, scoring=args.scoring, fusion=args.fusion,
                      fp_ratios=(args.fp_ratio_001, args.fp_ratio_01, args.fp_ratio_05),
                      pos_ratio=args.pos_ratio, fp_weight=args.fp_weight, pos_weight=args.pos_weight,
                      cls_weight=args.cls_weight, use_erc=args.use_erc, use_tao=args.use_tao,
                      fp_dynamic_start=args.fp_dynamic_start, fp_dynamic_decay=args.fp_dynamic_decay,
                      fp_dynamic_min=args.fp_dynamic_min, fp_fixed_5pct=args.fp_fixed_5pct,
                      fp_fixed_0p1pct=args.fp_fixed_0p1pct, pos_threshold=args.pos_threshold)
    check_config(cfg)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="NEW RSD-VL reproducibility training (road only)")
    p.add_argument("--road_root", required=True, help="Root containing cityscapes/ and coco/")
    p.add_argument("--clip_checkpoint", required=True, help="Local OpenAI ViT-L-14-336px.pt")
    p.add_argument("--output", required=True, help="Dedicated folder PER variant and seed; never reuse")
    p.add_argument("--variant", choices=list(VARIANTS), default=None)
    p.add_argument("--prompt_setting", choices=["G", "T0", "T1", "T2", "T3", "T4"], default="T3")
    p.add_argument("--scoring", choices=["rsd", "normalized_relative", "negative_normal", "abnormal_only"], default="rsd")
    p.add_argument("--fusion", choices=["trimmed_mean", "mean", "max", "single"], default="trimmed_mean")
    p.add_argument("--use_erc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_tao", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--feature_space", choices=["pixel", "patch"], default="pixel",
                   help="pixel=manuscript Eq.16; patch is diagnostic only and cannot claim paper reproduction")
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--synthetic_length", type=int, default=8000)
    p.add_argument("--text_epoch", type=int, default=5)
    p.add_argument("--image_epoch", type=int, default=10)
    p.add_argument("--text_batch_size", type=int, default=16)
    p.add_argument("--image_batch_size", type=int, default=4)
    p.add_argument("--text_lr", type=float, default=1e-5)
    p.add_argument("--image_lr", type=float, default=5e-4)
    p.add_argument("--cls_weight", type=float, default=0.25, help="Legacy config; value ABSENT from manuscript")
    p.add_argument("--fp_weight", type=float, default=0.25)
    p.add_argument("--pos_weight", type=float, default=0.10)
    p.add_argument("--fp_ratio_001", type=float, default=0.001)
    p.add_argument("--fp_ratio_01", type=float, default=0.01)
    p.add_argument("--fp_ratio_05", type=float, default=0.05)
    p.add_argument("--pos_ratio", type=float, default=0.10)
    p.add_argument("--fp_dynamic_start", type=float, default=0.22)
    p.add_argument("--fp_dynamic_decay", type=float, default=0.02)
    p.add_argument("--fp_dynamic_min", type=float, default=0.10)
    p.add_argument("--fp_fixed_5pct", type=float, default=0.08)
    p.add_argument("--fp_fixed_0p1pct", type=float, default=0.25)
    p.add_argument("--pos_threshold", type=float, default=0.55)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", action="store_true", help="Resume only a matching checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    if args.img_size % 14:
        raise ValueError("Image resolution must be divisible by CLIP patch size 14")
    if args.synthetic_length < 2 or args.synthetic_length % 2:
        raise ValueError("Synthetic length must be a positive even integer")
    cfg = make_config(args)
    if args.text_epoch <= 0 or args.image_epoch <= 0:
        raise ValueError("The paper's two-stage procedure requires positive text/image epochs")
    os.environ["RSD_PROMPT_SETTING"] = args.prompt_setting
    seed_everything(args.seed)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
        logging.FileHandler(out / "train.log", mode="a", encoding="utf-8")],
        format="%(asctime)s %(levelname)s %(message)s")
    from forward_utils import _get_road_prompt_bank
    templates, n_states, a_states = _get_road_prompt_bank()
    if not n_states or not a_states:
        raise RuntimeError("Empty prompt bank")
    if not Path(args.clip_checkpoint).is_file():
        raise FileNotFoundError("Please provide the real CLIP ViT-L/14@336 checkpoint")
    ckpt_hash = sha256_file(args.clip_checkpoint)
    metadata = {"paper": "RSD-VL revised 2026-09; implementation, outputs",
                "seed": args.seed, "prompt_setting": args.prompt_setting,
                "prompt_normal_states": n_states, "prompt_abnormal_states": a_states,
                "prompt_templates": templates, "config": dataclasses.asdict(cfg),
                "clip_sha256": ckpt_hash, "feature_space": args.feature_space,
                "img_size": args.img_size, "text_epoch": args.text_epoch,
                "image_epoch": args.image_epoch, "text_lr": args.text_lr,
                "image_lr": args.image_lr, "synthetic_length": args.synthetic_length,
                "text_batch_size": args.text_batch_size, "image_batch_size": args.image_batch_size}
    metadata = json.loads(json.dumps(metadata))
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("Output folder contains different config. Use separate folder for EVERY setting/seed")
    if not args.resume and ((out / "text_adapter.pth").exists() or (out / "image_adapter.pth").exists()):
        raise RuntimeError("Checkpoints already exist. Use --resume or a new output folder")
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    configure_dataset(args.road_root)
    from dataset import get_dataset
    text_dataset, image_dataset = get_dataset("RoadSynth", args.img_size, "full_shot", -1, "train",
                                            dataset_length=args.synthetic_length)
    kw = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    text_loader = DataLoader(text_dataset, batch_sampler=BalancedRoadSampler(len(text_dataset), args.text_batch_size), **kw)
    image_loader = DataLoader(image_dataset, batch_sampler=BalancedRoadSampler(len(image_dataset), args.image_batch_size), **kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_clip(args.clip_checkpoint, args.img_size, device)
    text_optimizer = torch.optim.Adam(model.text_adapter.parameters(), lr=args.text_lr, weight_decay=0.0)
    image_optimizer = torch.optim.Adam(model.image_adapter.parameters(), lr=args.image_lr, weight_decay=0.0)
    logging.info("Config=%s; normal full prompts=%d abnormal full prompts=%d; Device=%s", cfg,
                 len(n_states)*len(templates), len(a_states)*len(templates), device)
    if args.text_epoch:
        train_text(model, text_loader, text_optimizer, cfg, args, device, out, metadata)
    train_vision(model, image_loader, image_optimizer, cfg, args, device, out, metadata)


if __name__ == "__main__":
    main()
