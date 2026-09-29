from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from paper_core import FEATURE_LAYERS, PaperConfig, check_config, fuse_scores, scale_score
from road_eval import LABEL_MAPS, load_mapping, load_original_annotation, pixel_metrics
from road_train import load_clip, seed_everything, sha256_file

ROAD_EVAL = ("RoadAnomaly", "RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full", "fs_static")

PAPER_EXPECTED_COUNTS = {"RoadAnomaly": 60, "RoadObsticle21": 30,
                         "fs_static": 30, "FS_LostFound_full": 100}


def load_metadata(dataset: str, custom_manifest: str | None = None) -> list[dict]:
    path = (Path(custom_manifest).expanduser().resolve() if custom_manifest
            else Path(__file__).resolve().parent / "dataset" / "metadata" / dataset / "full-shot.jsonl")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata missing: {path}")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [r["image_path"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate evaluation image in {path}")
    if not rows:
        raise ValueError("Empty evaluation metadata")
    return rows


def score_one_image(model, image_tensor, normal, abnormal, cfg: PaperConfig, *, feature_space: str):
    """Returned prediction depends ONLY on RGB input + model + prompt bank."""
    with torch.no_grad():
        features, _ = model(image_tensor)
        h, w = image_tensor.shape[-2:]
        maps = [scale_score(f, normal, abnormal, (h, w), cfg,
                            feature_space=feature_space) for f in features]
        return fuse_scores(maps, cfg.fusion)  # [1, H, W], raw, never sigmoid


def get_checkpoint_settings(weights_dir: Path):
    cfg_file = weights_dir / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError("Training config.json missing; refusing unlabeled/unverified checkpoint")
    meta = json.loads(cfg_file.read_text(encoding="utf-8"))
    for name in ("prompt_setting", "clip_sha256", "feature_space", "config"):
        if name not in meta:
            raise ValueError(f"Training metadata lacks {name}")
    cfg = PaperConfig(**meta["config"])
    check_config(cfg)
    return meta, cfg


def load_weights(model, weights_dir: Path, metadata: dict):
    for part in ("text", "image"):
        path = weights_dir / f"{part}_adapter.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained adapter checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("config") != metadata or ckpt.get("kind") != part:
            raise RuntimeError(f"Checkpoint {path} does NOT match config.json")
        module = model.text_adapter if part == "text" else model.image_adapter
        module.load_state_dict(ckpt["component"], strict=True)
    model.eval()


def parse_args():
    p = argparse.ArgumentParser(description="NEW RSD-VL full-resolution evaluation (road only)")
    p.add_argument("--dataset", choices=ROAD_EVAL, required=True)
    p.add_argument("--dataset_root", required=True, help="Must contain images/ and labels_masks/")
    p.add_argument("--clip_checkpoint", default=None, help="Must be supplied for new forward inference")
    p.add_argument("--weights_dir", required=True, help="Directory created by road_train.py")
    p.add_argument("--output", required=True, help="Separate result folder for dataset and seed")
    p.add_argument("--label_map", default=None, help="Optional JSON mapping; VERIFY official version")
    p.add_argument("--manifest", default=None, help="Full dataset JSONL, necessary for RO21 because legacy file has 15/30")
    p.add_argument("--max_samples", type=int, default=0, help="DEBUG ONLY: 0=all")
    p.add_argument("--from_predictions", default=None, help="Re-evaluate stored raw npy maps, skip GPU/model")
    p.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--score_only", action="store_true", help="Compute predictions without opening any GT")
    p.add_argument("--feature_space", choices=["pixel", "patch"], default=None)
    p.add_argument("--fusion", choices=["trimmed_mean", "mean", "max", "single"], default=None,
                   help="Inference-only Table 8 strategy. Does NOT alter trained weights")
    p.add_argument("--allow_unverified_predictions", action="store_true",
                   help="Diagnostic: bypass saved manifest checks, never call result paper reproduction")
    return p.parse_args()


def main():
    args = parse_args()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    metadata, cfg = get_checkpoint_settings(weights_dir)
    if args.fusion is not None:
        cfg = dataclasses.replace(cfg, fusion=args.fusion)
    if args.feature_space and args.feature_space != metadata["feature_space"]:
        raise ValueError("Inference feature-space MUST match training configuration")
    feature_space = metadata["feature_space"]
    prompt_setting = metadata["prompt_setting"]
    os.environ["RSD_PROMPT_SETTING"] = prompt_setting
    seed_everything(int(metadata["seed"]))
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "test.log", encoding="utf-8")])
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not (dataset_root / "images").is_dir():
        raise FileNotFoundError("Expected data-root/images/")
    entries = load_metadata(args.dataset, args.manifest)
    full_size = len(entries)
    expected = PAPER_EXPECTED_COUNTS.get(args.dataset)
    if expected and full_size != expected and not args.max_samples:
        raise RuntimeError(f"Incomplete/non-paper {args.dataset} manifest: {full_size}/{expected} images. "
                           "Supply a VERIFIED --manifest for the intended benchmark split.")
    if args.max_samples:
        if args.max_samples < 0:
            raise ValueError("--max_samples must be >=0")
        entries = entries[:args.max_samples]
    mapping = load_mapping(args.label_map, args.dataset)
    if args.from_predictions:
        predict_root = Path(args.from_predictions).expanduser().resolve()
        previous_manifest = predict_root.parent / "prediction_manifest.jsonl"
        if not args.allow_unverified_predictions:
            if not previous_manifest.is_file():
                raise FileNotFoundError(f"Saved prediction manifest missing: {previous_manifest}")
            rows = [json.loads(line) for line in previous_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(r.get("seed") != metadata["seed"] or r.get("prompt_setting") != prompt_setting
                   or r.get("dataset") != args.dataset or r.get("scoring") != cfg.scoring
                   or r.get("fusion") != cfg.fusion or r.get("feature_space") != feature_space
                   or r.get("base_sha256") != metadata["clip_sha256"] for r in rows):
                raise ValueError("Prediction manifest does not match seed/prompt/scoring/fusion/feature-space/base CLIP")
            if set(r["image"] for r in rows) != set(r["image_path"] for r in entries):
                raise ValueError("Saved prediction manifest does not match expected evaluation image list")
        model, normal, abnormal, device, transform = None, None, None, None, None
    else:
        if not args.clip_checkpoint:
            raise ValueError("--clip_checkpoint is required unless using --from_predictions")
        path = Path(args.clip_checkpoint).expanduser().resolve()
        sha = sha256_file(str(path)) if path.is_file() else ""
        if sha != metadata["clip_sha256"]:
            raise RuntimeError("Base CLIP checkpoint SHA256 differs from training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_clip(str(path), int(metadata["img_size"]), device)
        load_weights(model, weights_dir, metadata)
        from forward_utils import get_adapted_multi_text_embeddings
        with torch.no_grad():
            normal, abnormal = get_adapted_multi_text_embeddings(model, "RoadAnomaly", device, requires_grad=False)
        if normal is None or abnormal is None:
            raise RuntimeError("Could not generate paper prompt bank")
        normal, abnormal = normal.detach(), abnormal.detach()
        img_size = int(metadata["img_size"])
        transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        predict_root = output / "predictions"

    predictions, annotations, valids = [], [], []
    out_manifest = output / "prediction_manifest.jsonl"
    if out_manifest.exists():
        raise FileExistsError(f"Output manifest already exists: {out_manifest}; use a new output directory")
    with out_manifest.open("w", encoding="utf-8") as manifest:
        for k, meta in enumerate(entries):
            image_path = dataset_root / meta["image_path"]
            filename = Path(meta["image_path"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError("Unsafe image path in dataset metadata")
            relative = filename.with_suffix(".npy")
            prediction_path = predict_root / relative
            if args.from_predictions:
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                raw = np.load(prediction_path, allow_pickle=False).astype(np.float32)
            else:
                with Image.open(image_path) as im:
                    rgb = im.convert("RGB")
                    native_w, native_h = rgb.size
                    input_tensor = transform(rgb).unsqueeze(0).to(device)
                scores = score_one_image(model, input_tensor, normal, abnormal, cfg,
                                         feature_space=feature_space)
                scores = F.interpolate(scores.unsqueeze(1), size=(native_h, native_w),
                                       mode="bilinear", align_corners=False)[0, 0]
                raw = scores.cpu().numpy().astype(np.float32)
                if args.save_predictions:
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(prediction_path, raw, allow_pickle=False)
            if not np.isfinite(raw).all():
                raise RuntimeError(f"Nonfinite raw scores {meta['image_path']}")
            manifest.write(json.dumps({"dataset": args.dataset, "image": meta["image_path"],
                                       "pred_shape": list(raw.shape), "prediction": str(prediction_path),
                                       "seed": metadata["seed"], "prompt_setting": prompt_setting,
                                       "scoring":cfg.scoring,"fusion":cfg.fusion,
                                       "feature_space":feature_space,"base_sha256":metadata["clip_sha256"],
                                       "status": "debug-partial" if args.max_samples else "complete"}) + "\n")
            if not args.score_only:
                if not meta.get("mask_path"):
                    raise FileNotFoundError(f"Missing mask_path for {meta['image_path']}")
                label_path = dataset_root / meta["mask_path"]
                y, valid = load_original_annotation(label_path, args.dataset, mapping)
                if raw.shape != y.shape:
                    raise ValueError(f"Prediction/native-label size mismatch for {image_path}: {raw.shape} vs {y.shape}")
                predictions.append(raw)
                annotations.append(y)
                valids.append(valid)
            logging.info("processed %d/%d %s", k+1, len(entries), meta["image_path"])
    summary = {"dataset": args.dataset, "processed": len(entries), "metadata_count": full_size,
               "partial_debug_run": bool(args.max_samples), "paper_config": dataclasses.asdict(cfg),
               "feature_space": feature_space, "checkpoint_dir": str(weights_dir),
               "seed": metadata["seed"], "prompt_setting": prompt_setting,
               "validity": ("UNVERIFIED-PREDICTIONS / NOT PAPER-READY" if args.allow_unverified_predictions
                            else "NEW MEASUREMENTS; NOT VERIFIED AS ORIGINAL PAPER RESULTS")}
    if not args.score_only:
        result = pixel_metrics(annotations, predictions, valids)
        summary["metrics"] = result
        logging.info("Metrics: %s", result)
    (output / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.max_samples:
        logging.warning("DEBUG subset only. NEVER REPORT AS BENCHMARK RESULTS")
    logging.info("Done; new, unverified experimental run. No test-label score-direction selection or road gating.")


if __name__ == "__main__":
    main()
