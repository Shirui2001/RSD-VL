import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from road_eval import load_original_annotation, pixel_metrics
from road_test import load_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--predictions_root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--small_max", type=int, default=266)
    ap.add_argument("--medium_max", type=int, default=1512)
    ap.add_argument("--strict_paper_counts", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    entries = load_metadata("FS_LostFound_full", args.manifest)
    if args.strict_paper_counts and len(entries) != 100:
        raise RuntimeError(f"Expected 100 FS L&F validation images for Table 11, found {len(entries)}")
    masks = []
    total_areas = []
    for meta in entries:
        y, valid = load_original_annotation(Path(args.dataset_root) / meta["mask_path"], "FS_LostFound_full")
        n, labels, stats, _ = cv2.connectedComponentsWithStats((y.astype(bool) & valid).astype(np.uint8), 8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        masks.append((y, valid, labels, areas))
        total_areas.extend(int(x) for x in areas)
    if args.strict_paper_counts and len(total_areas) != 188:
        raise RuntimeError(f"Expected 188 connected regions; found {len(total_areas)}. Check prepared label version.")
    buckets = {"Small": [], "Medium": [], "Relatively Large": []}
    for y, valid, labels, areas in masks:
        group = np.zeros(labels.shape, dtype=np.uint8)
        for j, area in enumerate(areas, start=1):
            g = 1 if area <= args.small_max else 2 if area <= args.medium_max else 3
            group[labels == j] = g
            buckets[("Small", "Medium", "Relatively Large")[g-1]].append(int(area))
    group_maps = []
    for y, valid, labels, areas in masks:
        group = np.zeros(labels.shape, dtype=np.uint8)
        for j, area in enumerate(areas, 1):
            group[labels == j] = 1 if area <= args.small_max else 2 if area <= args.medium_max else 3
        group_maps.append(group)
    if args.strict_paper_counts and tuple(map(len, buckets.values())) != (63, 62, 63):
        raise RuntimeError(f"Group counts don't match manuscript 63/62/63: {[len(x) for x in buckets.values()]}")
    predictions = []
    for meta, (y, valid, _, _) in zip(entries, masks):
        pred_path = Path(args.predictions_root) / Path(meta["image_path"]).with_suffix(".npy")
        pred = np.load(pred_path, allow_pickle=False).astype(np.float32)
        if pred.shape != y.shape:
            raise ValueError(f"Native prediction shape mismatch in {pred_path}")
        predictions.append(pred)
    report = {"group_counts": {k: len(v) for k, v in buckets.items()},
              "total_regions": len(total_areas), "small_max": args.small_max,
              "medium_max": args.medium_max, "metrics": {}}
    for g, name in enumerate(buckets, 1):
        y_group = [(group == g).astype(np.uint8) for group in group_maps]
        v_group = [v & ((y == 0) | (group == g)) for (y, v, _, _), group in zip(masks, group_maps)]
        report["metrics"][name] = pixel_metrics(y_group, predictions, v_group)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
