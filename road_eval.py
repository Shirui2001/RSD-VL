from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

LABEL_MAPS = {
    "RoadAnomaly": {0: 0, 1: 1, 2: 1, 255: -1},
    "RoadAnomaly21": {0: 0, 1: 1, 255: -1},
    "RoadObsticle21": {0: 0, 1: 1, 255: -1},
    "FS_LostFound_full": {0: 0, 1: 1, 255: -1},
    "fs_static": {0: 0, 1: 1, 255: -1},
}


def load_original_annotation(path: str | Path, dataset: str, custom: dict[int, int] | None = None):
    mapping = custom or LABEL_MAPS[dataset]
    with Image.open(path) as im:
        raw = np.asarray(im)
    if raw.ndim != 2:
        raise ValueError(f"Expected grayscale label mask, got {raw.shape} for {path}")
    observed = set(int(x) for x in np.unique(raw))
    unknown = observed.difference(mapping)
    if unknown:
        raise ValueError(f"Unexpected labels {unknown} in {path}; verify dataset mapping, do not guess")
    y = np.zeros(raw.shape, dtype=np.uint8)
    valid = np.ones(raw.shape, dtype=bool)
    for value, mapped in mapping.items():
        if mapped == -1:
            valid[raw == value] = False
        else:
            y[raw == value] = mapped
    return y, valid


def pixel_metrics(y_list: list[np.ndarray], s_list: list[np.ndarray], v_list: list[np.ndarray]):
    if not y_list or not (len(y_list) == len(s_list) == len(v_list)):
        raise ValueError("No matched predictions, annotations, and valid masks")
    ys, ss = [], []
    for y, s, v in zip(y_list, s_list, v_list):
        if y.shape != s.shape or y.shape != v.shape:
            raise ValueError(f"Predicted/GT/valid shapes differ: {s.shape} / {y.shape} / {v.shape}")
        if not np.isfinite(s[v]).all():
            raise ValueError("Nonfinite anomaly score detected")
        ys.append(y[v].reshape(-1))
        ss.append(s[v].reshape(-1))
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if np.unique(y).size != 2:
        raise ValueError("Evaluation requires both valid positive and negative pixels")
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    idx = int(np.flatnonzero(tpr >= 0.95)[0])
    return {
        "AuROC": 100.0 * float(roc_auc_score(y, s)),
        "AP": 100.0 * float(average_precision_score(y, s)),
        "FPR95": 100.0 * float(fpr[idx]),
        "valid_pixels": int(y.size),
        "positive_pixels": int(y.sum()),
    }


def load_mapping(path: str | None, dataset: str) -> dict[int, int] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = data.get(dataset, data)
    mapping = {int(k): int(v) for k, v in mapping.items()}
    if not set(mapping.values()).issubset({-1, 0, 1}):
        raise ValueError("Label map values must be -1(ignore), 0(normal), 1(anomaly)")
    return mapping
