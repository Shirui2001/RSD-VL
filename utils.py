import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import kornia as K
from torchvision import transforms
import cv2
from PIL import Image
from typing import Optional, Union


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
        ]
    )


def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a], [0, 1, b]])


def rot_img(x, scale):
    theta = scale
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def translation_img(x, translation):
    a, b = translation
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def hflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.hflip(x)
    x = x.squeeze(0)
    return x


def vflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.vflip(x)
    x = x.squeeze(0)
    return x


def add_gaussian_noise(x, scale=0.05):
    std = scale
    noise_mask = torch.randn(x.shape[-2:]) > 3
    noise = torch.randn_like(x) * std  # mean = 0
    noised_img = (x + noise) * noise_mask
    noise_img = torch.where(noised_img > 0, noised_img, x)
    return noise_img


def cos_sim(a_norm, b_norm):
    if len(a_norm.shape) == 2:
        sim_mt = b_norm @ a_norm.transpose(1, 0)
    elif len(a_norm.shape) == 1:
        sim_mt = b_norm @ a_norm
    else:
        raise NotImplementedError
    return sim_mt


# 定义一个自定义的噪音类
class AddGaussianNoise(object):
    def __init__(self, std=1.0, p=0.5):
        """
        mean: 高斯噪声的均值
        std: 高斯噪声的标准差
        p: 添加噪音的概率
        """
        self.std = std
        self.p = p

    def __call__(self, x):
        """
        在数据张量上应用噪音
        """
        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std  # mean = 0
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"(std={self.std}, p={self.p})"


# =====================================================================
# Road scenario: paste COCO OOD object onto a Cityscapes image
# =====================================================================
def paste_coco_to_cityscapes(
    cityscapes_img_path: str,
    coco_ann_path: str,
    coco_img_root: str,
    min_size: int = 30,
    max_size: int = 200,
    scale_range: tuple = (0.5, 1.5),
    gtFine_root: Optional[str] = None,
):
    """
    Paste one COCO OOD object (mask) onto a Cityscapes image.

    Steps:
    1) Read Cityscapes image (BGR) and COCO anomaly mask (PNG with non-zero = object).
    2) Crop object region from COCO image according to mask, resize to random size 30~200
       with a random scale factor in scale_range.
    3) If gtFine_root is provided, use Cityscapes semantic labelIds to find road/sidewalk
       regions as legal paste areas; sample a connected component and location fully
       containing the pasted object. Otherwise, or on failure, fall back to a geometric
       heuristic (lower part of image, left/right bands).
    4) Brightness match: adjust object mean to background local mean.
    5) Alpha blend with weights (bg 0.2, obj 0.8) on masked region.
    6) Generate anomaly mask (1 in pasted region, 0 elsewhere).

    Returns:
        pasted_img (np.ndarray, uint8, BGR), anomaly_mask (np.ndarray, uint8, 0/255), road_mask (np.ndarray, uint8, 0/1)
    """
    # Load Cityscapes image
    city_img = cv2.imread(cityscapes_img_path)
    if city_img is None:
        raise FileNotFoundError(f"Cityscapes image not found: {cityscapes_img_path}")
    h, w, _ = city_img.shape

    # Load COCO mask and image
    mask = cv2.imread(coco_ann_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"COCO mask not found: {coco_ann_path}")
    img_id = os.path.splitext(os.path.basename(coco_ann_path))[0]
    coco_img_path = os.path.join(coco_img_root, f"{img_id}.jpg")
    coco_img = cv2.imread(coco_img_path)
    if coco_img is None:
        raise FileNotFoundError(f"COCO image not found: {coco_img_path}")

    # Extract object region
    obj_mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(obj_mask)
    if len(xs) == 0 or len(ys) == 0:
        # empty mask, return original
        return city_img, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    obj_crop = coco_img[y1 : y2 + 1, x1 : x2 + 1]
    mask_crop = obj_mask[y1 : y2 + 1, x1 : x2 + 1]

    # Random resize
    scale = random.uniform(*scale_range)
    target_h = int(mask_crop.shape[0] * scale)
    target_w = int(mask_crop.shape[1] * scale)
    target_h = np.clip(target_h, min_size, max_size)
    target_w = np.clip(target_w, min_size, max_size)
    obj_crop = cv2.resize(obj_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    ph, pw = obj_crop.shape[:2]

    # ------------------------------------------------------------------
    # Choose paste position: prefer semantic road/sidewalk (if gtFine available)
    # ------------------------------------------------------------------
    def heuristic_position():
        """Fallback: bottom region + left/right bands."""
        y_min = int(h * 0.6)
        y_max = max(y_min, int(h * 0.9) - ph)
        y0_h = random.randint(y_min, max(y_min, min(y_max, h - ph - 1)))
        if random.random() < 0.5:
            x_min_h, x_max_h = 0, int(w * 0.25)
        else:
            x_min_h, x_max_h = int(w * 0.65), w - pw
        x0_h = random.randint(x_min_h, max(x_min_h, min(x_max_h, w - pw - 1)))
        return y0_h, x0_h

    y0, x0 = None, None
    label_img = None  # Initialize label_img for road_mask generation later
    is_trainid = False  # Track whether we're using trainIds or labelIds

    if gtFine_root is not None:
        # Derive gtFine labelIds path from Cityscapes image path:
        # .../leftImg8bit/train/<city>/<name>_leftImg8bit.png ->
        # gtFine_root/<city>/<name>_gtFine_labelTrainIds.png (preferred) or labelIds.png
        city_dir, city_fname = os.path.split(cityscapes_img_path)
        city_name = os.path.basename(city_dir)
        # strip '_leftImg8bit' and file extension to form label basename
        base = city_fname.replace("_leftImg8bit", "")
        stem, _ = os.path.splitext(base)
        
        # Try labelTrainIds.png first (trainId: 0=road)
        label_name = f"{stem}_gtFine_labelTrainIds.png"
        label_path = os.path.join(gtFine_root, city_name, label_name)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        is_trainid = True
        
        # Fallback to labelIds.png if labelTrainIds not found
        if label_img is None:
            label_name = f"{stem}_gtFine_labelIds.png"
            label_path = os.path.join(gtFine_root, city_name, label_name)
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            is_trainid = False

        if label_img is not None and label_img.shape[:2] == (h, w):
            # For labelTrainIds: road=0, sidewalk=1, ignore=255
            # For labelIds: road=7, sidewalk=8
            if is_trainid:
                road_ids = {0, 1}  # trainId: 0=road, 1=sidewalk
            else:
                road_ids = {7, 8}  # labelId: 7=road, 8=sidewalk
            legal_mask = np.isin(label_img, list(road_ids)).astype(np.uint8)
            if legal_mask.any():
                # 连通块分析
                num_labels, comp_map, stats, _ = cv2.connectedComponentsWithStats(
                    legal_mask, connectivity=8
                )
                # 过滤掉背景（label 0）
                areas = stats[1:, cv2.CC_STAT_AREA]
                if len(areas) > 0:
                    max_area = areas.max()
                    area_thresh = 0.5 * max_area
                    large_labels = [
                        i + 1 for i, a in enumerate(areas) if a >= area_thresh
                    ]
                    if large_labels:
                        # 组合这些大连通块，随机采样位置
                        mask_large = np.isin(comp_map, large_labels)
                        ys_l, xs_l = np.where(mask_large)
                        # 多次尝试，确保目标完整落在同一连通块内
                        for _ in range(50):
                            idx_rand = random.randrange(len(xs_l))
                            cy, cx = int(ys_l[idx_rand]), int(xs_l[idx_rand])
                            y0_candidate = cy - ph // 2
                            x0_candidate = cx - pw // 2
                            y0_candidate = max(0, min(y0_candidate, h - ph))
                            x0_candidate = max(0, min(x0_candidate, w - pw))
                            sub = mask_large[
                                y0_candidate : y0_candidate + ph,
                                x0_candidate : x0_candidate + pw,
                            ]
                            if sub.all():
                                y0, x0 = y0_candidate, x0_candidate
                                break

    # 若语义约束失败，则退化为几何启发式
    if y0 is None or x0 is None:
        y0, x0 = heuristic_position()

    # Brightness match: scale object mean to background mean
    bg_patch = city_img[y0 : y0 + ph, x0 : x0 + pw]
    obj_mean = obj_crop[mask_crop > 0].mean() if (mask_crop > 0).any() else 128.0
    bg_mean = bg_patch.mean() if bg_patch.size > 0 else 128.0
    adjust = (bg_mean / (obj_mean + 1e-6))
    obj_adj = np.clip(obj_crop.astype(np.float32) * adjust, 0, 255).astype(np.uint8)

    # Alpha blend on masked region
    alpha = 0.8
    beta = 0.2
    mask_bool = mask_crop.astype(bool)
    pasted = city_img.copy()
    region = pasted[y0 : y0 + ph, x0 : x0 + pw]
    region[mask_bool] = (
        alpha * obj_adj[mask_bool].astype(np.float32)
        + beta * region[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    pasted[y0 : y0 + ph, x0 : x0 + pw] = region

    # Build anomaly mask (0/255)
    anomaly_mask = np.zeros((h, w), dtype=np.uint8)
    anomaly_mask[y0 : y0 + ph, x0 : x0 + pw][mask_bool] = 255
    
    # Build road_mask (0=road, 1=non-road) using semantic labels if available
    road_mask = np.zeros((h, w), dtype=np.uint8)
    if label_img is not None and label_img.shape[:2] == (h, w):
        if is_trainid:
            # labelTrainIds: 0=road, 1=sidewalk, 255=ignore
            # road_mask: 0=road/sidewalk (valid), 1=non-road (ignore)
            road_mask = ((label_img != 0) & (label_img != 1) & (label_img != 255)).astype(np.uint8)
        else:
            # labelIds: 7=road, 8=sidewalk
            road_mask = ((label_img != 7) & (label_img != 8)).astype(np.uint8)
        
        # Apply road_only constraint: anomaly mask should only contain road regions
        # anomaly_mask &= (road_mask == 0)
        anomaly_mask = anomaly_mask * (1 - road_mask)  # Zero out non-road anomalies
    
    return pasted, anomaly_mask, road_mask
