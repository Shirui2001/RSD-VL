import os
import json
import math
import random
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from utils import AddGaussianNoise, paste_coco_to_cityscapes
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from .constants import CLASS_NAMES, DATA_PATH, DOMAINS
from .transforms import RandomRoadObject


def apply_geom_aug(img3, mask1, road_mask1, ignore_mask1,
                   rot_p=0.5, max_deg=30,
                   aff_p=0.5, max_translate=0.15,
                   hflip_p=0.5, vflip_p=0.5):
    _, H, W = img3.shape

    if random.random() < rot_p:
        angle = random.uniform(-max_deg, max_deg)
        img3 = TF.rotate(img3, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask1 = TF.rotate(mask1, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)
        road_mask1 = TF.rotate(road_mask1, angle=angle, interpolation=InterpolationMode.NEAREST, fill=1)
        ignore_mask1 = TF.rotate(ignore_mask1, angle=angle, interpolation=InterpolationMode.NEAREST, fill=1)

    if random.random() < aff_p:
        max_dx = int(max_translate * W)
        max_dy = int(max_translate * H)
        tx = random.randint(-max_dx, max_dx)
        ty = random.randint(-max_dy, max_dy)

        img3 = TF.affine(img3, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                         interpolation=InterpolationMode.BILINEAR, fill=0)
        mask1 = TF.affine(mask1, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                          interpolation=InterpolationMode.NEAREST, fill=0)
        road_mask1 = TF.affine(road_mask1, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                               interpolation=InterpolationMode.NEAREST, fill=1)
        ignore_mask1 = TF.affine(ignore_mask1, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                                 interpolation=InterpolationMode.NEAREST, fill=1)

    if random.random() < hflip_p:
        img3 = TF.hflip(img3)
        mask1 = TF.hflip(mask1)
        road_mask1 = TF.hflip(road_mask1)
        ignore_mask1 = TF.hflip(ignore_mask1)

    if random.random() < vflip_p:
        img3 = TF.vflip(img3)
        mask1 = TF.vflip(mask1)
        road_mask1 = TF.vflip(road_mask1)
        ignore_mask1 = TF.vflip(ignore_mask1)

    mask1 = (mask1 > 0.5).float()
    road_mask1 = (road_mask1 > 0.5).float()
    ignore_mask1 = (ignore_mask1 > 0.5).float()

    return img3, mask1, road_mask1, ignore_mask1


def apply_photometric_aug(img3, p_brightness=0.5, p_contrast=0.5, p_gamma=0.5, 
                          p_saturation=0.5, p_hue=0.3, p_glare=0.3, p_shadow=0.4):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(img3.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(img3.device)
    img_denorm = img3 * std + mean  
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    if random.random() < p_brightness:
        brightness_factor = random.uniform(0.8, 1.2)  
        img_denorm = torch.clamp(img_denorm * brightness_factor, 0, 1)
    
    if random.random() < p_contrast:
        contrast_factor = random.uniform(0.8, 1.2)  
        img_mean = img_denorm.mean()
        img_denorm = torch.clamp((img_denorm - img_mean) * contrast_factor + img_mean, 0, 1)
    
    if random.random() < p_gamma:
        gamma = random.uniform(0.85, 1.25) 
        img_denorm = torch.pow(img_denorm + 1e-8, gamma)
        img_denorm = torch.clamp(img_denorm, 0, 1)
    
    if random.random() < p_saturation:
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation_factor = random.uniform(0.5, 1.5)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_factor, 0, 255)
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        img_denorm = torch.from_numpy(img_rgb).permute(2, 0, 1).to(img3.device)
    
    if random.random() < p_hue:
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = random.uniform(-20, 20)
        img_hsv[:, :, 0] = np.clip((img_hsv[:, :, 0] + hue_shift) % 360, 0, 360)
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        img_denorm = torch.from_numpy(img_rgb).permute(2, 0, 1).to(img3.device)
    
    if random.random() < p_glare:
        H, W = img_denorm.shape[1], img_denorm.shape[2]
        center_x = random.randint(W//4, 3*W//4)
        center_y = random.randint(H//4, 3*H//4)
        radius = random.randint(min(H, W)//8, min(H, W)//4)
        
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        mask = torch.from_numpy(np.exp(-dist_sq / (2 * (radius**2)))).float().to(img3.device)
        mask = mask.unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
        
        glare_strength = random.uniform(0.08, 0.18) 
        img_denorm = torch.clamp(img_denorm + mask * glare_strength, 0, 1)
        
        gamma_glare = random.uniform(0.75, 0.95) 
        img_denorm = torch.pow(img_denorm + 1e-8, gamma_glare)
        img_denorm = torch.clamp(img_denorm, 0, 1)
    
    if random.random() < p_shadow:
        H, W = img_denorm.shape[1], img_denorm.shape[2]
        num_vertices = random.randint(3, 6)
        vertices = []
        for _ in range(num_vertices):
            x = random.randint(0, W)
            y = random.randint(0, H)
            vertices.append([x, y])
        vertices = np.array(vertices, dtype=np.int32)
        
        shadow_mask = np.zeros((H, W), dtype=np.float32)
        cv2.fillPoly(shadow_mask, [vertices], 1.0)
        shadow_mask = torch.from_numpy(shadow_mask).float().to(img3.device)
        shadow_mask = shadow_mask.unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
        
        shadow_strength = random.uniform(0.25, 0.45) 
        img_denorm = torch.clamp(img_denorm * (1.0 - shadow_mask * shadow_strength), 0, 1)
    
    img_aug = (img_denorm - mean) / std
    return img_aug


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
        use_random_road_object: bool = False,
        random_road_object_params: dict = None,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.text = text
        self.use_random_road_object = use_random_road_object and not text
        self.meta = []
        self.full_shot = "full-shot" in meta_path
        with open(meta_path, "r") as f:
            for line in f:
                self.meta.append(json.loads(line))
        
        if self.use_random_road_object:
            params = random_road_object_params or {}
            self.random_road_object = RandomRoadObject(
                rcp=params.get("rcp", 0.5),
                max_random_poly=params.get("max_random_poly", 10),
                min_sz_px=params.get("min_sz_px", 32),
                max_sz_px=params.get("max_sz_px", 256),
            )
        else:
            self.random_road_object = None

        self.transforms_list = [
            T.RandomApply(
                [T.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            T.RandomApply(
                [T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]

        transform_x = []
        if not text:
            transform_x.append(
                T.RandomApply([T.ColorJitter(brightness=0.5)], p=0.7)
            )
            transform_x.append(
                T.RandomApply([T.ColorJitter(contrast=0.5)], p=0.7)
            )
            transform_x.append(
                T.RandomApply([T.ColorJitter(saturation=0.5)], p=0.7)
            )
        self.transform_x = T.Compose(
            transform_x
            + [
                T.Resize((img_size, img_size), Image.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = T.Compose(
            [
                T.Resize((img_size, img_size), Image.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        if self.random_road_object is not None:
            if meta["label"]:
                mask_path = os.path.join(data_path, meta["mask_path"])
                road_mask = Image.open(mask_path).convert("L")
                road_mask_array = np.array(road_mask)
                road_mask_binary = (road_mask_array == 0).astype(np.uint8) * 255
                road_mask = Image.fromarray(road_mask_binary)
            else:
                road_mask = Image.new("L", img.size, 255)

            sample = {"image": img, "label": road_mask}
            augmented = self.random_road_object(sample)
            img = augmented["image"]

        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        road_mask = torch.zeros_like(mask)    
        ignore_mask = torch.zeros_like(mask)  

        is_cityscapes = ("cityscapes" in img_path.lower()) or ("leftImg8bit" in img_path)
        if is_cityscapes:
            img_dir = os.path.dirname(img_path)
            city_name = os.path.basename(img_dir)
            img_fname = os.path.basename(img_path)
            base = img_fname.replace("_leftImg8bit", "")
            stem, _ = os.path.splitext(base)

            if "leftImg8bit" in data_path:
                gtFine_root = data_path.replace("leftImg8bit", "gtFine")
            else:
                gtFine_root = os.path.join(data_path, "gtFine")

            if "/train/" in img_path or "\\train\\" in img_path:
                split = "train"
            elif "/val/" in img_path or "\\val\\" in img_path:
                split = "val"
            else:
                split = "test"

            label_name = f"{stem}_gtFine_labelTrainIds.png"
            label_path = os.path.join(gtFine_root, split, city_name, label_name)
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            is_trainid = True

            if label_img is None:
                label_name = f"{stem}_gtFine_labelIds.png"
                label_path = os.path.join(gtFine_root, split, city_name, label_name)
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                is_trainid = False

            if label_img is not None:
                label_img = cv2.resize(
                    label_img, (self.img_size, self.img_size),
                    interpolation=cv2.INTER_NEAREST
                )

                if is_trainid:
                    road_keep = (label_img == 0) | (label_img == 1)          
                    ign = (label_img == 255)                                 
                else:
                    road_keep = (label_img == 7) | (label_img == 8)
                    ign = (label_img == 255)

                non_road = (~road_keep) & (~ign)

                road_mask = torch.from_numpy(non_road.astype(np.float32)).unsqueeze(0)
                ign_mask  = torch.from_numpy(ign.astype(np.float32)).unsqueeze(0)

                mask = mask * (1.0 - road_mask)

                ignore_mask = torch.maximum(ignore_mask, torch.maximum(road_mask, ign_mask))

        img, mask, road_mask, ignore_mask = apply_geom_aug(
            img, mask, road_mask, ignore_mask,
            rot_p=0.5, max_deg=30,
            aff_p=0.5, max_translate=0.15,
            hflip_p=0.5,
            vflip_p=0.0,  
        )

        mask = mask * (1.0 - road_mask)
        

        if not self.text:  
            img = apply_photometric_aug(
                img,
                p_brightness=0.8, 
                p_contrast=0.8,    
                p_gamma=0.85,     
                p_saturation=0.2,
                p_hue=0.1,
                p_glare=0.08,     
                p_shadow=0.25,    
            )

        label_val = int(mask.sum().item() > 0)
        
        inputs = {
            "image": img,
            "mask": mask,
            "ignore_mask": ignore_mask,
            "road_mask": road_mask,
            "label": torch.tensor(label_val).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class RoadAnomalyDataset(Dataset):

    def __init__(
        self,
        normal_root: str,
        anomaly_coco_img_root: str,
        anomaly_coco_ann_root: str,
        img_size: int = 518,
        length: int = 6000,
        gtFine_root: Optional[str] = None,
        hard_fp_root: Optional[str] = None,
        hard_fp_prob: float = 0.25,
        hard_fp_topk: int = 20,
    ):
        super().__init__()
        self.normal_imgs = sorted(
            [
                os.path.join(dp, f)
                for dp, _, files in os.walk(normal_root)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.coco_masks = sorted(
            [os.path.join(anomaly_coco_ann_root, f) for f in os.listdir(anomaly_coco_ann_root) if f.endswith(".png")]
        )
        if not self.normal_imgs:
            raise FileNotFoundError(f"No Cityscapes images found at {normal_root}")
        if not self.coco_masks:
            raise FileNotFoundError(f"No COCO OOD object masks found at {anomaly_coco_ann_root}")
        self.coco_img_root = anomaly_coco_img_root
        self.img_size = img_size
        self.length = length
        self.gtFine_root = gtFine_root
        
        self.hard_fp_prob = float(hard_fp_prob)
        self.hard_pool = []  
        
        if hard_fp_root is not None:
            target_sets = ["RoadAnomaly", "RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full", "fs_static"]
            for d in target_sets:
                jp = os.path.join(hard_fp_root, d, "spread_fp_list.json")
                if os.path.isfile(jp):
                    with open(jp, "r") as f:
                        items = json.load(f)
                    for it in items[:hard_fp_topk]:
                        rel = it.get("file", None)  
                        if rel is None:
                            continue
                        self.hard_pool.append((d, rel))
            print(f"[HARD_FP] loaded hard_pool size={len(self.hard_pool)} from {hard_fp_root}, prob={self.hard_fp_prob}, topk={hard_fp_topk}")
        else:
            print("[HARD_FP] disabled (hard_fp_root=None)")

        self.to_tensor = T.Compose(
            [
                T.Resize((img_size, img_size), Image.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.mask_resize = T.Resize((img_size, img_size), Image.NEAREST)
        self.aug = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    def __len__(self):
        return self.length

    def _load_road_mask_best_effort(self, dataset_name: str, rel_img: str) -> np.ndarray:
        H = W = self.img_size
        fallback = np.zeros((H, W), np.float32) 

        if dataset_name not in DATA_PATH:
            return fallback

        root = DATA_PATH[dataset_name]  
        img_path = os.path.join(root, rel_img)
        base = os.path.splitext(img_path)[0]
        rm_path = base.replace(os.sep + "images" + os.sep, os.sep + "road_masks" + os.sep) + ".png"

        if not os.path.exists(rm_path):
            return fallback

        rm = cv2.imread(rm_path, cv2.IMREAD_GRAYSCALE)
        if rm is None:
            return fallback
        rm = cv2.resize(rm, (W, H), interpolation=cv2.INTER_NEAREST)


        nonroad_a = (rm > 0).astype(np.float32)
        nonroad_b = (rm == 0).astype(np.float32)

        road_ratio_a = float((nonroad_a == 0).mean())
        road_ratio_b = float((nonroad_b == 0).mean())

        def ok(r): 
            return (r > 0.05) and (r < 0.95)

        if ok(road_ratio_a) and (not ok(road_ratio_b) or abs(road_ratio_a - 0.6) < abs(road_ratio_b - 0.6)):
            return nonroad_a
        if ok(road_ratio_b):
            return nonroad_b

        return fallback

    def __getitem__(self, idx):
        if len(self.hard_pool) > 0 and np.random.rand() < self.hard_fp_prob:
            d, rel = self.hard_pool[np.random.randint(len(self.hard_pool))]

            root = DATA_PATH[d] 
            img_path = os.path.join(root, rel)

            if os.path.isfile(img_path):
                img_pil = None
                try:
                    img_pil = Image.open(img_path).convert("RGB")
                except Exception as e:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                
                if img_pil is not None:
                    img_tensor = self.to_tensor(img_pil)  

                    road_mask_np = self._load_road_mask_best_effort(d, rel)
                    road_mask = torch.from_numpy(road_mask_np).unsqueeze(0)  # (1,H,W)
                    ignore_mask = road_mask.clone()

                    if not hasattr(self, "_hard_fp_debug_cnt"):
                        self._hard_fp_debug_cnt = 0
                    if self._hard_fp_debug_cnt < 5:
                        rr = float((road_mask_np == 0).mean())
                        print(f"[HARD_FP][DBG] {d}/{rel} road_ratio(valid)={rr:.3f}")
                        self._hard_fp_debug_cnt += 1

                    mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
                    label = torch.tensor(0, dtype=torch.int64)

                    return {
                        "image": img_tensor,
                        "mask": mask,
                        "ignore_mask": ignore_mask,
                        "road_mask": road_mask,
                        "label": label,
                        "file_name": f"HARD_FP::{d}::{rel}",
                        "class_name": "road",
                        "is_hard_fp": torch.tensor(1, dtype=torch.int64),
                    }
        
        is_abnormal = idx % 2 == 1
        if not is_abnormal:
         
            img_path = random.choice(self.normal_imgs)
            img = Image.open(img_path).convert("RGB")
            img = self.to_tensor(img)
            mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.int64)
            
            road_mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            if self.gtFine_root is not None:
                city_dir = os.path.dirname(img_path)
                city_name = os.path.basename(city_dir)
                img_fname = os.path.basename(img_path)
                base = img_fname.replace("_leftImg8bit", "")
                stem, _ = os.path.splitext(base)
                
                label_name = f"{stem}_gtFine_labelTrainIds.png"
                label_path = os.path.join(self.gtFine_root, city_name, label_name)
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                is_trainid = True
                
                if label_img is None:
                    label_name = f"{stem}_gtFine_labelIds.png"
                    label_path = os.path.join(self.gtFine_root, city_name, label_name)
                    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    is_trainid = False
                
                if label_img is None:
                    raise FileNotFoundError(f"Training road annotation missing for {img_path} in {self.gtFine_root}")
                if label_img is not None:
                    label_img = cv2.resize(label_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                    if is_trainid:
                        road_mask_np = (label_img != 0).astype(np.float32)
                    else:
                        road_mask_np = (label_img != 7).astype(np.float32)
                    road_mask = torch.from_numpy(road_mask_np).unsqueeze(0)

            ignore_mask = road_mask.clone()
            
            return {
                "image": img,
                "mask": mask,
                "ignore_mask": ignore_mask,
                "road_mask": road_mask,
                "label": label,
                "file_name": os.path.basename(img_path),
                "class_name": "road",
            }

        city_path = random.choice(self.normal_imgs)
        coco_mask_path = random.choice(self.coco_masks)
        for _attempt in range(10):
            pasted_img, anomaly_mask, road_mask_np = paste_coco_to_cityscapes(
                city_path, coco_mask_path, self.coco_img_root, gtFine_root=self.gtFine_root
            )
            if np.count_nonzero(anomaly_mask) > 0:
                break
            coco_mask_path = random.choice(self.coco_masks)
        else:
            raise RuntimeError("10 synthesis attempts contained no drivable-road anomaly; check COCO masks/gtFine")
        pasted_img = Image.fromarray(cv2.cvtColor(pasted_img, cv2.COLOR_BGR2RGB))
        anomaly_mask_pil = Image.fromarray(anomaly_mask)
        road_mask_pil = Image.fromarray(road_mask_np * 255) 
        
        pasted_img = self.to_tensor(pasted_img)
        anomaly_mask = self.mask_resize(anomaly_mask_pil)
        anomaly_mask = (T.ToTensor()(anomaly_mask) > 0).float()
        
        road_mask = self.mask_resize(road_mask_pil)
        road_mask = (T.ToTensor()(road_mask) > 0).float()
        
        anomaly_mask = anomaly_mask * (1.0 - road_mask)
        if not bool((anomaly_mask > 0.5).any()):
            raise RuntimeError("Synthetic anomaly vanished during road-only mask conversion")
        ignore_mask = road_mask.clone()
        ignore_mask = ignore_mask * (1.0 - anomaly_mask)

        label = torch.tensor(1, dtype=torch.int64)

        if np.random.rand() < 0.001:
            valid = (ignore_mask < 0.5).float()
            anom = (anomaly_mask > 0.5).float()
            print(f"[SANITY E1] valid_ratio={valid.mean().item():.3f} anom_ratio={anom.mean().item():.4f} anom_valid_ratio={(valid*anom).mean().item():.4f}")
        
        return {
            "image": pasted_img,
            "mask": anomaly_mask,
            "ignore_mask": ignore_mask,
            "road_mask": road_mask,
            "label": label,
            "file_name": f"{os.path.basename(city_path)}+{os.path.basename(coco_mask_path)}",
            "class_name": "unknown",
        }


class BaseSingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        class_name: str,
        dataset_name: str = None,
        logger=None,
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.dataset_name = dataset_name
        if "Road" in str(data_path) or any(road_ds in str(data_path) for road_ds in ["RoadAnomaly", "RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full", "fs_static"]):
            print(f"[BaseSingleClassDataset] Initializing with data_path: {self.data_path}")
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                if m["class_name"] == class_name:
                    self.meta.append(m)

        self.transform_x = T.Compose(
            [
                T.Resize((img_size, img_size), Image.BICUBIC),
                T.ToTensor(),
                T.Normalize(  
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = T.Compose(
            [
                T.Resize((img_size, img_size), Image.NEAREST),
                T.ToTensor(),
            ]
        )

        if logger:
            logger.info(f"Class name: {class_name}")
            logger.info(f"data_path: {self.data_path}")
            logger.info(f"Sample number: {len(self.meta)}")
            logger.info("=====================================")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cv2.imread failed for {img_path} (PIL error was: {e})")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(self.data_path, meta["mask_path"])

            mask_img_raw = Image.open(mask_path).convert("L")
            mask_np_raw = np.array(mask_img_raw, dtype=np.uint8)  
            
            ROAD_CHECK_DATASETS = {"RoadAnomaly", "RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full", "fs_static"}
            if self.dataset_name in ROAD_CHECK_DATASETS and not hasattr(BaseSingleClassDataset, '_mask_rules_printed'):
                print("\n" + "="*80)
                print("5个测试数据集的mask处理规则 (dataset/__init__.py):")
                print("="*80)
                print("ROAD_DATASETS = {'RoadAnomaly', 'RoadAnomaly21', 'RoadObsticle21', 'FS_LostFound_full', 'fs_static'}")
                print("if self.dataset_name in ROAD_DATASETS:")
                print("    # For all Road datasets: unified processing")
                print("    # anomaly 只认 1；ignore 认 255")
                print("    anom_np = (mask_np == 1).astype(np.float32)")
                print("    ign_np  = (mask_np == 255).astype(np.float32)")
                print("else:")
                print("    # 其他数据集保持原逻辑：非0即异常；无ignore")
                print("    anom_np = (mask_np != 0).astype(np.float32)")
                print("    ign_np  = np.zeros_like(mask_np, dtype=np.float32)")
                print("="*80 + "\n")
                BaseSingleClassDataset._mask_rules_printed = True
            
            if self.dataset_name == "RoadAnomaly":
                mask_np_raw = np.where(mask_np_raw == 2, 1, mask_np_raw).astype(np.uint8)
            
            if self.dataset_name in ROAD_CHECK_DATASETS and not hasattr(BaseSingleClassDataset, f'_mask_printed_{self.dataset_name}'):
                print(f"[{self.dataset_name}] raw unique: {np.unique(mask_np_raw)[:50]}, min/max: {mask_np_raw.min()}/{mask_np_raw.max()}, shape: {mask_np_raw.shape}")
            
            mask_img_resized = Image.fromarray(mask_np_raw).resize((self.img_size, self.img_size), Image.NEAREST)
            mask_np = np.array(mask_img_resized, dtype=np.uint8)  

            ROAD_IGNORE_255 = {"FS_LostFound_full", "fs_static", "RoadAnomaly21", "RoadObsticle21"}

            if self.dataset_name in ROAD_IGNORE_255:
                anom_np = (mask_np == 1).astype(np.float32)
                ign_np  = (mask_np == 255).astype(np.float32)
            else:
                anom_np = (mask_np != 0).astype(np.float32)
                ign_np  = np.zeros_like(mask_np, dtype=np.float32)

            if self.dataset_name in ROAD_CHECK_DATASETS and not hasattr(BaseSingleClassDataset, f'_mask_printed_{self.dataset_name}'):
                print(f"[{self.dataset_name}] final unique: {np.unique(anom_np)[:50]}, min/max: {anom_np.min()}/{anom_np.max()}")
                setattr(BaseSingleClassDataset, f'_mask_printed_{self.dataset_name}', True)

            mask = torch.from_numpy(anom_np).unsqueeze(0)        # (1,H,W)
            ignore_mask = torch.from_numpy(ign_np).unsqueeze(0)  # (1,H,W)

        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
            ignore_mask = torch.zeros_like(mask)
        
        stem = Path(meta["image_path"]).stem  
        road_mask_path = os.path.join(self.data_path, "road_masks", f"{stem}.png")
        if os.path.exists(road_mask_path):
            rm = Image.open(road_mask_path).convert("L")
            rm = self.transform_mask(rm)  
            rm = (rm != 0).float()  
        else:
            rm = torch.zeros([1, self.img_size, self.img_size]) 
        
        if not hasattr(BaseSingleClassDataset, '_first_sample_printed'):
            print(f"\n[{self.dataset_name}] 第一张样本的mask分布检查:")
            print(f"  torch.unique(mask): {torch.unique(mask).tolist()}")
            print(f"  torch.unique(ignore_mask): {torch.unique(ignore_mask).tolist()}")
            print(f"  torch.unique(road_mask): {torch.unique(rm).tolist()}")
            print(f"  文件名: {meta['image_path']}")
            print()
            BaseSingleClassDataset._first_sample_printed = True
        
        inputs = {
            "image": img,
            "mask": mask,
            "ignore_mask": ignore_mask, 
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
            "road_mask": rm,  
        }
        return inputs


def get_dataset(
    dataset_name: str,
    img_size: int,
    training_mode: str,
    shot: int = -1,
    stage: str = "train",
    logger=None,
    use_random_road_object: bool = False,
    random_road_object_params: dict = None,
    hard_fp_root: Optional[str] = None,
    hard_fp_prob: float = 0.25,
    hard_fp_topk: int = 20,
    dataset_length: Optional[int] = None,  
):
    # Custom synthetic road dataset (Cityscapes normal + COCO OOD pasted)
    if dataset_name == "RoadSynth":
        if stage != "train":
            raise ValueError("RoadSynth only supports stage='train' for now.")
        normal_root = os.path.join(DATA_PATH["Road"], "cityscapes", "leftImg8bit", "train")
        coco_img_root = os.path.join(DATA_PATH["Road"], "coco", "train2017")
        coco_ann_root = os.path.join(DATA_PATH["Road"], "coco", "annotations", "ood_seg_train2017")
        gtFine_root = os.path.join(DATA_PATH["Road"], "cityscapes", "gtFine", "train")
        length = dataset_length if dataset_length is not None else 8000
        if length % 2 != 0:
            length = length - 1  
            if logger:
                logger.warning(f"RoadSynth dataset length adjusted to {length} (must be even to maintain 1:1 normal:abnormal ratio)")
        if logger:
            logger.info(f"RoadSynth dataset length: {length} (normal: {length//2}, abnormal: {length//2}, default: 8000)")
        dataset = RoadAnomalyDataset(
            normal_root=normal_root,
            anomaly_coco_img_root=coco_img_root,
            anomaly_coco_ann_root=coco_ann_root,
            img_size=img_size,
            length=length,
            gtFine_root=gtFine_root,
            hard_fp_root=hard_fp_root,
            hard_fp_prob=hard_fp_prob,
            hard_fp_topk=hard_fp_topk,
        )
        return dataset, dataset

    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )

    if stage == "train":
        if training_mode == "few_shot":
            assert shot > 0, "shot should be positive"
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, f"{shot}-shot.jsonl"
            )
        else:
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, "full-shot.jsonl"
            )

        data_path = DATA_PATH[dataset_name.split("-")[0]]
        text_dataset = BaseDataset(
            data_path, meta_path, img_size, text=True,
            use_random_road_object=False,  # Don't apply to text dataset
            random_road_object_params=random_road_object_params,
        )
        image_dataset = BaseDataset(
            data_path, meta_path, img_size, text=False,
            use_random_road_object=use_random_road_object,
            random_road_object_params=random_road_object_params,
        )
        return text_dataset, image_dataset
    elif stage == "test":
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        class_names = CLASS_NAMES[dataset_name]
        data_path = DATA_PATH[dataset_name]
        if "Road" in dataset_name or dataset_name in ["RoadAnomaly", "RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full", "fs_static"]:
            print(f"[get_dataset] dataset_name: {dataset_name}, data_path: {data_path}")
        if logger:
            logger.info(f"[get_dataset] dataset_name: {dataset_name}, data_path: {data_path}")
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=data_path,
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                dataset_name=dataset_name,
                logger=logger,
            )
            datasets[class_name] = image_dataset
        return datasets
    elif stage == "visualize":
        class_names = CLASS_NAMES[dataset_name]
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                dataset_name=dataset_name,
                logger=None,
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")
