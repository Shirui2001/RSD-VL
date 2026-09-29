import numpy as np
import cv2
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
# import ipdb  # Optional debugger, not required for running
from typing import List
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS, PROMPTS_BY_DATASET, DATA_PATH
from model.tokenizer import tokenize
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pandas as pd
from utils import cos_sim
from scipy.ndimage import zoom


class FocalLoss(nn.Module):

    def __init__(
        self,
        apply_nonlin=None,
        alpha=0.75,
        gamma=2.0,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logits, target):
        if target.dim() == 4:  # (B,1,H,W)
            target = target.squeeze(1)
        if logits.dim() == 4:  # (B,2,H,W) -> (N,2), target -> (N,)
            B, C, H, W = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.reshape(-1)
        else:
            target = target.view(-1)
        target = target.long()
        
        logp = F.log_softmax(logits, dim=1)  # (N,2)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
        pt = logpt.exp()  # (N,)
        
        alpha_val = self.alpha if isinstance(self.alpha, float) else 0.25
        alpha_t = torch.where(target == 1, alpha_val, 1 - alpha_val)
        
        loss = -alpha_t * (1 - pt).pow(self.gamma) * logpt
        
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


def masked_binary_dice_loss(input_prob: torch.Tensor,
                            target: torch.Tensor,
                            valid: torch.Tensor,
                            smooth: float = 1.0) -> torch.Tensor:
    
    B = target.shape[0]
    input_flat  = input_prob.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    valid_flat  = valid.reshape(B, -1)

    input_flat  = input_flat * valid_flat
    target_flat = target_flat * valid_flat

    intersection = (input_flat * target_flat).sum(1)
    denom = input_flat.sum(1) + target_flat.sum(1)

    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()



def _get_prompts_for_dataset(dataset_name: str):
    prompt_cfg = PROMPTS_BY_DATASET.get(dataset_name, PROMPTS)
    prompt_normal = prompt_cfg["prompt_normal"]
    prompt_abnormal = prompt_cfg["prompt_abnormal"]
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = prompt_cfg["prompt_templates"]
    return prompt_state, prompt_templates


ROAD_PROMPT_SETTINGS = ("G", "T0", "T1", "T2", "T3", "T4")
ROAD_SHARED_TEMPLATES = (
    "{}.",
    "a photo of {}.",
    "an image of {}.",
)

ROAD_T3_NORMAL = {
    "road": (
        "a clear drivable lane",
        "an empty road with no obstacles",
        "a normal drivable road area",
    ),
    "appearance": (
        "a coarse asphalt road surface",
        "a concrete road surface",
        "a worn road texture",
        "a patched asphalt road surface",
    ),
    "pseudo_anomalous": (
        "tree shadows across the road",
        "road markings on the road surface",
        "cracks on a normal road surface",
        "sunlight reflections on the road surface",
        "glare on wet asphalt",
        "a crosswalk painted on the road surface",
    ),
    "context": (
        "a drivable lane in an urban road scene",
        "a normal lane under strong sunlight",
        "a road surface in a rainy driving scene",
    ),
}
ROAD_T3_ABNORMAL = {
    "obstacle": (
        "an unexpected object blocking the lane",
        "a foreign obstacle on the road",
        "an anomalous object in the drivable area",
        "an obstruction occupying the driving path",
    ),
    "open_set": (
        "an unknown object on the road",
        "an unrecognized obstacle ahead",
        "a hazardous foreign item in the lane",
        "an unidentified object in the drivable area",
    ),
}

ROAD_T4_NORMAL = {
    "road": (
        "an unobstructed lane suitable for driving",
        "a roadway free of obstacles",
        "an ordinary road area suitable for driving",
    ),
    "appearance": (
        "an asphalt roadway with a rough surface texture",
        "a roadway paved with concrete",
        "a normally worn texture on the road surface",
        "a repaired asphalt surface with visible patches",
    ),
    "pseudo_anomalous": (
        "shadows from trees falling across the roadway",
        "painted road markings visible on the roadway",
        "surface cracks on an otherwise normal roadway",
        "sunlight reflected from the roadway surface",
        "bright glare reflected from a wet asphalt surface",
        "painted pedestrian-crossing markings on the roadway",
    ),
    "context": (
        "a lane suitable for driving in an urban street scene",
        "an ordinary traffic lane illuminated by strong sunlight",
        "a normal roadway surface observed during rainy driving conditions",
    ),
}
ROAD_T4_ABNORMAL = {
    "obstacle": (
        "an unforeseen object obstructing the driving lane",
        "an unfamiliar obstacle located on the roadway",
        "an abnormal object within the driving area",
        "an unexpected obstruction positioned in the vehicle path",
    ),
    "open_set": (
        "an unidentified object located on the roadway",
        "an obstacle ahead that cannot be identified",
        "an unfamiliar item in the lane that may pose a driving hazard",
        "an unrecognized object located within the driving area",
    ),
}
ROAD_NORMAL_GROUPS = {
    "T0": ("road",),
    "T1": ("road", "appearance"),
    "T2": ("road", "appearance", "pseudo_anomalous"),
    "T3": ("road", "appearance", "pseudo_anomalous", "context"),
    "T4": ("road", "appearance", "pseudo_anomalous", "context"),
}


def _get_road_prompt_bank(setting=None):
    name = (setting or os.environ.get("RSD_PROMPT_SETTING", "T3")).upper()
    if name not in ROAD_PROMPT_SETTINGS:
        raise ValueError(f"Unknown RSD prompt setting {name!r}; choose {ROAD_PROMPT_SETTINGS}")
    templates = list(ROAD_SHARED_TEMPLATES)
    if name == "G":
        return templates, ["a normal region"], ["an abnormal region"]
    normal_bank = ROAD_T4_NORMAL if name == "T4" else ROAD_T3_NORMAL
    abnormal_bank = ROAD_T4_ABNORMAL if name == "T4" else ROAD_T3_ABNORMAL
    normal = [p for group in ROAD_NORMAL_GROUPS[name] for p in normal_bank[group]]
    abnormal = [p for group in ("obstacle", "open_set") for p in abnormal_bank[group]]
    return templates, normal, abnormal


def get_adapted_multi_text_embeddings(model, dataset_name, device, requires_grad=False):
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21",
        "RoadObsticle21", "FS_LostFound_full", "fs_static",
    }
    if dataset_name not in ROAD_DATASETS:
        return None, None
    templates, normal_states, abnormal_states = _get_road_prompt_bank()

    def encode_multi_states(states_list):
        embeddings_per_state = []
        with torch.set_grad_enabled(requires_grad):
            for state in states_list:
                sentences = [template.format(state) for template in templates]
                tokens = tokenize(sentences).to(device)
                embeddings = model.encode_text(tokens)
                embeddings_per_state.append(F.normalize(embeddings, dim=-1))
        return torch.cat(embeddings_per_state, dim=0)

    return encode_multi_states(normal_states), encode_multi_states(abnormal_states)


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21", 
        "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }
    
    if dataset_name in ROAD_DATASETS:
        prompt_templates, prompt_normal, prompt_abnormal = _get_road_prompt_bank()
    else:
        prompt_cfg = PROMPTS_BY_DATASET.get(dataset_name, PROMPTS)
        prompt_normal = prompt_cfg["prompt_normal"]
        prompt_abnormal = prompt_cfg["prompt_abnormal"]
        prompt_templates = prompt_cfg["prompt_templates"]
    
    def encode_states(states, templates):
        prompted_sentence = []
        for state in states:
            for template in templates:
                prompted_sentence.append(template.format(state))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        embeddings = model.encode_text(prompted_sentence)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embedding = embeddings.mean(dim=0)
        embedding = embedding / embedding.norm()
        return embedding
    
    if dataset_name in ROAD_DATASETS:
        normal_states = prompt_normal
        abnormal_states = prompt_abnormal
    else:
        if class_name == "object":
            real_name = class_name
        else:
            assert class_name in CLASS_NAMES[dataset_name], (
                f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
            )
            real_name = REAL_NAMES[dataset_name][class_name]
        normal_states = [s.format(real_name) for s in prompt_normal]
        abnormal_states = [s.format(real_name) for s in prompt_abnormal]
    
    normal_embedding = encode_states(normal_states, prompt_templates)
    abnormal_embedding = encode_states(abnormal_states, prompt_templates)
    
    cos = float((normal_embedding * abnormal_embedding).sum().item())
    print(f"[TEXT] cos(normal,abnormal)={cos:.4f}")
    if cos > 0.95:
        print(f"[TEXT] ⚠️  WARNING: normal/abnormal文本几乎同义 (cos={cos:.4f} > 0.95)，像素分数天然就只剩极小差值")
    
    text_features = torch.stack([normal_embedding, abnormal_embedding], dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    prompt_state, prompt_templates = _get_prompts_for_dataset(dataset_name)
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device):
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21", 
        "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }
    
    if dataset_name in ROAD_DATASETS:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, "unknown", device  # class_name doesn't matter for Road
        )
        ret_dict = {}
        for class_name in CLASS_NAMES[dataset_name]:
            ret_dict[class_name] = text_features
        return ret_dict
    else:
        ret_dict = {}
        for class_name in CLASS_NAMES[dataset_name]:
            text_features = get_adapted_single_class_text_embedding(
                model, dataset_name, class_name, device
            )
            ret_dict[class_name] = text_features
        return ret_dict


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical", temperature=0.1, use_blur=False, logit_scale=None, use_max_sim=False, E_norm=None, E_abn=None
):
    if use_max_sim and E_norm is not None and E_abn is not None:
        # patch_features: (B, L, 768)
        # E_norm: (K, 768), E_abn: (M, 768)
        sim_norm_all = torch.matmul(patch_features, E_norm.t())  # (B, L, K)
        sim_abn_all = torch.matmul(patch_features, E_abn.t())   # (B, L, M)
        
        # logsumexp pooling (soft-max pool) for stability
        tau = 0.07  
        sim_norm = (torch.logsumexp(sim_norm_all / tau, dim=-1) - math.log(E_norm.shape[0])) * tau  # (B, L)
        sim_abn = (torch.logsumexp(sim_abn_all / tau, dim=-1) - math.log(E_abn.shape[0])) * tau    # (B, L)
        
        score_diff = sim_abn - sim_norm  # (B, L)
        
        if logit_scale is not None:
            score_diff = score_diff * logit_scale
  
        S = torch.stack([torch.zeros_like(score_diff), score_diff], dim=-1)  # (B, L, 2) [normal_score, abnormal_score]
        C = 2
    else:
        # cosine similarity (features already L2-normalized)
        # patch_features: (B, L, 768), epoch_text_feature: (768, 2)
        S = torch.matmul(patch_features, epoch_text_feature)  # (B, L, 2) in [-1, 1]
        
        if logit_scale is not None:
            S = S * logit_scale  
        C = S.shape[-1]
    
    B, L, C = S.shape
    H = int(np.sqrt(L))
    
    if H * H == L:
        H_actual, W_actual = H, H
    else:
        H_actual = int(np.sqrt(L))
        W_actual = (L + H_actual - 1) // H_actual  
        while H_actual * W_actual < L:
            W_actual += 1
        pad_size = H_actual * W_actual - L
        if pad_size > 0:
            padding = torch.zeros(B, pad_size, C, device=S.device, dtype=S.dtype)
            S = torch.cat([S, padding], dim=1)
            L = H_actual * W_actual
    
    if test:
        assert C == 2
        score = S[..., 1] - S[..., 0]  # (B, L) in [-2, 2]
        score = torch.sigmoid(score / temperature)  # (B, L) in [0, 1]
        patch_pred = score.view(B, H_actual, W_actual).unsqueeze(1)  # (B, 1, H, W)
        
        if use_blur:
            sigma = 1 if domain == "Industrial" else 1.5
            kernel_size = 7 if domain == "Industrial" else 9
            patch_pred = gaussian_blur2d(
                patch_pred, (kernel_size, kernel_size), (sigma, sigma)
            )
    else:
        patch_pred = S.permute(0, 2, 1).view(B, C, H_actual, W_actual)
    
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    return patch_preds


focal_loss = FocalLoss(alpha=0.5, gamma=2.0)
dice_loss = BinaryDiceLoss()

_loss_debug_printed = False


def calculate_seg_loss(patch_preds, mask, ignore_mask=None):
    global _finite_check_printed
    if not hasattr(calculate_seg_loss, '_finite_check_printed'):
        calculate_seg_loss._finite_check_printed = False
    if not calculate_seg_loss._finite_check_printed:
        if not torch.isfinite(patch_preds).all():
            print("[DEBUG] patch_preds has NaN/Inf",
                  patch_preds.min().item(), patch_preds.max().item())
        calculate_seg_loss._finite_check_printed = True
    
    if ignore_mask is None:
        prob = torch.softmax(patch_preds, dim=1)  # (B,2,H,W)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)  
        loss = focal_loss(patch_preds, mask)
        loss += dice_loss(prob[:, 0, :, :], (1 - mask).squeeze(1))
        loss += dice_loss(prob[:, 1, :, :], mask.squeeze(1))
        return loss

    tgt = mask.squeeze(1)                       # (B,H,W) 0/1
    road_valid = (ignore_mask.squeeze(1) == 0)  
 
    valid = (ignore_mask.squeeze(1) == 0) | (tgt > 0.5) 

    if valid.sum() == 0:
        return patch_preds.sum() * 0.0

    global _loss_debug_printed
    if not _loss_debug_printed:
        _loss_debug_printed = True
        
        prob_all = torch.softmax(patch_preds, dim=1)
        loss_unmasked = focal_loss(patch_preds, mask)
        loss_unmasked += dice_loss(prob_all[:, 0, :, :], (1 - mask).squeeze(1))
        loss_unmasked += dice_loss(prob_all[:, 1, :, :], mask.squeeze(1))
        
        logit_flat = patch_preds.permute(0, 2, 3, 1)[valid]
        target_flat = tgt[valid].view(-1, 1)
        loss_masked = focal_loss(logit_flat, target_flat)
        
        prob = torch.softmax(patch_preds, dim=1)
        valid_f = valid.float()
        loss_masked += masked_binary_dice_loss(prob[:, 0], (1 - tgt), valid_f)
        loss_masked += masked_binary_dice_loss(prob[:, 1], tgt, valid_f)
        
        total_pixels = tgt.numel()
        valid_pixels = valid.sum().item()
        valid_ratio = valid_pixels / total_pixels
        
        print("\n" + "="*60)
        print("[LOSS DEBUG] First batch comparison:")
        print(f"  Total pixels:  {total_pixels}")
        print(f"  Valid pixels:  {valid_pixels} ({valid_ratio:.2%})")
        print(f"  Ignored pixels: {total_pixels - valid_pixels} ({1-valid_ratio:.2%})")
        print(f"  Loss (unmasked): {loss_unmasked.item():.6f}")
        print(f"  Loss (masked):   {loss_masked.item():.6f}")
        print(f"  Ratio (masked/unmasked): {loss_masked.item() / (loss_unmasked.item() + 1e-9):.4f}")
        print("="*60 + "\n")

    logit_flat = patch_preds.permute(0, 2, 3, 1)[valid]   # (M,2)
    target_flat = tgt[valid].view(-1, 1)                  # (M,1)
    loss = focal_loss(logit_flat, target_flat)

    prob = torch.softmax(patch_preds, dim=1)              # (B,2,H,W)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)  
    valid_f = valid.float()                               # (B,H,W) float

    loss += masked_binary_dice_loss(prob[:, 0], (1 - tgt), valid_f)
    loss += masked_binary_dice_loss(prob[:, 1], tgt, valid_f)
    
    return loss


# ================================================================================================


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) == 0 or len(labels) == 0:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if len(tpr) == 0:
        return np.nan
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return np.nan  # TPR cannot reach 95%
    return float(fpr[idxs[0]])


def dump_top_fp_examples(dataset_name, file_names, pixel_preds, pixel_label, road_mask_2d,
                         eval_mode, save_dir, topk=20, thr=0.99):
    os.makedirs(save_dir, exist_ok=True)

    N, H, W = pixel_preds.shape
    if eval_mode == "road_only":
        valid = (road_mask_2d == 0)
    else:
        valid = np.ones((N, H, W), dtype=bool)

    fp_strength = []  
    q_dyn = 0.999   
    q_top = 0.99    

    for i in range(N):
        neg = (pixel_label[i] == 0) & valid[i]
        if neg.sum() == 0:
            fp_strength.append((-1.0, 0, -1.0, -1.0, -1.0, 0, 1.0, i))
            continue

        s = pixel_preds[i][neg].astype(np.float32)  

        if s.max() > 1.0 or s.min() < 0.0:
            s = 1.0 / (1.0 + np.exp(-s))

        mx = float(np.max(s))
        p99 = float(np.quantile(s, 0.99))
        p999 = float(np.quantile(s, q_dyn))

  
        area_thr = int((s >= thr).sum())

        thr_dyn = p999
        area_dyn = int((s >= thr_dyn).sum())

        thr_top = float(np.quantile(s, q_top))
        top_vals = s[s >= thr_top]
        mean_top1 = float(top_vals.mean()) if top_vals.size > 0 else float(mx)

        fp_strength.append((mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i))

    peak = sorted(fp_strength, reverse=True, key=lambda x: (x[0], x[1]))
    top_peak = peak[:topk]

    print(f"\n[{dataset_name}] Top-{topk} FP samples (peak_fp, sorted by mx then area@thr={thr}):")
    for rank, (mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i) in enumerate(top_peak, 1):
        print(f"  #{rank:02d} mx={mx:.4f}  area@thr={area_thr}  p999={p999:.4f}  mean_top1={mean_top1:.4f}  area@p999={area_dyn}  file={file_names[i]}")

        img_path = os.path.join(DATA_PATH[dataset_name], file_names[i])
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        score = pixel_preds[i]
        heat = np.clip(score * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        road = valid[i].astype(np.uint8) * 255
        road3 = cv2.merge([road, road, road])
        heat = cv2.bitwise_and(heat, road3)

        overlay = cv2.addWeighted(img_bgr, 0.65, heat, 0.35, 0.0)
        out_path = os.path.join(save_dir, f"fp_rank{rank:02d}_{os.path.basename(file_names[i])}.jpg")
        cv2.imwrite(out_path, overlay)

    spread = sorted(fp_strength, reverse=True, key=lambda x: (x[3], x[4], x[5], x[0]))
    top_spread = spread[:topk]

    print(f"\n[{dataset_name}] Top-{topk} FP samples (spread_fp, sorted by p99.9/mean_top1/area@p99.9):")
    for rank, (mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i) in enumerate(top_spread, 1):
        print(f"  #{rank:02d} p999={p999:.4f}  mean_top1={mean_top1:.4f}  area@p999={area_dyn}  mx={mx:.4f}  thr_dyn={thr_dyn:.4f}  file={file_names[i]}")

        img_path = os.path.join(DATA_PATH[dataset_name], file_names[i])
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        score = pixel_preds[i]
        heat = np.clip(score * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        road = valid[i].astype(np.uint8) * 255
        road3 = cv2.merge([road, road, road])
        heat = cv2.bitwise_and(heat, road3)

        overlay = cv2.addWeighted(img_bgr, 0.65, heat, 0.35, 0.0)
        out_path = os.path.join(save_dir, f"fp_spread_rank{rank:02d}_{os.path.basename(file_names[i])}.jpg")
        cv2.imwrite(out_path, overlay)
    
    # --- Save spread_fp_list.json ---
    import json

    fp_json = []
    for rank, item in enumerate(top_spread, 1):
        # item: (mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i)
        mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i = item

        fp_json.append({
            "rank": int(rank),
            "file": str(file_names[i]),   # e.g. "images/39.jpg"
            "mx": float(mx),
            "area_thr": int(area_thr),
            "p99": float(p99),
            "p999": float(p999),
            "mean_top1": float(mean_top1),
            "area_dyn": int(area_dyn),
            "thr_dyn": float(thr_dyn),
        })

    json_path = os.path.join(save_dir, "spread_fp_list.json")
    with open(json_path, "w") as f:
        json.dump(fp_json, f, indent=2)

    print(f"[{dataset_name}] saved spread_fp_list.json -> {json_path} (n={len(fp_json)})")


def metrics_eval(*args, **kwargs):
    raise RuntimeError("Legacy metrics_eval was unsafe for road benchmarking. Use road_test.py and road_eval.py")


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    return (alpha * image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize(
    pixel_label: np.ndarray,
    pixel_preds: np.ndarray,
    file_names: List[str],
    save_dir: str,
    dataset_name: str,
    class_name: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
        pixel_preds = (pixel_preds * 255).astype(np.uint8)
    if pixel_label.dtype != np.uint8:
        pixel_label = pixel_label != 0
        pixel_label = (pixel_label * 255).astype(np.uint8)

    save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    for idx, file in enumerate(file_names):
        image_file = os.path.join(DATA_PATH[dataset_name], file)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, pixel_label.shape[-2:])
        save_image_list = [image]

        if dataset_name == "MVTec":
            damage_name, image_name = file.split("/")[-2:]
            file_name = f"{damage_name}_{image_name}"
        else:
            raise NotImplementedError

        save_image_list.append(cv2.cvtColor(pixel_label[idx, 0], cv2.COLOR_GRAY2RGB))
        save_image_list.append(cv2.cvtColor(pixel_preds[idx], cv2.COLOR_GRAY2RGB))
        save_image_list = save_image_list[:1] + [
            apply_ad_scoremap(image, _) for _ in save_image_list[1:]
        ]
        scoremap = np.vstack(save_image_list)
        cv2.imwrite(os.path.join(save_dir, file_name), scoremap)
