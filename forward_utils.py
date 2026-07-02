import numpy as np
import cv2
import os
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

# ================================================================================================
# The following code is used to get criterion for training


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

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
        """
        Stable Focal Loss implementation using log_softmax.
        
        Args:
            logits: (N,2) or (B,2,H,W) raw logits
            target: (N,) or (B,H,W) or (B,1,H,W) with values {0,1}
        """
        # ---- shape normalize ----
        if target.dim() == 4:  # (B,1,H,W)
            target = target.squeeze(1)
        if logits.dim() == 4:  # (B,2,H,W) -> (N,2), target -> (N,)
            B, C, H, W = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.reshape(-1)
        else:
            target = target.view(-1)
        target = target.long()
        
        # ---- stable log-softmax ----
        logp = F.log_softmax(logits, dim=1)  # (N,2)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
        pt = logpt.exp()  # (N,)
        
        # ---- alpha weighting: class 1 uses alpha, class 0 uses (1-alpha) ----
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
    """
    Compute binary Dice loss only on valid pixels.
    
    Args:
        input_prob: (B,H,W) in [0,1] - predicted probability
        target:     (B,H,W) in {0,1} - ground truth
        valid:      (B,H,W) in {0,1} - 1=keep, 0=ignore
    
    Returns:
        Dice loss (scalar)
    """
    B = target.shape[0]
    input_flat  = input_prob.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    valid_flat  = valid.reshape(B, -1)

    # Apply valid mask
    input_flat  = input_flat * valid_flat
    target_flat = target_flat * valid_flat

    intersection = (input_flat * target_flat).sum(1)
    denom = input_flat.sum(1) + target_flat.sum(1)

    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()


# ================================================================================================
# The following code is used to get adapted text embeddings
def _get_prompts_for_dataset(dataset_name: str):
    """
    Return dataset-specific prompts (normal/abnormal + templates) if provided,
    otherwise fall back to the default PROMPTS.
    """
    prompt_cfg = PROMPTS_BY_DATASET.get(dataset_name, PROMPTS)
    prompt_normal = prompt_cfg["prompt_normal"]
    prompt_abnormal = prompt_cfg["prompt_abnormal"]
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = prompt_cfg["prompt_templates"]
    return prompt_state, prompt_templates


def _get_road_prompt_bank():
    """Shared prompt bank for Road series datasets (training/testing aligned)."""
    prompt_templates = [
        "a photo of {}.",
        "a street scene with {}.",
        "a dashcam view of {}.",
    ]

    road_normals = [
        # 基础正常
        "a clear drivable lane",
        "an empty road with no obstacles",
        "a clean asphalt road surface",
        "a clear road ahead",
        "a free-flowing lane with no hazards",
        # hard negatives（阴影/反光/湿路）
        "shadows cast on the road",
        "sunlight reflection on the road",
        "glare on the road",
        "reflections on wet asphalt",
        "a wet road surface",
        "lens flare in a road scene",
        # RoadAnomaly/LostFound 常见 FP（纹理/标线/路面结构）
        "road markings on the road",
        "a crosswalk on the road",
        "a manhole cover on the road",
        "oil stains on the road surface",
        #光照 
        "tree shadows across the road",
        "specular highlights on the road surface",
        "overexposed highlights on the road",
        #结构/纹理 hard negatives
        "a stop line painted on the road",
        "asphalt cracks on the road surface",
        "a patched asphalt road surface",
        "tar seams on the asphalt",
        "water stains on asphalt",

    ]

    road_abnormals = [
        # 泛化类（必须有）
        "an obstacle blocking the lane",
        "a foreign object on the road surface",
        "an unexpected object in the driving lane",
        "an object blocking the drivable area",
        "debris scattered on the road",
        # 常见小物体（RoadAnomaly/21 很关键）
        "a cardboard box on the road",
        "a plastic bag on the road",
        "a trash bag on the road",
        "a tire on the road",
        "a rock on the road",
        "a fallen tree branch on the road",
        "a wooden plank on the road",
        "broken parts on the road surface",
        # 施工/路障（RoadObsticle/LostFound 关键）
        "a traffic cone on the road",
        "a construction barrier blocking the lane",
        "a road work sign in the lane",
        # 大物体（LostFound 很关键）
        "a large obstacle blocking the lane",
    ]

    return prompt_templates, road_normals, road_abnormals


def get_adapted_multi_text_embeddings(model, dataset_name, device):
    """
    B) 最小改法：encode多条ROAD_NORMALS和ROAD_ABNORMALS，返回所有embedding
    用于max相似度计算
    """
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21", 
        "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }
    
    if dataset_name not in ROAD_DATASETS:
        return None, None
    
    # Use shared Road prompt bank for alignment across training/testing.
    prompt_templates, ROAD_NORMALS, ROAD_ABNORMALS = _get_road_prompt_bank()
    
    def encode_multi_states(states_list, templates):
        """Encode multiple states, return all embeddings (not averaged)."""
        all_embeddings = []
        with torch.no_grad():  # 减少内存占用
            for state in states_list:
                prompted_sentence = []
                for template in templates:
                    prompted_sentence.append(template.format(state))
                prompted_sentence = tokenize(prompted_sentence).to(device)
                embeddings = model.encode_text(prompted_sentence)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                # 对每个state的所有template变体求平均
                state_embedding = embeddings.mean(dim=0)
                state_embedding = state_embedding / state_embedding.norm()
                all_embeddings.append(state_embedding)
                # 清理中间变量以释放内存
                del embeddings, prompted_sentence
                torch.cuda.empty_cache()
        return torch.stack(all_embeddings, dim=0)  # (K, 768) or (M, 768)
    
    E_norm = encode_multi_states(ROAD_NORMALS, prompt_templates)  # (K, 768)
    E_abn = encode_multi_states(ROAD_ABNORMALS, prompt_templates)  # (M, 768)
    
    return E_norm, E_abn


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    """
    Generate text embeddings for a single class.
    For Road series datasets, use a prompt bank focusing on shadow/reflection false positives.
    """
    # Road series datasets use fixed anchor pair
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21", 
        "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }
    
    # Get prompt configuration (Road uses shared bank to align with max-sim)
    if dataset_name in ROAD_DATASETS:
        prompt_templates, prompt_normal, prompt_abnormal = _get_road_prompt_bank()
    else:
        prompt_cfg = PROMPTS_BY_DATASET.get(dataset_name, PROMPTS)
        prompt_normal = prompt_cfg["prompt_normal"]
        prompt_abnormal = prompt_cfg["prompt_abnormal"]
        prompt_templates = prompt_cfg["prompt_templates"]
    
    def encode_states(states, templates):
        """Encode a list of prompt states into a single embedding."""
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
        # ---- Road-only: CLIP-friendly prompt bank ----
        # Normal: must emphasize "drivable + no obstacles"
        ROAD_NORMALS = [
            "a clear drivable lane",
            "an empty road with no obstacles",
            "a clean asphalt road surface",
            "a clear road ahead",
            "a free-flowing lane with no hazards",
            
            # hard negatives: shadow / reflection / wet (already added, very good)
            "a road shadow on the road surface",
            "shadows cast on the road",
            "tree shadows on the road",
            "building shadows on the road",
            "vehicle shadows on the road",
            "sunlight reflection on the road",
            "specular highlights on the road surface",
            "glare on the road",
            "reflections on wet asphalt",
            "a wet road surface",
            "a puddle on the road",
            "asphalt texture on the road",
            "road markings on the road",
            "lens flare in a road scene",
            "overexposed highlights on the road",
            "harsh lighting on the road",
            "reflection from headlights on the road",
            "bright sun glare on asphalt",
            
            # Additional hard negatives: more specific shadow/reflection descriptions
            "tree shadows across the road",
            "building shadows on the asphalt",
            "cast shadows from vehicles on the road",
            "specular reflection on wet asphalt",
            "sun glare on road markings",
        ]

        # Abnormal: only describe "physical obstacles/blocking/foreign objects", avoid shadow/reflection/wet
        ROAD_ABNORMALS = [
            "an obstacle blocking the lane",
            "debris on the road",
            "a foreign object on the road surface",
            "a blocked driving lane",
            "a hazard object on the road",
        ]

        # override templates for Road to avoid awkward sentences like "a photo of A normal ..."
        prompt_templates = [
            "a photo of {}.",
            "a street scene with {}.",
        ]

        normal_states = ROAD_NORMALS
        abnormal_states = ROAD_ABNORMALS
    else:
        # Original logic for other datasets
        if class_name == "object":
            real_name = class_name
        else:
            assert class_name in CLASS_NAMES[dataset_name], (
                f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
            )
            real_name = REAL_NAMES[dataset_name][class_name]
        normal_states = [s.format(real_name) for s in prompt_normal]
        abnormal_states = [s.format(real_name) for s in prompt_abnormal]
    
    # Generate embeddings
    normal_embedding = encode_states(normal_states, prompt_templates)
    abnormal_embedding = encode_states(abnormal_states, prompt_templates)
    
    # ✅ A) 打印 cos(normal, abnormal) 检查文本原型是否太像
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
    """
    Generate text embeddings for all classes in a dataset.
    For Road series datasets, return a single fixed anchor pair (not class-specific).
    """
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21", 
        "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }
    
    if dataset_name in ROAD_DATASETS:
        # For Road datasets, use fixed anchor pair regardless of class_name
        # Use "unknown" as key for consistency, but the embedding is fixed
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, "unknown", device  # class_name doesn't matter for Road
        )
        # Return the same embedding for all class_names
        ret_dict = {}
        for class_name in CLASS_NAMES[dataset_name]:
            ret_dict[class_name] = text_features
        return ret_dict
    else:
        # Original logic for other datasets
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
    """
    Calculate similarity map between patch features and text features.
    
    Args:
        use_max_sim: If True, use max similarity over multiple text embeddings (B method)
        E_norm: (K, 768) normal text embeddings for max similarity
        E_abn: (M, 768) abnormal text embeddings for max similarity
    """
    # ✅ B) Max相似度方法：对每个patch feature，计算与多条文本的max相似度
    if use_max_sim and E_norm is not None and E_abn is not None:
        # patch_features: (B, L, 768)
        # E_norm: (K, 768), E_abn: (M, 768)
        sim_norm_all = torch.matmul(patch_features, E_norm.t())  # (B, L, K)
        sim_abn_all = torch.matmul(patch_features, E_abn.t())   # (B, L, M)
        
        # logsumexp pooling (soft-max pool) for stability
        tau = 0.07  # 0.05~0.1 recommended
        sim_norm = torch.logsumexp(sim_norm_all / tau, dim=-1) * tau  # (B, L)
        sim_abn = torch.logsumexp(sim_abn_all / tau, dim=-1) * tau    # (B, L)
        
        # score = sim_abn - sim_norm
        score_diff = sim_abn - sim_norm  # (B, L)
        
        # ✅ C) 应用logit_scale
        if logit_scale is not None:
            score_diff = score_diff * logit_scale
        
        # 转换为(B, L, 2)格式以兼容后续处理
        S = torch.stack([-score_diff, score_diff], dim=-1)  # (B, L, 2) [normal_score, abnormal_score]
        C = 2
    else:
        # 原始方法：使用单个text embedding
        # cosine similarity (features already L2-normalized)
        # patch_features: (B, L, 768), epoch_text_feature: (768, 2)
        S = torch.matmul(patch_features, epoch_text_feature)  # (B, L, 2) in [-1, 1]
        
        # ✅ C) 给分数加尺度（logit_scale）
        if logit_scale is not None:
            S = S * logit_scale  # 放大相似度差异，增强margin
        C = S.shape[-1]
    
    B, L, C = S.shape
    H = int(np.sqrt(L))
    
    # ✅ 修复：鲁棒处理非完全平方数的 patch 数量
    # 计算实际的 H 和 W（可能是矩形而不是正方形）
    if H * H == L:
        H_actual, W_actual = H, H
    else:
        # 尝试找到最接近的矩形
        H_actual = int(np.sqrt(L))
        W_actual = (L + H_actual - 1) // H_actual  # 向上取整
        # 如果乘积超过 L，调整
        while H_actual * W_actual < L:
            W_actual += 1
        # 填充到矩形
        pad_size = H_actual * W_actual - L
        if pad_size > 0:
            # 用零填充
            padding = torch.zeros(B, pad_size, C, device=S.device, dtype=S.dtype)
            S = torch.cat([S, padding], dim=1)
            L = H_actual * W_actual
    
    if test:
        assert C == 2
        # Use stable score calculation: abnormal - normal, then sigmoid
        score = S[..., 1] - S[..., 0]  # (B, L) in [-2, 2]
        score = torch.sigmoid(score / temperature)  # (B, L) in [0, 1]
        # Reshape to spatial dimensions
        patch_pred = score.view(B, H_actual, W_actual).unsqueeze(1)  # (B, 1, H, W)
        
        # Apply gaussian blur if enabled
        if use_blur:
            sigma = 1 if domain == "Industrial" else 1.5
            kernel_size = 7 if domain == "Industrial" else 9
            patch_pred = gaussian_blur2d(
                patch_pred, (kernel_size, kernel_size), (sigma, sigma)
            )
    else:
        # Training mode: reshape to (B, C, H, W) for softmax
        patch_pred = S.permute(0, 2, 1).view(B, C, H_actual, W_actual)
    
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    # IMPORTANT:
    # - training: return logits (B,2,H,W) for focal/dice
    # - test: return prob score map (B,1,H,W) already in [0,1]
    return patch_preds


focal_loss = FocalLoss(alpha=0.5, gamma=2.0)
dice_loss = BinaryDiceLoss()

# Global flag for loss comparison debug (only print once)
_loss_debug_printed = False


def calculate_seg_loss(patch_preds, mask, ignore_mask=None):
    """
    Calculate segmentation loss (Focal + Dice) with optional masking.
    Only pixels where ignore_mask==0 participate in the loss computation.
    
    Args:
        patch_preds: (B,2,H,W) - logits or scores for 2 classes
        mask:        (B,1,H,W) - ground truth, 0=normal, 1=anomaly
        ignore_mask: (B,1,H,W) - ignore regions, 1=ignore, 0=valid; None means all valid
    
    Returns:
        Combined loss (scalar)
    """
    # ---- Finite check (only print once) ----
    global _finite_check_printed
    if not hasattr(calculate_seg_loss, '_finite_check_printed'):
        calculate_seg_loss._finite_check_printed = False
    if not calculate_seg_loss._finite_check_printed:
        if not torch.isfinite(patch_preds).all():
            print("[DEBUG] patch_preds has NaN/Inf",
                  patch_preds.min().item(), patch_preds.max().item())
        calculate_seg_loss._finite_check_printed = True
    
    if ignore_mask is None:
        # Original logic: compute loss on all pixels
        # Use softmax probabilities for Dice loss (more stable than logits)
        prob = torch.softmax(patch_preds, dim=1)  # (B,2,H,W)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)  # 防极端
        loss = focal_loss(patch_preds, mask)
        loss += dice_loss(prob[:, 0, :, :], (1 - mask).squeeze(1))
        loss += dice_loss(prob[:, 1, :, :], mask.squeeze(1))
        return loss

    # ---- Masked logic: only compute loss on valid (non-ignored) pixels ----
    # Convert ignore_mask (1=ignore) to valid mask (True=valid)
    tgt = mask.squeeze(1)                       # (B,H,W) 0/1
    road_valid = (ignore_mask.squeeze(1) == 0)  # road region (True=valid)
    # E1 配置：road_or_anom 模式 - 保证所有异常像素（无论在道路上还是非道路区域）都参与评估
    # 这样训练和评估保持一致，避免训练时忽略道路外的异常像素
    valid = (ignore_mask.squeeze(1) == 0) | (tgt > 0.5)  # road OR anomaly

    # If no valid pixels in this batch, return zero loss to avoid NaN
    if valid.sum() == 0:
        return patch_preds.sum() * 0.0

    # ---- DEBUG: Loss comparison (only print once) ----
    global _loss_debug_printed
    if not _loss_debug_printed:
        _loss_debug_printed = True
        
        # Compute unmasked loss (original logic)
        prob_all = torch.softmax(patch_preds, dim=1)
        loss_unmasked = focal_loss(patch_preds, mask)
        loss_unmasked += dice_loss(prob_all[:, 0, :, :], (1 - mask).squeeze(1))
        loss_unmasked += dice_loss(prob_all[:, 1, :, :], mask.squeeze(1))
        
        # Compute masked loss (current logic)
        logit_flat = patch_preds.permute(0, 2, 3, 1)[valid]
        target_flat = tgt[valid].view(-1, 1)
        loss_masked = focal_loss(logit_flat, target_flat)
        
        prob = torch.softmax(patch_preds, dim=1)
        valid_f = valid.float()
        loss_masked += masked_binary_dice_loss(prob[:, 0], (1 - tgt), valid_f)
        loss_masked += masked_binary_dice_loss(prob[:, 1], tgt, valid_f)
        
        # Statistics
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

    # 1) Masked Focal Loss: only use valid pixels
    # Reshape to (M,2) and (M,1) where M = number of valid pixels
    logit_flat = patch_preds.permute(0, 2, 3, 1)[valid]   # (M,2)
    target_flat = tgt[valid].view(-1, 1)                  # (M,1)
    loss = focal_loss(logit_flat, target_flat)

    # 2) Masked Dice Loss: compute on softmax probabilities
    prob = torch.softmax(patch_preds, dim=1)              # (B,2,H,W)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)  # 防极端
    valid_f = valid.float()                               # (B,H,W) float

    loss += masked_binary_dice_loss(prob[:, 0], (1 - tgt), valid_f)
    loss += masked_binary_dice_loss(prob[:, 1], tgt, valid_f)
    
    return loss


# ================================================================================================


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute FPR@95TPR for binary labels.
    labels: 1 = anomaly, 0 = normal.
    Returns np.nan if TPR cannot reach 95%.
    """
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
    """Dump top-K false positive examples with overlay visualization."""
    os.makedirs(save_dir, exist_ok=True)

    N, H, W = pixel_preds.shape
    # road_only: 只在 road 区域看 FP
    if eval_mode == "road_only":
        valid = (road_mask_2d == 0)
    else:
        valid = np.ones((N, H, W), dtype=bool)

    # ---------------------------
    # FP ranking: peak vs spread
    # ---------------------------
    fp_strength = []  # each item: (mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i)
    q_dyn = 0.999   # 你说的 p99.9
    q_top = 0.99    # top 1% 用于 mean_top1

    for i in range(N):
        neg = (pixel_label[i] == 0) & valid[i]
        if neg.sum() == 0:
            fp_strength.append((-1.0, 0, -1.0, -1.0, -1.0, 0, 1.0, i))
            continue

        s = pixel_preds[i][neg].astype(np.float32)  # 负样本分数向量

        # 如果你这里出现 mx>1 或 <0，说明 pixel_preds 不是 [0,1] 概率（常见于 z-score 输出）
        # 为了让 thr=0.99 有意义，这里把它 sigmoid 到 [0,1]
        if s.max() > 1.0 or s.min() < 0.0:
            s = 1.0 / (1.0 + np.exp(-s))

        mx = float(np.max(s))
        p99 = float(np.quantile(s, 0.99))
        p999 = float(np.quantile(s, q_dyn))

        # fixed thr 面积（你原来的逻辑仍保留，便于对比）
        area_thr = int((s >= thr).sum())

        # 动态阈值：每张图负样本 p99.9
        thr_dyn = p999
        area_dyn = int((s >= thr_dyn).sum())

        # top 1% 负样本均值（比 max 更稳定，专门抓阴影/反光这种"铺开"的 FP）
        thr_top = float(np.quantile(s, q_top))
        top_vals = s[s >= thr_top]
        mean_top1 = float(top_vals.mean()) if top_vals.size > 0 else float(mx)

        fp_strength.append((mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i))

    # 1) peak_fp：抓"尖峰误报"（max 高）
    # peak_fp: 按 (mx, area_thr) 排序
    peak = sorted(fp_strength, reverse=True, key=lambda x: (x[0], x[1]))
    top_peak = peak[:topk]

    print(f"\n[{dataset_name}] Top-{topk} FP samples (peak_fp, sorted by mx then area@thr={thr}):")
    for rank, (mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i) in enumerate(top_peak, 1):
        print(f"  #{rank:02d} mx={mx:.4f}  area@thr={area_thr}  p999={p999:.4f}  mean_top1={mean_top1:.4f}  area@p999={area_dyn}  file={file_names[i]}")

        # 保存 overlay（原图 + heatmap）
        img_path = os.path.join(DATA_PATH[dataset_name], file_names[i])
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        score = pixel_preds[i]
        heat = np.clip(score * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        # 只显示 road 区域（更直观看 road_only 的 FP）
        road = valid[i].astype(np.uint8) * 255
        road3 = cv2.merge([road, road, road])
        heat = cv2.bitwise_and(heat, road3)

        overlay = cv2.addWeighted(img_bgr, 0.65, heat, 0.35, 0.0)
        out_path = os.path.join(save_dir, f"fp_rank{rank:02d}_{os.path.basename(file_names[i])}.jpg")
        cv2.imwrite(out_path, overlay)

    # 2) spread_fp：抓"铺开误报"（大面积中高分，贴近 FPR95）
    # spread_fp: 按 (p999, mean_top1, area_dyn, mx) 排序 ——更贴"阴影/反光"
    spread = sorted(fp_strength, reverse=True, key=lambda x: (x[3], x[4], x[5], x[0]))
    top_spread = spread[:topk]

    print(f"\n[{dataset_name}] Top-{topk} FP samples (spread_fp, sorted by p99.9/mean_top1/area@p99.9):")
    for rank, (mx, area_thr, p99, p999, mean_top1, area_dyn, thr_dyn, i) in enumerate(top_spread, 1):
        print(f"  #{rank:02d} p999={p999:.4f}  mean_top1={mean_top1:.4f}  area@p999={area_dyn}  mx={mx:.4f}  thr_dyn={thr_dyn:.4f}  file={file_names[i]}")

        # 保存 overlay（原图 + heatmap）
        img_path = os.path.join(DATA_PATH[dataset_name], file_names[i])
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        score = pixel_preds[i]
        heat = np.clip(score * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        # 只显示 road 区域（更直观看 road_only 的 FP）
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


def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
    dataset_name: str = None,
    road_mask: np.ndarray = None,
    ignore_mask: np.ndarray = None,
    eval_mode: str = "road_or_anom",
    file_names: List[str] = None,
    save_path: str = None,
    disable_normalization: bool = False,
    score_mode: str = "prob",  # "prob" or "raw"
):
    """
    Evaluate metrics. For Road datasets, focus on pixel-level metrics only.
    
    Args:
        road_mask: Optional mask for road regions. If provided, pixel-level metrics
                   will be computed only on road pixels (road_mask == 0 means road region).
                   Shape: (N, H, W) or (N, 1, H, W), where 0 = road, 1 = non-road.
    """
    ROAD_DATASETS = {
        "Road", "RoadSynth", "RoadAnomaly", "RoadAnomaly21", 
        "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }
    is_road_dataset = dataset_name in ROAD_DATASETS if dataset_name else False
    
    # ================================================================================================
    # (2) Binarize labels: force to 0/1 (before any flatten)
    pixel_label = (pixel_label > 0).astype(np.uint8)
    
    # ================================================================================================
    # Debug: Print positive/negative mean scores (before any mask filtering)
    # This helps detect: 1) score direction inversion (pos_mean < neg_mean)
    #                    2) constant scores (pos_mean ≈ neg_mean)
    # ================================================================================================
    try:
        # Take first image/batch for debugging
        # Handle different shapes
        pixel_preds_flat = pixel_preds
        pixel_label_flat = pixel_label
        
        # Ensure both are 3D: (N, H, W)
        if pixel_preds.ndim == 4:
            if pixel_preds.shape[1] == 1:
                pixel_preds_flat = pixel_preds[:, 0, :, :]  # (N, H, W)
            else:
                pixel_preds_flat = pixel_preds.mean(axis=1)  # (N, H, W) - average over channels
        
        if pixel_label.ndim == 4:
            if pixel_label.shape[1] == 1:
                pixel_label_flat = pixel_label[:, 0, :, :]  # (N, H, W)
            else:
                pixel_label_flat = pixel_label[:, 0, :, :]  # Take first channel
        
        # Take first image
        preds_first = pixel_preds_flat[0]  # (H, W)
        mask_first = pixel_label_flat[0]   # (H, W)
        
        # Ensure shapes match - resize mask to match preds if needed
        if preds_first.shape != mask_first.shape:
            if preds_first.shape[0] > 0 and preds_first.shape[1] > 0:
                mask_first = cv2.resize(
                    mask_first.astype(np.uint8),
                    (preds_first.shape[1], preds_first.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)
            else:
                print(f"[POS/NEG MEAN] Skipped: invalid shape preds={preds_first.shape}, mask={mask_first.shape}")
                preds_first = None
        
        if preds_first is not None and preds_first.size > 0:
            # Calculate positive and negative means
            pos_mask = (mask_first == 1)
            neg_mask = (mask_first == 0)
            
            if pos_mask.any() and neg_mask.any():
                pos_mean = float(preds_first[pos_mask].mean())
                neg_mean = float(preds_first[neg_mask].mean())
                print(f"[POS/NEG MEAN] pos_mean={pos_mean:.6f}, neg_mean={neg_mean:.6f}, diff={pos_mean-neg_mean:.6f}")
                if pos_mean < neg_mean:
                    print(f"[WARNING] ⚠️  Score direction may be inverted! pos_mean ({pos_mean:.6f}) < neg_mean ({neg_mean:.6f})")
                if abs(pos_mean - neg_mean) < 1e-6:
                    print(f"[WARNING] ⚠️  Scores are almost constant! pos_mean ≈ neg_mean ≈ {pos_mean:.6f}")
            else:
                if not pos_mask.any():
                    print(f"[POS/NEG MEAN] No positive pixels found in first image")
                if not neg_mask.any():
                    print(f"[POS/NEG MEAN] No negative pixels found in first image")
    except Exception as e:
        print(f"[POS/NEG MEAN] Error calculating pos/neg means: {e}")
        import traceback
        traceback.print_exc()
    
    # Note: Per-image percentile normalization is now done AFTER filtering (see below)
    
    if image_preds.max() != 1:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min() + 1e-8
        )

    pmax_pred = pixel_preds.max(axis=(1, 2))
    if domain != "Medical":
        image_preds = pmax_pred * 0.5 + image_preds * 0.5
    else:
        image_preds = pmax_pred
    # ================================================================================================
    # pixel level auc & ap & fpr95
    # For Road datasets: optionally restrict to road regions only (to avoid "negative pixel ocean" problem)
    # ================================================================================================
    # Road ROI evaluation: keep road pixels OR anomaly pixels (never drop anomalies)
    TARGET_ROAD_DATASETS = {
        "RoadAnomaly", "RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full", "fs_static"
    }

    def _ensure_nhw(x, name: str):
        x = np.asarray(x)
        if x.ndim == 4 and x.shape[1] == 1:   # (N,1,H,W) -> (N,H,W)
            return x[:, 0]
        if x.ndim == 3:                       # (N,H,W)
            return x
        raise ValueError(f"Unexpected {name} shape: {x.shape}")

    def _direction_check(
        scores_1d: np.ndarray,
        labels_1d: np.ndarray,
        dataset_name: str,
        score_mode: str = "prob",  # "prob" | "raw" | "sigmoid"
        thr: float = 0.0,
    ):
        """
        Check whether score direction is inverted.

        For raw mode: compare auc(score) vs auc(-score), flip using -score
        For prob/sigmoid mode: compare auc(score) vs auc(1-score), flip using 1-score
        
        Args:
            score_mode: "raw" for raw scores (unbounded), "prob"/"sigmoid" for [0,1] scores
            thr: threshold for flip decision (default 0.0, no margin)
        """
        if dataset_name not in TARGET_ROAD_DATASETS:
            return scores_1d, False, None, None

        # need both classes
        if len(np.unique(labels_1d)) < 2:
            return scores_1d, False, None, None

        # ---- Compute AUC for original and flipped ----
        auc0 = roc_auc_score(labels_1d, scores_1d)
        if score_mode == "raw":
            # Raw mode: flip using -score
            flipped_alt = -scores_1d
            auc1 = roc_auc_score(labels_1d, flipped_alt)
            if auc1 > auc0:
                print(
                    f"[DBG_DIRECTION_CHECK] mode=raw  auc(score)={auc0:.6f}  auc(-score)={auc1:.6f}  thr={thr}"
                )
                print(
                    f"[DBG_DIRECTION_CHECK] Before flip: mean={scores_1d.mean():.6f} "
                    f"min/max={scores_1d.min():.6f}/{scores_1d.max():.6f}"
                )
                flipped_scores = flipped_alt
                print(
                    f"[DBG_DIRECTION_CHECK] After flip:  mean={flipped_scores.mean():.6f} "
                    f"min/max={flipped_scores.min():.6f}/{flipped_scores.max():.6f}"
                )
                return flipped_scores, True, auc0, auc1
            return scores_1d, False, auc0, auc1
        else:
            # prob/sigmoid mode: flip using 1-score
            flipped_alt = 1.0 - scores_1d
            auc1 = roc_auc_score(labels_1d, flipped_alt)
            if auc1 > auc0 + thr:
                print(
                    f"[DBG_DIRECTION_CHECK] mode={score_mode}  auc(score)={auc0:.6f}  auc(1-score)={auc1:.6f}  thr={thr}"
                )
                print(
                    f"[DBG_DIRECTION_CHECK] Before flip: mean={scores_1d.mean():.6f} "
                    f"min/max={scores_1d.min():.6f}/{scores_1d.max():.6f}"
                )
                flipped_scores = flipped_alt
                print(
                    f"[DBG_DIRECTION_CHECK] After flip:  mean={flipped_scores.mean():.6f} "
                    f"min/max={flipped_scores.min():.6f}/{flipped_scores.max():.6f}"
                )
                return flipped_scores, True, auc0, auc1
            return scores_1d, False, auc0, auc1

    if is_road_dataset and road_mask is not None:
        # --- normalize shapes to (N,H,W) to avoid broadcasting bugs
        pixel_label_2d = _ensure_nhw(pixel_label, "pixel_label")
        pixel_preds_2d = _ensure_nhw(pixel_preds, "pixel_preds")
        road_mask_2d   = _ensure_nhw(road_mask, "road_mask")

        # --- resize road_mask to match preds/labels if needed (keep your existing resize logic, but ensure final shape matches)
        if road_mask_2d.shape != pixel_preds_2d.shape:
            # Resize per-image with nearest neighbor
            resized = []
            for i in range(pixel_preds_2d.shape[0]):
                m = road_mask_2d[i % road_mask_2d.shape[0]]
                m = cv2.resize(
                    m.astype(np.uint8),
                    (pixel_preds_2d.shape[2], pixel_preds_2d.shape[1]),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized.append(m)
            road_mask_2d = np.stack(resized, axis=0)

        # --- strict sanity checks (fail fast instead of silent wrong results)
        if road_mask_2d.shape != pixel_label_2d.shape:
            raise ValueError(f"Shape mismatch: road_mask_2d={road_mask_2d.shape}, pixel_label_2d={pixel_label_2d.shape}")
        if pixel_preds_2d.shape != pixel_label_2d.shape:
            raise ValueError(f"Shape mismatch: pixel_preds_2d={pixel_preds_2d.shape}, pixel_label_2d={pixel_label_2d.shape}")

        # --- binarize label robustly (AA-CLIP is already 0/1 float, this just makes it bulletproof)
        pixel_label_bin = pixel_label_2d
        if pixel_label_bin.ndim == 4:
            pixel_label_bin = pixel_label_bin.squeeze(1)  # (N,H,W)

        # 1) base_valid: 先去掉 ignore
        base_valid = np.ones_like(pixel_label_bin, dtype=bool)
        if ignore_mask is not None:
            ign = ignore_mask
            if ign.ndim == 4:
                ign = ign.squeeze(1)  # (N,H,W)
            # Resize ignore_mask to match if needed
            if ign.shape != pixel_label_bin.shape:
                resized_ign = []
                for i in range(pixel_label_bin.shape[0]):
                    m = ign[i % ign.shape[0]]
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (pixel_label_bin.shape[2], pixel_label_bin.shape[1]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    resized_ign.append(m)
                ign = np.stack(resized_ign, axis=0)
            # ignore_mask 约定：1=ignore, 0=valid
            base_valid = (ign == 0)

        # 2) road_valid：根据 eval_mode 选择
        # road_mask: 0=road, 1=non-road
        # pixel_label_bin: 0/1
        if eval_mode == "road_only":
            # 实验A：严格 road-only（不保留 road 外的 anomaly）
            road_keep = (road_mask_2d == 0)
        else:
            # 实验B：road OR anomaly（默认，推荐）
            road_keep = (road_mask_2d == 0) | (pixel_label_bin == 1)

        # 3) 合并
        valid_mask = (base_valid & road_keep).reshape(-1)
        
        # ---- DEBUG: 打印 shapes 和 ratios ----
        print("[DEBUG] shapes:",
              "pixel_label_bin", pixel_label_bin.shape,
              "pixel_preds_2d", pixel_preds_2d.shape,
              "base_valid", base_valid.shape,
              "road_keep", road_keep.shape)
        
        print("[DEBUG] ratios:",
              "base_valid.mean", float(base_valid.mean()),
              "road_keep.mean", float(road_keep.mean()),
              "valid_mask.mean", float((base_valid & road_keep).mean()))
        
        # 关键：看 ignore_mask / road_mask 是否全 0 或全 1
        if ignore_mask is not None:
            print("[DEBUG] ignore unique:", np.unique(ign)[:10], " ...",
                  "ign mean", float((ign==1).mean()))
        print("[DEBUG] road_mask_2d unique:", np.unique(road_mask_2d)[:10], " ...",
              "road_mask_2d mean", float((road_mask_2d==1).mean()))
        print("[DEBUG] label unique:", np.unique(pixel_label_bin)[:10])
        
        # 统计 anom_on_road_ratio（anomaly 在 road 上的占比）
        anom_total = (pixel_label_bin == 1).sum()
        anom_on_road = ((pixel_label_bin == 1) & (road_mask_2d == 0)).sum()
        anom_on_road_ratio = (anom_on_road / anom_total) if anom_total > 0 else 0.0

        pixel_label_flat = pixel_label_bin.reshape(-1)
        pixel_preds_flat = pixel_preds_2d.reshape(-1)
        label_vec = pixel_label_flat[valid_mask]
        score_vec = pixel_preds_flat[valid_mask]

        # --- drop NaN/Inf scores if any (rare but worth guarding)
        finite = np.isfinite(score_vec)
        label_vec = label_vec[finite]
        score_vec = score_vec[finite]

        # --- Raw score mode: 不做任何normalization
        if score_mode == "raw":
            # Raw mode: 不做percentile normalization，不做sigmoid
            # score_vec已经是logit1 - logit0，直接使用
            pass
        else:
            # --- Per-image percentile normalization AFTER filtering (on score_vec)
            if not disable_normalization and len(score_vec) > 0:
                lo = np.percentile(score_vec, 1)
                hi = np.percentile(score_vec, 99)
                score_vec = np.clip(score_vec, lo, hi)
                score_vec = (score_vec - lo) / (hi - lo + 1e-8)

        # --- debug print (after filtering)
        pos_ratio = label_vec.mean() if len(label_vec) > 0 else 0.0
        ignore_ratio = 1.0 - base_valid.mean() if ignore_mask is not None else 0.0
        road_keep_ratio = road_keep.mean()
        total_pixels = pixel_label_bin.size
        valid_pixels = len(label_vec)
        anom_pixels_total = (pixel_label_bin == 1).sum()
        anom_pixels_kept = (label_vec == 1).sum()
        
        print(f"\n{'='*80}")
        print(f"[{dataset_name}] 评估统计信息 (eval_mode={eval_mode})")
        print(f"{'='*80}")
        print(f"总像素数:           {total_pixels:,}")
        print(f"评估像素数:         {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")
        print(f"丢弃像素数:         {total_pixels-valid_pixels:,} ({(total_pixels-valid_pixels)/total_pixels*100:.2f}%)")
        print(f"-" * 80)
        print(f"总异常像素数:       {anom_pixels_total:,}")
        print(f"保留异常像素数:     {anom_pixels_kept:,} ({anom_pixels_kept/anom_pixels_total*100:.2f}% of total anomalies)")
        print(f"丢弃异常像素数:     {anom_pixels_total-anom_pixels_kept:,}")
        print(f"-" * 80)
        print(f"pos_ratio (评估集中异常比例):    {pos_ratio:.6f}")
        if ignore_mask is not None:
            print(f"ignore_ratio (ignore区域比例):   {ignore_ratio:.6f}")
        print(f"road_keep_ratio (保留区域比例):  {road_keep_ratio:.6f}")
        print(f"anom_on_road_ratio (异常在道路上的比例): {anom_on_road_ratio:.6f}")
        print(f"{'='*80}\n")

        # --- direction check (for 5 Road datasets)
        score_vec, flipped, auc0, auc1 = _direction_check(score_vec, label_vec, dataset_name, score_mode=score_mode, thr=0.0)
        if dataset_name in TARGET_ROAD_DATASETS:
            if auc0 is not None:
                if score_mode == "raw":
                    print(f"[{dataset_name}] AUROC(score)={auc0:.6f} vs AUROC(-score)={auc1:.6f}  flipped={flipped}")
                else:
                    print(f"[{dataset_name}] AUROC(score)={auc0:.6f} vs AUROC(1-score)={auc1:.6f}  flipped={flipped}")
            else:
                print(f"[{dataset_name}] direction check skipped (need both classes)")

        # --- compute metrics (need both classes)
        if len(np.unique(label_vec)) < 2:
            zero_pixel_auc = np.nan
            zero_pixel_ap  = np.nan
            pixel_fpr95    = np.nan
        else:
            # ============================================================
            # Step 2 硬校验：根据 score_mode 选择正确的翻转方式
            # ============================================================
            if dataset_name in TARGET_ROAD_DATASETS:
                auc_score = roc_auc_score(label_vec, score_vec)
                ap_score = average_precision_score(label_vec, score_vec)
                
                if score_mode == "raw":
                    # raw 模式：翻转方向使用 -score
                    alt = -score_vec
                    auc_alt = roc_auc_score(label_vec, alt)
                    ap_alt = average_precision_score(label_vec, alt)
                    
                    print(f"\n{'='*80}")
                    print(f"[{dataset_name}] Step 2 硬校验：方向对比 (score_mode=raw)")
                    print(f"{'='*80}")
                    print(f"使用 score_vec:")
                    print(f"  AUROC = {auc_score:.6f}")
                    print(f"  AP    = {ap_score:.6f}")
                    print(f"\n使用 -score_vec:")
                    print(f"  AUROC = {auc_alt:.6f}")
                    print(f"  AP    = {ap_alt:.6f}")
                    print(f"\n差异:")
                    print(f"  ΔAUROC = {auc_alt - auc_score:+.6f}")
                    print(f"  ΔAP    = {ap_alt - ap_score:+.6f}")
                    
                    if auc_alt > auc_score:
                        print(f"\n⚠️  raw 方向错误！使用 -score_vec (ΔAUROC = {auc_alt - auc_score:+.6f} > 0)")
                        print(f"[DBG_FLIP] Before flip: score_vec mean={score_vec.mean():.6f} min/max={score_vec.min():.6f}/{score_vec.max():.6f}")
                        score_vec = -score_vec
                        print(f"[DBG_FLIP] After flip: score_vec mean={score_vec.mean():.6f} min/max={score_vec.min():.6f}/{score_vec.max():.6f}")
                        print(f"✅ 已应用翻转")
                    else:
                        print(f"\n✅ 方向正确，使用原始 score_vec (ΔAUROC = {auc_score - auc_alt:+.6f})")
                    print(f"{'='*80}\n")
                else:
                    # prob 模式：翻转方向使用 1-score
                    alt = 1.0 - score_vec
                    auc_alt = roc_auc_score(label_vec, alt)
                    ap_alt = average_precision_score(label_vec, alt)
                    
                    print(f"\n{'='*80}")
                    print(f"[{dataset_name}] Step 2 硬校验：方向对比 (score_mode=prob)")
                    print(f"{'='*80}")
                    print(f"使用 score_vec:")
                    print(f"  AUROC = {auc_score:.6f}")
                    print(f"  AP    = {ap_score:.6f}")
                    print(f"\n使用 1-score_vec:")
                    print(f"  AUROC = {auc_alt:.6f}")
                    print(f"  AP    = {ap_alt:.6f}")
                    print(f"\n差异:")
                    print(f"  ΔAUROC = {auc_alt - auc_score:+.6f}")
                    print(f"  ΔAP    = {ap_alt - ap_score:+.6f}")
                    
                    if auc_alt > auc_score:
                        print(f"\n⚠️  prob 方向错误！使用 1-score_vec (ΔAUROC = {auc_alt - auc_score:+.6f} > 0)")
                        print(f"[DBG_FLIP] Before flip: score_vec mean={score_vec.mean():.6f} min/max={score_vec.min():.6f}/{score_vec.max():.6f}")
                        score_vec = 1.0 - score_vec
                        print(f"[DBG_FLIP] After flip: score_vec mean={score_vec.mean():.6f} min/max={score_vec.min():.6f}/{score_vec.max():.6f}")
                        print(f"✅ 已应用翻转")
                    else:
                        print(f"\n✅ 方向正确，使用原始 score_vec (ΔAUROC = {auc_score - auc_alt:+.6f})")
                    print(f"{'='*80}\n")
            
            # ✅ 全数据集pos/neg统计打印（方向检查/flip之后、计算metrics之前）
            neg = score_vec[label_vec == 0]
            pos = score_vec[label_vec == 1]
            print(f"[{dataset_name}] n_pos={len(pos)} n_neg={len(neg)}")
            print(f"[{dataset_name}] score(pos/neg) mean: {pos.mean():.6f} / {neg.mean():.6f}")
            print(f"[{dataset_name}] score(pos/neg) p99 : {np.quantile(pos, 0.99):.6f} / {np.quantile(neg, 0.99):.6f}")
            print(f"[{dataset_name}] score_std={np.std(score_vec):.6f}\n")
            
            zero_pixel_auc = roc_auc_score(label_vec, score_vec)
            zero_pixel_ap  = average_precision_score(label_vec, score_vec)
            pixel_fpr95    = fpr_at_95_tpr(score_vec, label_vec)
            
            # ============================================================
            # Calculate and print neg_p99, neg_p999, mean_top1, and thr95
            # ============================================================
            neg = score_vec[label_vec == 0]
            if len(neg) > 0:
                neg_p99  = float(np.quantile(neg, 0.99))
                neg_p999 = float(np.quantile(neg, 0.999))
                k = max(1, int(len(neg) * 0.01))
                mean_top1 = float(np.mean(np.partition(neg, -k)[-k:]))
            else:
                neg_p99 = neg_p999 = mean_top1 = float("nan")
            
            # Calculate thr95 from ROC curve
            fpr, tpr, thr = roc_curve(label_vec, score_vec)
            idx = np.searchsorted(tpr, 0.95, side="left")
            idx = min(idx, len(thr)-1)
            thr95 = float(thr[idx])
            
            # Calculate score_vec std for raw mode
            score_std = float(np.std(score_vec)) if len(score_vec) > 0 else float("nan")
            print(f"[{dataset_name}] thr95={thr95:.4f}  neg_p99={neg_p99:.4f}  neg_p999={neg_p999:.4f}  mean_top1={mean_top1:.4f}  score_std={score_std:.6f}")
            
            # Calculate neg>=thr95 and pos>=thr95
            neg_above_thr95 = float((neg >= thr95).mean()) if len(neg) > 0 else float("nan")
            pos = score_vec[label_vec == 1]
            pos_above_thr95 = float((pos >= thr95).mean()) if len(pos) > 0 else float("nan")
            print(f"[{dataset_name}] neg>=thr95 = {neg_above_thr95:.4f}  pos>=thr95 = {pos_above_thr95:.4f}")
            
            # Calculate and print pos/neg quantiles
            if len(pos) > 0:
                pos_p01 = float(np.quantile(pos, 0.01))
                pos_p05 = float(np.quantile(pos, 0.05))
                pos_p10 = float(np.quantile(pos, 0.10))
                pos_p50 = float(np.quantile(pos, 0.50))
                print(f"[{dataset_name}] POS quantile: p01={pos_p01:.4f} p05={pos_p05:.4f} p10={pos_p10:.4f} p50={pos_p50:.4f}")
            if len(neg) > 0:
                neg_p99 = float(np.quantile(neg, 0.99))
                neg_p999 = float(np.quantile(neg, 0.999))
                print(f"[{dataset_name}] NEG quantile: p99={neg_p99:.4f} p999={neg_p999:.4f}")
            
            # ✅ 打印neg/pos的score_diff分位数（确认方向和负样本是否整体>0）
            if len(neg) > 0:
                neg_mean = float(np.mean(neg))
                neg_p99 = float(np.quantile(neg, 0.99))
                neg_p999 = float(np.quantile(neg, 0.999))
            else:
                neg_mean = neg_p99 = neg_p999 = float("nan")
            if len(pos) > 0:
                pos_mean = float(np.mean(pos))
                pos_p01 = float(np.quantile(pos, 0.01))
                pos_p05 = float(np.quantile(pos, 0.05))
                pos_p99 = float(np.quantile(pos, 0.99))
                pos_p999 = float(np.quantile(pos, 0.999))
            else:
                pos_mean = pos_p01 = pos_p05 = pos_p99 = pos_p999 = float("nan")
            print(f"[{dataset_name}] score_diff分位数: neg_mean={neg_mean:.6f} neg_p99={neg_p99:.6f} neg_p999={neg_p999:.6f}")
            print(f"[{dataset_name}] score_diff分位数: pos_mean={pos_mean:.6f} pos_p01={pos_p01:.6f} pos_p05={pos_p05:.6f} pos_p99={pos_p99:.6f} pos_p999={pos_p999:.6f}")
            
            # ============================================================
            # Dump top FP examples (before Step 2)
            # ============================================================
            if dataset_name in TARGET_ROAD_DATASETS and file_names is not None and save_path is not None:
                dump_top_fp_examples(
                    dataset_name=dataset_name,
                    file_names=file_names,
                    pixel_preds=pixel_preds_2d,
                    pixel_label=pixel_label_bin,
                    road_mask_2d=road_mask_2d,
                    eval_mode=eval_mode,
                    save_dir=os.path.join(save_path, "top_fp", dataset_name),
                    topk=20,
                    thr=0.99
                )
            
            # ============================================================
            # Step 2: 分位数统计和 Top-K Precision
            # ============================================================
            if dataset_name in TARGET_ROAD_DATASETS:
                print(f"\n{'='*80}")
                print(f"[{dataset_name}] Step 2: 分位数统计和 Top-K Precision 分析")
                print(f"{'='*80}")
                
                # 1. 正常像素（label=0）的高分位数
                normal_scores = score_vec[label_vec == 0]
                if len(normal_scores) > 0:
                    normal_percentiles = np.percentile(normal_scores, [90, 95, 99, 99.5, 99.9])
                    print(f"\n正常像素 (label=0) 高分位数:")
                    print(f"  P90:   {normal_percentiles[0]:.6f}")
                    print(f"  P95:   {normal_percentiles[1]:.6f}")
                    print(f"  P99:   {normal_percentiles[2]:.6f}")
                    print(f"  P99.5: {normal_percentiles[3]:.6f}")
                    print(f"  P99.9: {normal_percentiles[4]:.6f}")
                
                # 2. 异常像素（label=1）的分位数
                anom_scores = score_vec[label_vec == 1]
                if len(anom_scores) > 0:
                    anom_percentiles = np.percentile(anom_scores, [10, 50, 90])
                    print(f"\n异常像素 (label=1) 分位数:")
                    print(f"  P10:  {anom_percentiles[0]:.6f}")
                    print(f"  P50:  {anom_percentiles[1]:.6f}")
                    print(f"  P90:  {anom_percentiles[2]:.6f}")
                
                # 3. Top-K Precision（K = 正样本数量的 1x, 2x, 5x）
                n_pos = int((label_vec == 1).sum())
                if n_pos > 0:
                    print(f"\nTop-K Precision 分析 (正样本总数 = {n_pos}):")
                    
                    # 按分数降序排序
                    sorted_indices = np.argsort(score_vec)[::-1]
                    sorted_labels = label_vec[sorted_indices]
                    
                    for multiplier in [1, 2, 5]:
                        k = min(n_pos * multiplier, len(score_vec))
                        if k > 0:
                            top_k_labels = sorted_labels[:k]
                            precision = (top_k_labels == 1).sum() / k
                            recall = (top_k_labels == 1).sum() / n_pos
                            print(f"  Top-{k:6d} (K={multiplier}x): Precision={precision:.4f} ({precision*100:.2f}%), Recall={recall:.4f} ({recall*100:.2f}%)")
                
                # 4. 分数分布统计
                print(f"\n分数分布统计:")
                print(f"  正常像素: min={normal_scores.min():.6f}, max={normal_scores.max():.6f}, mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
                if len(anom_scores) > 0:
                    print(f"  异常像素: min={anom_scores.min():.6f}, max={anom_scores.max():.6f}, mean={anom_scores.mean():.6f}, std={anom_scores.std():.6f}")
                
                print(f"{'='*80}\n")

    else:
        # original full-image logic (no road_mask)
        pixel_label_flat = pixel_label.flatten()
        pixel_preds_flat = pixel_preds.flatten()
        
        label_vec = pixel_label_flat
        score_vec = pixel_preds_flat
        
        # --- Raw score mode: 不做任何normalization
        if score_mode == "raw":
            # Raw mode: 不做percentile normalization，不做sigmoid
            # score_vec已经是logit1 - logit0，直接使用
            pass
        else:
            # --- Per-image percentile normalization
            if not disable_normalization and len(score_vec) > 0:
                lo = np.percentile(score_vec, 1)
                hi = np.percentile(score_vec, 99)
                score_vec = np.clip(score_vec, lo, hi)
                score_vec = (score_vec - lo) / (hi - lo + 1e-8)
        
        # --- direction check (for 5 Road datasets)
        if dataset_name in TARGET_ROAD_DATASETS:
            if len(np.unique(label_vec)) >= 2:
                auc0 = roc_auc_score(label_vec, score_vec)
                if score_mode == "raw":
                    # raw 模式：翻转方向使用 -score
                    alt = -score_vec
                    auc1 = roc_auc_score(label_vec, alt)
                    flipped = (auc1 > auc0 + 0.05)
                    if flipped:
                        score_vec = -score_vec
                    print(f"[{dataset_name}] raw flip check: AUROC(score)={auc0:.6f} vs AUROC(-score)={auc1:.6f}  flipped={flipped}")
                else:
                    # prob 模式：翻转方向使用 1-score
                    alt = 1.0 - score_vec
                    auc1 = roc_auc_score(label_vec, alt)
                    flipped = (auc1 > auc0 + 0.05)
                    if flipped:
                        score_vec = 1.0 - score_vec
                    print(f"[{dataset_name}] prob flip check: AUROC(score)={auc0:.6f} vs AUROC(1-score)={auc1:.6f}  flipped={flipped}")
            else:
                print(f"[{dataset_name}] direction check skipped (need both classes)")
        
        # ✅ 全数据集pos/neg统计打印（方向检查/flip之后、计算metrics之前）
        neg = score_vec[label_vec == 0]
        pos = score_vec[label_vec == 1]
        print(f"[{dataset_name}] n_pos={len(pos)} n_neg={len(neg)}")
        print(f"[{dataset_name}] score(pos/neg) mean: {pos.mean():.6f} / {neg.mean():.6f}")
        print(f"[{dataset_name}] score(pos/neg) p99 : {np.quantile(pos, 0.99):.6f} / {np.quantile(neg, 0.99):.6f}")
        print(f"[{dataset_name}] score_std={np.std(score_vec):.6f}\n")
        
        zero_pixel_auc = roc_auc_score(label_vec, score_vec)
        zero_pixel_ap  = average_precision_score(label_vec, score_vec)
        pixel_fpr95    = fpr_at_95_tpr(score_vec, label_vec)
        
        # ============================================================
        # Calculate and print neg_p99, neg_p999, mean_top1, and thr95
        # ============================================================
        neg = score_vec[label_vec == 0]
        if len(neg) > 0:
            neg_p99  = float(np.quantile(neg, 0.99))
            neg_p999 = float(np.quantile(neg, 0.999))
            k = max(1, int(len(neg) * 0.01))
            mean_top1 = float(np.mean(np.partition(neg, -k)[-k:]))
        else:
            neg_p99 = neg_p999 = mean_top1 = float("nan")
        
        # Calculate thr95 from ROC curve
        fpr, tpr, thr = roc_curve(label_vec, score_vec)
        idx = np.searchsorted(tpr, 0.95, side="left")
        idx = min(idx, len(thr)-1)
        thr95 = float(thr[idx])
        
        # Calculate score_vec std for raw mode
        score_std = float(np.std(score_vec)) if len(score_vec) > 0 else float("nan")
        print(f"[{dataset_name}] thr95={thr95:.4f}  neg_p99={neg_p99:.4f}  neg_p999={neg_p999:.4f}  mean_top1={mean_top1:.4f}  score_std={score_std:.6f}")
        
        # Calculate neg>=thr95 and pos>=thr95
        neg_above_thr95 = float((neg >= thr95).mean()) if len(neg) > 0 else float("nan")
        pos = score_vec[label_vec == 1]
        pos_above_thr95 = float((pos >= thr95).mean()) if len(pos) > 0 else float("nan")
        print(f"[{dataset_name}] neg>=thr95 = {neg_above_thr95:.4f}  pos>=thr95 = {pos_above_thr95:.4f}")
        
        # Calculate and print pos/neg quantiles
        if len(pos) > 0:
            pos_p01 = float(np.quantile(pos, 0.01))
            pos_p05 = float(np.quantile(pos, 0.05))
            pos_p10 = float(np.quantile(pos, 0.10))
            pos_p50 = float(np.quantile(pos, 0.50))
            print(f"[{dataset_name}] POS quantile: p01={pos_p01:.4f} p05={pos_p05:.4f} p10={pos_p10:.4f} p50={pos_p50:.4f}")
        if len(neg) > 0:
            neg_p99 = float(np.quantile(neg, 0.99))
            neg_p999 = float(np.quantile(neg, 0.999))
            print(f"[{dataset_name}] NEG quantile: p99={neg_p99:.4f} p999={neg_p999:.4f}")
        
        # ============================================================
        # Dump top FP examples (before Step 2, for original path)
        # ============================================================
        if dataset_name in TARGET_ROAD_DATASETS and file_names is not None and save_path is not None:
            # For original path, we need to reshape back to (N, H, W)
            pixel_preds_2d = pixel_preds.reshape(-1, pixel_preds.shape[-2], pixel_preds.shape[-1]) if pixel_preds.ndim == 3 else pixel_preds
            pixel_label_2d = pixel_label.reshape(-1, pixel_label.shape[-2], pixel_label.shape[-1]) if pixel_label.ndim == 3 else pixel_label
            road_mask_2d = road_mask if road_mask is not None else np.zeros_like(pixel_label_2d)
            dump_top_fp_examples(
                dataset_name=dataset_name,
                file_names=file_names,
                pixel_preds=pixel_preds_2d,
                pixel_label=pixel_label_2d,
                road_mask_2d=road_mask_2d,
                eval_mode=eval_mode,
                save_dir=os.path.join(save_path, "top_fp", dataset_name),
                topk=20,
                thr=0.99
            )
        
        # ============================================================
        # Step 2: 分位数统计和 Top-K Precision (for original path)
        # ============================================================
        if dataset_name in TARGET_ROAD_DATASETS and len(np.unique(label_vec)) >= 2:
            print(f"\n{'='*80}")
            print(f"[{dataset_name}] Step 2: 分位数统计和 Top-K Precision 分析")
            print(f"{'='*80}")
            
            # 1. 正常像素（label=0）的高分位数
            normal_scores = score_vec[label_vec == 0]
            if len(normal_scores) > 0:
                normal_percentiles = np.percentile(normal_scores, [90, 95, 99, 99.5, 99.9])
                print(f"\n正常像素 (label=0) 高分位数:")
                print(f"  P90:   {normal_percentiles[0]:.6f}")
                print(f"  P95:   {normal_percentiles[1]:.6f}")
                print(f"  P99:   {normal_percentiles[2]:.6f}")
                print(f"  P99.5: {normal_percentiles[3]:.6f}")
                print(f"  P99.9: {normal_percentiles[4]:.6f}")
            
            # 2. 异常像素（label=1）的分位数
            anom_scores = score_vec[label_vec == 1]
            if len(anom_scores) > 0:
                anom_percentiles = np.percentile(anom_scores, [10, 50, 90])
                print(f"\n异常像素 (label=1) 分位数:")
                print(f"  P10:  {anom_percentiles[0]:.6f}")
                print(f"  P50:  {anom_percentiles[1]:.6f}")
                print(f"  P90:  {anom_percentiles[2]:.6f}")
            
            # 3. Top-K Precision（K = 正样本数量的 1x, 2x, 5x）
            n_pos = int((label_vec == 1).sum())
            if n_pos > 0:
                print(f"\nTop-K Precision 分析 (正样本总数 = {n_pos}):")
                
                # 按分数降序排序
                sorted_indices = np.argsort(score_vec)[::-1]
                sorted_labels = label_vec[sorted_indices]
                
                for multiplier in [1, 2, 5]:
                    k = min(n_pos * multiplier, len(score_vec))
                    if k > 0:
                        top_k_labels = sorted_labels[:k]
                        precision = (top_k_labels == 1).sum() / k
                        recall = (top_k_labels == 1).sum() / n_pos
                        print(f"  Top-{k:6d} (K={multiplier}x): Precision={precision:.4f} ({precision*100:.2f}%), Recall={recall:.4f} ({recall*100:.2f}%)")
            
            # 4. 分数分布统计
            print(f"\n分数分布统计:")
            print(f"  正常像素: min={normal_scores.min():.6f}, max={normal_scores.max():.6f}, mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
            if len(anom_scores) > 0:
                print(f"  异常像素: min={anom_scores.min():.6f}, max={anom_scores.max():.6f}, mean={anom_scores.mean():.6f}, std={anom_scores.std():.6f}")
            
            print(f"{'='*80}\n")
    
    # ================================================================================================
    # image level auc & ap (skip for Road datasets or if only one class exists)
    if is_road_dataset:
        # For Road datasets: pixel-level only, skip image-level metrics
        agg_image_auc = np.nan
        agg_image_ap = np.nan
    else:
        # Original logic for other datasets
        # Image level metrics only if both classes exist
        if len(np.unique(image_label)) < 2:
            agg_image_auc = np.nan
            agg_image_ap = np.nan
        else:
            image_label_flat = image_label.flatten()
            agg_image_preds = image_preds.flatten()
            agg_image_auc = roc_auc_score(image_label_flat, agg_image_preds)
            agg_image_ap = average_precision_score(image_label_flat, agg_image_preds)
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc, 4) * 100,
        "pixel AP": round(zero_pixel_ap, 4) * 100,
        "pixel FPR95": round(pixel_fpr95 * 100, 4) if not np.isnan(pixel_fpr95) else np.nan,
    }
    
    if not is_road_dataset:
        # Add image-level metrics for non-Road datasets
        result["image AUC"] = round(agg_image_auc, 4) * 100 if not np.isnan(agg_image_auc) else np.nan
        result["image AP"] = round(agg_image_ap, 4) * 100 if not np.isnan(agg_image_ap) else np.nan
    else:
        # For Road datasets: explicitly set image metrics to NaN
        result["image AUC"] = np.nan
        result["image AP"] = np.nan
    
    return result


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
    # ===============================================================================================
    # save path
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
