"""Utility functions for training and evaluating prostate MRI segmentation models."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from scipy import ndimage as ndi

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_SCIPY = False
from skimage.measure import label as sk_label


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across supported libraries.

    This configures Python's ``random`` module, NumPy, and PyTorch (CPU and CUDA).
    ``torch.backends.cudnn.deterministic`` is intentionally left ``False`` to avoid
    significant performance penalties or unsupported backends (e.g. MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDNN deterministic mode stays False to retain backend flexibility.


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss between logits and target masks.

    Args:
        logits: Raw model outputs of shape ``(N, 1, H, W)``.
        target: Binary target masks of shape ``(N, 1, H, W)``.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor representing ``1 - Dice``; zero when both inputs are empty.
    """
    probs = torch.sigmoid(logits)
    target = target.to(dtype=probs.dtype)

    probs_flat = probs.view(probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    numerator = 2.0 * torch.sum(probs_flat * target_flat, dim=1)
    denominator = torch.sum(probs_flat + target_flat, dim=1) + eps

    dice = numerator / denominator

    empty_pred = torch.sum(probs_flat, dim=1) <= eps
    empty_target = torch.sum(target_flat, dim=1) <= eps
    empty_mask = empty_pred & empty_target

    if torch.any(empty_mask):
        dice = torch.where(empty_mask, torch.ones_like(dice), dice)

    loss = 1.0 - dice
    return loss.mean()


def dice_metric(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute Dice coefficient between logits and target masks.

    Args:
        logits: Raw model outputs of shape ``(N, 1, H, W)``.
        target: Binary target masks of shape ``(N, 1, H, W)``.
        threshold: Probability threshold applied after sigmoid.
        eps: Small constant for numerical stability.

    Returns:
        Dice coefficient in ``[0, 1]``; equals 1.0 when both inputs are empty.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    preds_flat = preds.view(preds.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = torch.sum(preds_flat * target_flat, dim=1)
    union = torch.sum(preds_flat + target_flat, dim=1) + eps

    dice = (2.0 * intersection) / union

    empty_pred = torch.sum(preds_flat, dim=1) <= eps
    empty_target = torch.sum(target_flat, dim=1) <= eps
    empty_mask = empty_pred & empty_target

    dice = torch.where(empty_mask, torch.ones_like(dice), dice)
    return float(dice.mean().item())


def save_curves(log: Dict[str, List[float]], outdir: str) -> None:
    """Persist training curves as PNG figures.

    Args:
        log: Mapping with keys ``train_loss``, ``val_loss``, and ``val_dice``.
        outdir: Output directory for the images.
    """
    output_path = Path(outdir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_loss = log.get("train_loss", [])
    val_loss = log.get("val_loss", [])
    val_dice = log.get("val_dice", [])

    if train_loss or val_loss:
        plt.figure(figsize=(6, 4))
        if train_loss:
            plt.plot(train_loss, label="Train Loss")
        if val_loss:
            plt.plot(val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path / "loss.png", dpi=150)
        plt.close()

    if val_dice:
        plt.figure(figsize=(6, 4))
        plt.plot(val_dice, label="Val Dice", color="tab:green")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Validation Dice")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path / "dice.png", dpi=150)
        plt.close()


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only largest 8-connected component of a 2D binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"largest_component expects 2D, got {mask.shape}")
    mask_bin = (mask > 0.5).astype(np.uint8)
    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin, dtype=np.float32)
    if _HAVE_SCIPY:
        labeled, n = ndi.label(mask_bin, structure=np.ones((3, 3), dtype=np.uint8))
    else:
        labeled = sk_label(mask_bin, connectivity=2)
        n = int(labeled.max())
    if n <= 1:
        return mask_bin.astype(np.float32)
    vals, counts = np.unique(labeled, return_counts=True)
    mask_vals = vals != 0
    vals = vals[mask_vals]
    counts = counts[mask_vals]
    keep = vals[np.argmax(counts)]
    out = (labeled == keep).astype(np.float32)
    return out


def apply_lcc_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid->threshold->LCC per sample. logits: (N,1,H,W). Returns np masks (N,H,W) in {0,1}."""
    if logits.ndim != 4 or logits.size(1) != 1:
        raise ValueError(f"Expected logits (N,1,H,W), got {tuple(logits.shape)}")
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    masks = (probs >= threshold).astype(np.float32)
    masks = masks[:, 0]
    out = [largest_component(mask) for mask in masks]
    return np.stack(out, axis=0)


__all__ = [
    "set_global_seed",
    "soft_dice_loss",
    "dice_metric",
    "save_curves",
    "largest_component",
    "apply_lcc_from_logits",
]
