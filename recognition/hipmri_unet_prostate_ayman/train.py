"""Training script for the dilated U-Net prostate MRI segmentation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from recognition.hipmri_unet_prostate_ayman.dataset import HipMRI2DDataset
from recognition.hipmri_unet_prostate_ayman.modules import ImprovedUNet2D
from recognition.hipmri_unet_prostate_ayman import utils


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace containing CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_root", type=str, required=True, help="Training split root.")
    parser.add_argument("--val_root", type=str, required=True, help="Validation split root.")
    parser.add_argument("--test_root", type=str, required=True, help="Test split root.")
    parser.add_argument("--out", type=str, default="outputs/hipmri_unet", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--size", type=int, default=256, help="Input resize dimension.")
    parser.add_argument("--base", type=int, default=32, help="Base channel width for the U-Net.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--label", type=int, default=5, help="Foreground label id (e.g., prostate=5).")
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Override positive class weight for BCEWithLogitsLoss. If None, compute from train data.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.35,0.4,0.45,0.5",
        help="Comma-separated thresholds to sweep for Dice during eval.",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    """Select computation device with priority MPS > CUDA > CPU.

    Returns:
        Torch device to place tensors on.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(args: argparse.Namespace, pin_memory: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train/validation/test dataloaders.

    Args:
        args: Command-line arguments with dataset roots and loader params.
        pin_memory: Whether to enable pinned memory (CUDA optimised).

    Returns:
        Train, validation, and test dataloaders.

    Raises:
        ValueError: If any dataset split is empty.
    """
    train_ds = HipMRI2DDataset(
        args.train_root,
        size=args.size,
        aug_flip=True,
        aug_rotate=True,
        fg_label=args.label,
    )
    val_ds = HipMRI2DDataset(
        args.val_root,
        size=args.size,
        aug_flip=False,
        aug_rotate=False,
        fg_label=args.label,
    )
    test_ds = HipMRI2DDataset(
        args.test_root,
        size=args.size,
        aug_flip=False,
        aug_rotate=False,
        fg_label=args.label,
    )

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty.")
    if len(test_ds) == 0:
        raise ValueError("Test dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def compute_pos_weight(train_loader: DataLoader, device: torch.device, max_batches: int = 200) -> float:
    """Estimate pos_weight ≈ mean(neg/pos) over a few batches; clip to [5,30]."""
    import math  # Local import to avoid unused dependency when not computing.

    neg_over_pos: List[float] = []
    seen = 0
    for _, targets, _ in train_loader:
        targets = targets.to(device=device, non_blocking=True)
        positives = torch.count_nonzero(targets).item()
        negatives = targets.numel() - positives
        if positives > 0:
            neg_over_pos.append(negatives / positives)
        seen += 1
        if seen >= max_batches:
            break
    if not neg_over_pos:
        return 10.0
    val = float(np.mean(neg_over_pos))
    if not math.isfinite(val):
        return 10.0
    return float(max(5.0, min(30.0, val)))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_bce: nn.Module,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model: Model to optimise.
        loader: Training dataloader.
        optimizer: Optimiser instance.
        criterion_bce: BCE loss module.
        device: Target device.

    Returns:
        Average training loss for the epoch.

    Raises:
        ValueError: If no samples were processed.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for inputs, targets, _ in progress:
        if inputs.size(0) == 0:
            continue
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)

        bce_loss = criterion_bce(logits, targets)
        dice_loss = utils.soft_dice_loss(logits, targets)
        loss = bce_loss + dice_loss

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("No samples processed during training epoch.")
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_bce: nn.Module,
    device: torch.device,
    thresholds: List[float],
) -> Tuple[float, float, float]:
    """Evaluate the model on a validation or test loader.

    Args:
        model: Model to evaluate.
        loader: Evaluation dataloader.
        criterion_bce: BCE loss module (shared with training).
        device: Target device.
        thresholds: Probability thresholds to sweep for Dice scores.

    Returns:
        Tuple containing average loss, best Dice score, and the corresponding threshold.

    Raises:
        ValueError: If no samples were processed.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    dice_totals: Dict[float, float] = {thresh: 0.0 for thresh in thresholds}

    with torch.no_grad():
        progress = tqdm(loader, desc="Eval", leave=False)
        for inputs, targets, _ in progress:
            if inputs.size(0) == 0:
                continue

            inputs = inputs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            logits = model(inputs)

            bce_loss = criterion_bce(logits, targets)
            dice_loss = utils.soft_dice_loss(logits, targets)
            loss = bce_loss + dice_loss

            batch_size = inputs.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            for thresh in thresholds:
                batch_dice = utils.dice_metric(logits, targets, threshold=thresh)
                dice_totals[thresh] += batch_dice * batch_size

    if total_samples == 0:
        raise ValueError("No samples processed during evaluation.")

    avg_loss = total_loss / total_samples
    best_thresh = thresholds[0]
    best_dice = -float("inf")
    for thresh in thresholds:
        avg_dice = dice_totals[thresh] / total_samples if total_samples else 0.0
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_thresh = thresh
    return avg_loss, best_dice, best_thresh


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    val_dice: float,
    best_thresh: float,
    device: torch.device,
) -> None:
    """Persist the best model checkpoint.

    Args:
        model: Trained model instance.
        path: Destination to store the checkpoint.
        epoch: Epoch number when the snapshot was saved.
        val_dice: Validation Dice score associated with the checkpoint.
        device: Device identifier string for reference.
        best_thresh: Threshold that yielded the best validation Dice.
    """
    checkpoint = {
        "epoch": epoch,
        "val_dice": val_dice,
        "best_thresh": best_thresh,
        "model_state": model.state_dict(),
        "device": str(device),
    }
    torch.save(checkpoint, path)


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    utils.set_global_seed(args.seed)
    device = select_device()
    pin_memory = device.type == "cuda"

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        raise ValueError("No thresholds provided for evaluation.")
    thresholds = sorted(set(thresholds))

    train_loader, val_loader, test_loader = build_dataloaders(args, pin_memory=pin_memory)

    model = ImprovedUNet2D(in_channels=1, n_classes=1, base_channels=args.base)
    model.to(device=device)

    if args.pos_weight is None:
        estimation_loader = DataLoader(
            train_loader.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        pos_weight_value = compute_pos_weight(estimation_loader, device=device)
    else:
        pos_weight_value = float(args.pos_weight)
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    log: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_dice = -float("inf")
    best_val_thresh = thresholds[0]
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_desc = f"Epoch {epoch}/{args.epochs}"
        print(epoch_desc)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_bce, device)
        val_loss, val_dice, val_thresh = evaluate(model, val_loader, criterion_bce, device, thresholds)
        scheduler.step()

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["val_dice"].append(val_dice)

        print(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} @ thresh={val_thresh:.2f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_thresh = val_thresh
            save_checkpoint(model, best_path, epoch, val_dice, val_thresh, device)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_val_dice = max(best_val_dice, float(checkpoint.get("val_dice", best_val_dice)))
        best_val_thresh = float(checkpoint.get("best_thresh", best_val_thresh))

    _, test_dice, test_thresh = evaluate(model, test_loader, criterion_bce, device, thresholds)
    print(f"Test Dice (best over sweep): {test_dice:.4f} at thresh={test_thresh:.2f}")

    utils.save_curves(log, output_dir.as_posix())

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)

    metrics = {
        "best_val_dice": best_val_dice,
        "best_thresh": best_val_thresh,
        "test_dice": test_dice,
        "test_thresh": test_thresh,
        "pos_weight": pos_weight_value,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


if __name__ == "__main__":
    main()
