"""Inference script to generate prediction overlays for prostate MRI scans."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from skimage import transform

from recognition.hipmri_unet_prostate_ayman.modules import ImprovedUNet2D
from recognition.hipmri_unet_prostate_ayman.utils import largest_component


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (best.pt).")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Input NIfTI images.")
    parser.add_argument("--out", type=str, default="outputs/preds", help="Directory to store PNGs.")
    parser.add_argument("--size", type=int, default=256, help="Spatial size for resizing.")
    parser.add_argument("--base", type=int, default=32, help="Base channel width of the model.")
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Binarization threshold. If None, use checkpoint best_thresh or 0.5.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable light TTA (hflip).",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    """Select device with priority MPS > CUDA > CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _center_slice2d(array: np.ndarray) -> np.ndarray:
    """Extract central slice if 3D, otherwise return 2D array."""
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[:, :, array.shape[2] // 2]
    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def _load_and_preprocess(path: Path, size: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a NIfTI image, preprocess, and return image/mask-ready arrays."""
    nifti = nib.load(str(path))
    data = nifti.get_fdata()
    if data.size == 0:
        raise ValueError(f"NIfTI file is empty: {path}")

    slice_2d = _center_slice2d(np.asarray(data))
    slice_2d = slice_2d.astype(np.float32)

    mean = float(slice_2d.mean())
    std = float(slice_2d.std())
    if std > 0:
        slice_2d = (slice_2d - mean) / std
    else:
        slice_2d = np.zeros_like(slice_2d, dtype=np.float32)

    resized = transform.resize(
        slice_2d,
        output_shape=(size, size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    tensor = torch.from_numpy(resized[np.newaxis, np.newaxis, ...])
    stem = path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    return resized, tensor, stem


def _predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    thresh: float,
    tta: bool = False,
) -> np.ndarray:
    """Run inference with optional TTA and return refined binary mask."""
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device=device)

        logits0 = model(tensor)
        probs0 = torch.sigmoid(logits0)

        if tta:
            tensor_flip = torch.flip(tensor, dims=[-1])
            logits1 = model(tensor_flip)
            probs1 = torch.sigmoid(logits1)
            probs1 = torch.flip(probs1, dims=[-1])
            probs = 0.5 * (probs0 + probs1)
        else:
            probs = probs0

        mask = (probs >= thresh).float()

    mask_np = mask.squeeze().cpu().numpy()
    refined = largest_component(mask_np)
    return refined


def _plot_prediction(image: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Create a three-panel visualisation and save as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].contour(mask, colors="red", linewidths=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run inference for provided inputs."""
    args = parse_args()
    device = select_device()

    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ImprovedUNet2D(in_channels=1, n_classes=1, base_channels=args.base)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_thresh = checkpoint.get("best_thresh", 0.5)
    thresh = args.thresh if args.thresh is not None else float(best_thresh)
    model.to(device=device)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        image, tensor, stem = _load_and_preprocess(input_path, args.size)
        tensor = tensor.to(device=device)
        mask = _predict_mask(model, tensor, device, thresh, tta=args.tta)

        out_path = output_dir / f"{stem}_pred.png"
        _plot_prediction(image, mask, out_path)
        print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()
