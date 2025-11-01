"""Visualise segmentation labels overlaid on a prostate MRI slice."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img", type=str, required=True, help="Path to MRI NIfTI image.")
    parser.add_argument("--seg", type=str, required=True, help="Path to segmentation NIfTI.")
    parser.add_argument("--out", type=str, default="preview_labels.png", help="Output PNG path.")
    return parser.parse_args()


def _center_slice(array: np.ndarray) -> np.ndarray:
    """Return a 2D slice from a 2D/3D array."""
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[:, :, array.shape[2] // 2]
    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def load_slice(path: Path) -> np.ndarray:
    """Load a NIfTI file and return a 2D slice."""
    data = nib.load(str(path)).get_fdata()
    if data.size == 0:
        raise ValueError(f"NIfTI file is empty: {path}")
    slice_2d = _center_slice(np.asarray(data))
    return slice_2d.astype(np.float32)


def main() -> None:
    """Run the preview utility."""
    args = parse_args()
    img_path = Path(args.img).expanduser()
    seg_path = Path(args.seg).expanduser()
    out_path = Path(args.out).expanduser()

    image = load_slice(img_path)
    segmentation = load_slice(seg_path)

    if image.shape != segmentation.shape:
        raise ValueError(
            f"Image and segmentation shapes differ: {image.shape} vs {segmentation.shape}"
        )

    labels = np.unique(segmentation.astype(int))
    labels = labels[labels != 0]

    cmap = plt.cm.get_cmap("tab10", 6)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.axis("off")

    legend_handles = []
    for label in labels:
        mask = segmentation == label
        if not np.any(mask):
            continue
        color = cmap(label % cmap.N)
        overlay_cmap = ListedColormap(((0, 0, 0, 0), color))
        ax.imshow(mask.astype(float), alpha=0.4, cmap=overlay_cmap)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=str(label),
                markerfacecolor=color,
                markersize=10,
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, title="Labels", loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved label preview to {out_path}")


if __name__ == "__main__":
    main()
