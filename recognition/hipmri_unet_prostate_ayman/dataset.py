"""Dataset utilities for loading 2D prostate MRI slices with preprocessing."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from skimage import transform
from torch.utils.data import Dataset

KNOWN_PREFIXES: Sequence[str] = ("img_", "image_", "seg_", "label_")
KNOWN_SUFFIXES: Sequence[str] = ("_img", "_image", "_seg", "_label")


def _is_nifti_file(path: Path) -> bool:
    """Return True if the path points to a NIfTI file."""
    if not path.is_file():
        return False
    suffixes = path.suffixes
    return suffixes == [".nii"] or suffixes == [".nii", ".gz"]


def _file_stem(path: Path) -> str:
    """Return the filename stem without the .nii or .nii.gz extension."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _strip_affixes(name: str) -> str:
    """Strip known prefixes and suffixes from a filename stem."""
    stripped = name
    for prefix in KNOWN_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
    for suffix in KNOWN_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
    return stripped


def _numeric_key(name: str) -> str:
    """Extract the numeric portion of a string."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return digits


def _sorted_nifti_files(directory: Path) -> List[Path]:
    """List and sort NIfTI files in a directory."""
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    files = [path for path in directory.iterdir() if _is_nifti_file(path)]
    if not files:
        raise ValueError(f"No NIfTI files found in {directory}")
    return sorted(files, key=lambda path: _file_stem(path).lower())


def _match_pairs(img_dir: Path, seg_dir: Path) -> List[Tuple[Path, Path]]:
    """Match image and segmentation files according to the pairing rules.

    Args:
        img_dir: Directory containing image files.
        seg_dir: Directory containing segmentation files.

    Returns:
        A list of matched (image_path, segmentation_path) tuples sorted by image stem.

    Raises:
        ValueError: If a required pair cannot be found or multiple candidates exist.
    """
    img_files = _sorted_nifti_files(img_dir)
    seg_files = _sorted_nifti_files(seg_dir)

    seg_records: List[dict] = []
    for seg_path in seg_files:
        stem = _file_stem(seg_path)
        core = _strip_affixes(stem)
        seg_records.append(
            {
                "path": seg_path,
                "stem": stem,
                "core": core,
                "digits": _numeric_key(core),
            }
        )

    pairs: List[Tuple[Path, Path]] = []
    for img_path in img_files:
        stem = _file_stem(img_path)
        # Rule 1: exact stem match.
        match = next((rec for rec in seg_records if rec["stem"] == stem), None)

        if match is None:
            core = _strip_affixes(stem)
            digits = _numeric_key(core)

            # Rule 2: strip affixes and compare numeric keys.
            if digits:
                candidates = [rec for rec in seg_records if rec["digits"] == digits]
                if len(candidates) == 1:
                    match = candidates[0]

            # Fallback: exact core match when numerics are unavailable.
            if match is None and core:
                candidates = [rec for rec in seg_records if rec["core"] == core]
                if len(candidates) == 1:
                    match = candidates[0]

        if match is None:
            raise ValueError(
                f"Unable to find segmentation for image '{img_path.name}'. "
                "Ensure filenames follow the expected pairing rules."
            )

        pairs.append((img_path, match["path"]))
        seg_records.remove(match)

    if seg_records:
        leftover = ", ".join(rec["path"].name for rec in seg_records)
        raise ValueError(f"Unmatched segmentation files: {leftover}")

    return pairs


def _center_slice2d(arr: np.ndarray) -> np.ndarray:
    """Extract the central slice from 3D arrays or validate 2D arrays."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        depth = arr.shape[2]
        return arr[:, :, depth // 2]
    raise ValueError(f"Expected 2D or 3D array, received shape {arr.shape}")


def _random_flip(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random horizontal and vertical flips to image and mask."""
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    return image.copy(), mask.copy()


def _random_rotate(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask by a random angle within [-10°, 10°]."""
    angle = random.uniform(-10.0, 10.0)
    image_rot = transform.rotate(
        image,
        angle=angle,
        resize=False,
        order=1,
        mode="reflect",
        preserve_range=True,
    )
    mask_rot = transform.rotate(
        mask,
        angle=angle,
        resize=False,
        order=0,
        mode="reflect",
        preserve_range=True,
    )
    return image_rot.astype(np.float32), mask_rot.astype(np.float32)


class HipMRI2DDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset for loading central slices of prostate MRI scans with preprocessing."""

    def __init__(
        self,
        root: str,
        size: int = 256,
        aug_flip: bool = True,
        aug_rotate: bool = True,
        fg_label: int | None = 5,
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Root directory containing ``img`` and ``seg`` subdirectories.
            size: Output spatial size (height and width) after resizing.
            aug_flip: Whether to enable random horizontal and vertical flips.
            aug_rotate: Whether to enable random small-angle rotations.
            fg_label: Foreground label value to extract. If ``None``, any non-zero
                voxel is treated as foreground; otherwise equality to ``fg_label``.
        """
        super().__init__()
        self.root = Path(root)
        self.size = size
        self.aug_flip = aug_flip
        self.aug_rotate = aug_rotate
        self.fg_label = fg_label

        img_dir = self.root / "img"
        seg_dir = self.root / "seg"
        self.pairs = _match_pairs(img_dir, seg_dir)

    def __len__(self) -> int:
        """Return the number of cases available."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Load and preprocess a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Tuple containing the image tensor, mask tensor, and case identifier.
        """
        img_path, seg_path = self.pairs[idx]
        case_id = _file_stem(img_path)

        image = self._load_nifti(img_path)
        mask = self._load_nifti(seg_path)

        image = _center_slice2d(image).astype(np.float32)
        mask = _center_slice2d(mask).astype(np.float32)

        if image.shape != mask.shape:
            raise ValueError(
                f"Image and mask shapes differ for case '{case_id}': "
                f"{image.shape} vs {mask.shape}"
            )

        if self.fg_label is None:
            mask = (mask > 0).astype(np.float32)
        else:
            mask = (mask == float(self.fg_label)).astype(np.float32)

        if self.aug_flip:
            image, mask = _random_flip(image, mask)
        if self.aug_rotate:
            image, mask = _random_rotate(image, mask)
            mask = (mask > 0.5).astype(np.float32)

        image = transform.resize(
            image,
            output_shape=(self.size, self.size),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

        mask = transform.resize(
            mask,
            output_shape=(self.size, self.size),
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )
        mask = (mask > 0.5).astype(np.float32)

        mean = float(image.mean())
        std = float(image.std())
        if std > 0:
            image = ((image - mean) / std).astype(np.float32)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        image_tensor = torch.from_numpy(image[np.newaxis, ...])
        mask_tensor = torch.from_numpy(mask[np.newaxis, ...])

        return image_tensor, mask_tensor, case_id

    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        """Load a NIfTI file into a NumPy array."""
        try:
            nifti = nib.load(str(path))
        except FileNotFoundError as exc:
            raise ValueError(f"File not found: {path}") from exc
        data = nifti.get_fdata()
        if data.size == 0:
            raise ValueError(f"NIfTI file is empty: {path}")
        return np.asarray(data)


__all__ = ["HipMRI2DDataset"]
