"""Building blocks for a dilated U-Net architecture tailored to prostate MRI."""

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _compute_same_padding(kernel_size: int, dilation: int) -> int:
    """Compute padding that preserves spatial resolution for odd kernels."""
    return ((kernel_size - 1) * dilation) // 2


def _align_to_reference(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Pad or crop ``tensor`` so its spatial size matches ``reference``."""
    diff_h = reference.size(2) - tensor.size(2)
    diff_w = reference.size(3) - tensor.size(3)

    if diff_h < 0:
        crop = -diff_h
        top = crop // 2
        bottom = crop - top
        tensor = tensor[:, :, top : tensor.size(2) - bottom, :]

    if diff_w < 0:
        crop = -diff_w
        left = crop // 2
        right = crop - left
        tensor = tensor[:, :, :, left : tensor.size(3) - right]

    diff_h = reference.size(2) - tensor.size(2)
    diff_w = reference.size(3) - tensor.size(3)

    if diff_h > 0 or diff_w > 0:
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top
        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        tensor = F.pad(tensor, [pad_left, pad_right, pad_top, pad_bottom])

    return tensor


class ConvBNReLU(nn.Module):
    """Two convolutional layers, each followed by batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        """Initialise the block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            dilation: Dilation rate for both convolutions.
        """
        super().__init__()
        padding = _compute_same_padding(kernel_size=3, dilation=dilation)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutions with normalisation and activation."""
        return self.block(x)


class Down(nn.Module):
    """Down-sampling block with max pooling followed by convolutional refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise the down-sampling block."""
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBNReLU(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce spatial resolution by 2× while enriching features."""
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Up-sampling block with bilinear resize, skip connection, and refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialise the up-sampling block."""
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
        self.conv = ConvBNReLU(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``x``, align to ``skip`` spatially, concatenate, and refine."""
        x = self.upsample(x)
        x = _align_to_reference(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DilatedContextBlock(nn.Module):
    """Stack of dilated Conv-BN-ReLU modules operating at the bottleneck."""

    def __init__(self, channels: int, rates: Sequence[int]) -> None:
        """Initialise the context block with the provided dilation rates."""
        super().__init__()
        layers = [ConvBNReLU(channels, channels, dilation=rate) for rate in rates]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated convolutions to widen the receptive field."""
        return self.block(x)


class ImprovedUNet2D(nn.Module):
    """U-Net style model with dilated context block for 2D medical segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        base_channels: int = 32,
    ) -> None:
        """Construct the network.

        Args:
            in_channels: Number of channels in the input image.
            n_classes: Number of output channels (logits).
            base_channels: Width multiplier for the first encoder stage.
        """
        super().__init__()
        self.inc = ConvBNReLU(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.context = DilatedContextBlock(base_channels * 16, rates=(1, 2, 4))

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialise convolutional layers with Kaiming normal weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the U-Net.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Raw per-class logits of shape ``(N, n_classes, H, W)``.
        """
        enc1 = self.inc(x)  # Level 1 features
        enc2 = self.down1(enc1)  # Level 2 features
        enc3 = self.down2(enc2)  # Level 3 features
        enc4 = self.down3(enc3)  # Level 4 features
        bottleneck = self.down4(enc4)  # Deepest features before context
        bottleneck = self.context(bottleneck)  # Dilated context aggregation

        dec1 = self.up1(bottleneck, enc4)  # Merge with level 4 skip connection
        dec2 = self.up2(dec1, enc3)  # Merge with level 3 skip connection
        dec3 = self.up3(dec2, enc2)  # Merge with level 2 skip connection
        dec4 = self.up4(dec3, enc1)  # Merge with level 1 skip connection

        logits = self.out_conv(dec4)
        return logits


__all__ = [
    "ConvBNReLU",
    "Down",
    "Up",
    "DilatedContextBlock",
    "ImprovedUNet2D",
]
