from recognition.hipmri_unet_prostate_ayman.modules import ImprovedUNet2D
import torch


def test_forward_shape() -> None:
    m = ImprovedUNet2D(in_channels=1, n_classes=1, base_channels=16)
    x = torch.randn(2, 1, 256, 256)
    y = m(x)
    assert y.shape == (2, 1, 256, 256)
