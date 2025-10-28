from recognition.hipmri_unet_prostate_ayman.utils import dice_metric
import torch


def test_dice_perfect() -> None:
    y = torch.zeros(1, 1, 8, 8)
    y[:, :, 2:6, 3:5] = 1
    assert abs(dice_metric(y, y) - 1.0) < 1e-6
