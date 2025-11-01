import pytest
import torch

from recognition.hipmri_unet_prostate_ayman.utils import dice_metric


def test_dice_perfect() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 3:5] = 1

    logits = torch.full_like(target, -10.0)
    logits[target == 1] = 10.0

    score = dice_metric(logits, target)
    assert score == pytest.approx(1.0, abs=1e-6)
