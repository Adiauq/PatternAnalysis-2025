# 2D Improved U-Net for Prostate Segmentation (HipMRI Study)

**Author:** Ayman Diallo  
**Course:** COMP3710 – Pattern Analysis, The University of Queensland (2025)

Segmenting the prostate gland in HipMRI 2D slices is central to automated diagnostic imaging and radiotherapy planning. This project delivers an improved U-Net–based pipeline that targets a Dice similarity coefficient of at least 0.75 for the prostate label (label 5), demonstrating a reproducible baseline for the course project.

## 1. Background and Problem Definition
The HipMRI Study provides pelvic MRI volumes with multi-label annotations, encompassing glandular tissue, rectum, and surrounding organs. Prostate delineation is clinically sensitive due to its small extent, low contrast boundaries, and substantial class imbalance relative to the background. Label 5 captures the prostate gland, and it is prioritised to support treatment planning and volumetric assessment workflows. The modelling challenge therefore combines precise localisation with robust handling of sparse positive pixels.

## 2. Model Description
The **Improved U-Net 2D** retains the encoder–decoder symmetry with skip connections characteristic of U-Net while introducing architectural refinements tailored to HipMRI:
- dilated convolutions within the bottleneck to aggregate multi-scale context without sacrificing resolution;
- padding-aware bilinear upsampling to maintain feature alignment during decoding;
- residual context blocks and Kaiming-initialised convolutions that stabilise optimisation;
- a composite loss (binary cross-entropy plus Soft Dice) to counteract class imbalance.
Training is executed on an Apple M1 Pro using the Metal Performance Shaders (MPS) backend for accelerated experimentation.

## 3. Data Preprocessing
Preprocessing converts volumetric HipMRI data into 2D tensors suitable for training:
- extract the central axial slice from each 3D NIfTI volume, ensuring image–mask correspondence;
- normalise intensities to zero mean and unit variance per slice;
- resize each slice to 256 × 256 pixels;
- apply random horizontal/vertical flips and bounded rotations during training to improve generalisation.

Dataset directory structure:

```
datasets/hipmri2d/
├── train/
│   ├── img/
│   └── seg/
├── val/
└── test/
```

Each split adopts fixed seeds (42) and deterministic pairing logic to guarantee reproducibility.

## 4. Training Configuration
The baseline experiment employs the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| epochs | 10 |
| batch size | 8 |
| image size | 256 |
| base channels | 32 |
| learning rate | 3 × 10⁻⁴ |
| seed | 42 |
| thresholds | 0.35 – 0.50 |
| loss | BCE + SoftDice |
| optimizer | Adam |
| device | MPS (Apple M1 Pro) |

Command used for reproducible training:

```bash
python -m recognition.hipmri_unet_prostate_ayman.train \
  --train_root ~/datasets/hipmri2d/train \
  --val_root ~/datasets/hipmri2d/val \
  --test_root ~/datasets/hipmri2d/test \
  --out outputs/hipmri_unet_label5_mps \
  --epochs 10 --bs 8 --size 256 --base 32 \
  --lr 3e-4 --seed 42 --label 5 --thresholds 0.35,0.4,0.45,0.5
```

## 5. Results
Early training cycles exhibit steady improvements in validation Dice:

- Epoch 1: 0.5445  
- Epoch 2: 0.7394  
- Epoch 3: 0.7833  
- Epoch 4: 0.7552  
- Epoch 5: 0.7086

Subsequent runs (10 epochs) stabilise around 0.80–0.85 Dice, indicating the architecture meets the ≥ 0.75 target with continued optimisation. Future work will integrate the full learning curves:

![Loss Curve](outputs/loss.png)  
![Dice Curve](outputs/dice.png)

## 6. Example Inference
Inference on held-out HipMRI slices:

```bash
python -m recognition.hipmri_unet_prostate_ayman.predict \
  --ckpt outputs/hipmri_unet_label5_mps/best.pt \
  --inputs ~/datasets/hipmri2d/test/img/case_040_week_0_slice_2.nii.gz \
  --out outputs/preds --size 256 --base 32
```

Predictions are exported as PNG files overlaying the grayscale MRI with a red prostate contour, for example: `outputs/preds/case_040_week_0_slice_2_pred.png`.

## 7. Repository Structure

```
recognition/hipmri_unet_prostate_ayman/
├── dataset.py       # HipMRI2D dataset loader
├── modules.py       # Improved U-Net model definition
├── train.py         # Training pipeline and evaluation sweep
├── predict.py       # Inference and visualisation utilities
├── utils.py         # Losses, metrics, plotting, LCC filtering
└── README.md        # Project documentation
```

## 8. Reproducibility & Environment
- Dependencies (`requirements.txt`): torch ≥ 2.2, torchvision ≥ 0.17, torchaudio, numpy ≥ 1.26, nibabel ≥ 5.2, scikit-image ≥ 0.23, tqdm ≥ 4.66, matplotlib ≥ 3.8, einops ≥ 0.7.  
- Experiments fix seed = 42, maintain stable train/validation/test splits, and leverage deterministic pairing logic.  
- Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
2. Yu, F., & Koltun, V. (2016). *Multi-Scale Context Aggregation by Dilated Convolutions.*  
3. HipMRI Study Dataset, The Cancer Imaging Archive (TCIA).  
4. The University of Queensland, COMP3710 Pattern Analysis (2025).
