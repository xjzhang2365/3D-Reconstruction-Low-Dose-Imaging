# Low-Dose TEM Denoising Method Comparison

## Summary

During the PhD work underlying this repository (arXiv:2604.07271),
three denoising methods were systematically compared on the paper's
low-dose TEM dataset (80 kV, 8×10³ e⁻/Å², 1 ms frames):

- **BM3D** (Block-Matching 3D filtering) — Danielyan et al., 2011
- **Dictionary Learning (K-SVD)** — sparse coding, per-image dictionary
- **CNN (U-Net)** — custom architecture trained on paired clean/noisy pairs

BM3D was selected for production. The rationale is documented below.

## Method Selection Rationale

**BM3D was selected because:**

1. **No training data required.** For single-frame, dose-calibrated
   imaging where paired clean/noisy data is unavailable at the
   target dose, BM3D's self-similarity prior works directly on each
   frame without any training phase.

2. **Preserves atomic-scale contrast.** BM3D's collaborative filtering
   in 3D groups of similar patches preserves the graphene honeycomb
   lattice without over-smoothing individual atomic peaks.

3. **Physically interpretable noise parameter.** BM3D's σ parameter
   maps directly to the expected noise standard deviation, which for
   dose-calibrated Poisson statistics at 8×10³ e⁻/Å² is analytically
   predictable. Sigma is auto-estimated from the image via Median
   Absolute Deviation (MAD) in this implementation.

4. **Reliable across dose levels.** BM3D generalises across frames
   with different effective dose without retraining.

**Why not K-SVD:** K-SVD achieved comparable reconstruction quality
on the training frame but required per-image dictionary training,
which is expensive (~30× slower per frame than BM3D in this
implementation) and produces dictionaries that do not transfer
cleanly between frames at different dose levels.

**Why not U-Net:** The custom U-Net produced the highest PSNR on
training-distribution images but required paired clean/noisy training
data that is unavailable at the paper's target dose of 8×10³ e⁻/Å².
When evaluated on frames with dose levels or defect structures not
represented in training, U-Net over-smoothed atomic features and
occasionally hallucinated lattice continuity across real defect
regions. This is the well-known generalisation problem of supervised
denoising at the single-image, single-dose regime.

## BM3D Configuration in This Repository

The BM3D implementation ([`src/graphene3d/preprocessing/denoising.py`](../src/graphene3d/preprocessing/denoising.py))
uses:

- `sigma_psd`: auto-estimated per frame via Median Absolute Deviation
- `profile`: `'np'` (normal profile)
- `stage_arg`: `BM3DStages.HARD_THRESHOLDING`

These settings match the paper's Methods section.

## Reproducing the Implementation

The three denoiser implementations are all available:

```python
from graphene3d.preprocessing.denoising import (
    BM3DDenoiser, KSVDDenoiser, UNetDenoiser,
)

bm3d = BM3DDenoiser()
denoised = bm3d.denoise(noisy_image)
```

`UNetDenoiser` requires pretrained weights which are not distributed
with this repository. Refer to the original thesis work for the
training protocol and weights.

## References

- BM3D: Danielyan, A., Katkovnik, V., and Egiazarian, K. (2011).
  *BM3D frames and variational image deblurring.* IEEE TIP, 21(4).
- K-SVD: Aharon, M., Elad, M., and Bruckstein, A. (2006).
  *K-SVD: An algorithm for designing overcomplete dictionaries.*
- U-Net: Ronneberger, O., Fischer, P., and Brox, T. (2015).
  *U-Net: Convolutional networks for biomedical image segmentation.*
