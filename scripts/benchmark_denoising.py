"""
Denoising benchmark for low-dose TEM frames.

Applies BM3D (paper-aligned parameters) to the pre-denoising averaged frame
and computes PSNR and SSIM.

Reference image selection (in priority order):
  1. data/simulated/target_preprocessed_like_raw21.npy  — if shape matches
  2. outputs/preprocessing/averaged_frame21.npy          — 5-frame temporal avg
     used as a proxy if no true clean reference is available

Input (pre-BM3D):
  outputs/preprocessing/averaged_frame21.npy  — flat-field corrected,
  dead-pixel removed, 5-frame temporally averaged (but not yet BM3D denoised)

Usage:
    python scripts/benchmark_denoising.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AVERAGED_NPY = REPO_ROOT / "outputs" / "preprocessing" / "averaged_frame21.npy"
SINGLE_NPY   = REPO_ROOT / "outputs" / "preprocessing" / "single_corrected_frame21.npy"
SIMULATED_TARGET = REPO_ROOT / "data" / "simulated" / "target_preprocessed_like_raw21.npy"
OUTPUT_DIR   = REPO_ROOT / "outputs" / "preprocessing"
FIGURE_PATH  = OUTPUT_DIR / "benchmark_denoising_comparison.png"


def _load(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64)


def _psnr_ssim(reference: np.ndarray, denoised: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    data_range = reference.max() - reference.min()
    p = peak_signal_noise_ratio(reference, denoised, data_range=data_range)
    s = structural_similarity(reference, denoised, data_range=data_range)
    return p, s


def _save_figure(
    noisy: np.ndarray,
    denoised: np.ndarray,
    reference: np.ndarray,
    ref_label: str,
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Input (pre-BM3D)", "BM3D denoised", f"Reference\n({ref_label})"]
    images = [noisy, denoised, reference]
    for ax, img, title in zip(axes, images, titles):
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("BM3D denoising benchmark — paper-aligned parameters", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # ------------------------------------------------------------------
    # Load input frame (pre-BM3D)
    # ------------------------------------------------------------------
    if not AVERAGED_NPY.exists():
        print(
            f"ERROR: Input not found: {AVERAGED_NPY}\n"
            "Run preprocessing first:\n"
            "  python scripts/run_preprocessing.py"
        )
        sys.exit(1)

    noisy = _load(AVERAGED_NPY)
    print(f"Input frame      : {AVERAGED_NPY.name}  shape={noisy.shape}")

    # ------------------------------------------------------------------
    # Choose reference
    # ------------------------------------------------------------------
    ref_label: str
    if SIMULATED_TARGET.exists():
        candidate = _load(SIMULATED_TARGET)
        if candidate.shape[:2] == noisy.shape[:2]:
            reference = candidate[:, :] if candidate.ndim == 2 else candidate
            ref_label = "simulated target (target_preprocessed_like_raw21)"
        else:
            reference = noisy    # fallback — metrics not meaningful; noted below
            ref_label = "(shape mismatch — metrics vs. self, not meaningful)"
    elif SINGLE_NPY.exists():
        reference = _load(SINGLE_NPY)
        ref_label = "single corrected frame (no BM3D; proxy reference)"
    else:
        reference = noisy
        ref_label = "(no reference — metrics vs. self)"

    print(f"Reference        : {ref_label}")

    # ------------------------------------------------------------------
    # Apply BM3D (paper-aligned parameters)
    # ------------------------------------------------------------------
    from graphene3d.preprocessing.denoising import BM3DDenoiser

    denoiser = BM3DDenoiser()    # sigma_psd=None → auto-estimated via MAD
    t0 = time.perf_counter()
    denoised = denoiser.denoise(noisy)
    elapsed = time.perf_counter() - t0

    print(f"BM3D runtime     : {elapsed:.2f} s")
    print(f"  sigma (auto)   : {denoiser.sigma_psd if denoiser.sigma_psd is not None else 'auto (MAD)'}")
    print(f"  profile        : np")
    print(f"  stage_arg      : BM3DStages.HARD_THRESHOLDING")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if reference is not noisy:
        ref_is_self = False
        psnr_before, ssim_before = _psnr_ssim(reference, noisy)
        psnr_after,  ssim_after  = _psnr_ssim(reference, denoised)

        print("\n--- Benchmark results ---")
        print(f"{'Method':<30}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'Runtime (s)':>12}")
        print("-" * 68)
        print(f"{'Input (pre-BM3D)':<30}  {psnr_before:>10.2f}  {ssim_before:>8.4f}  {'—':>12}")
        print(f"{'BM3D (paper-aligned)':<30}  {psnr_after:>10.2f}  {ssim_after:>8.4f}  {elapsed:>12.2f}")
    else:
        ref_is_self = True
        print("\nNOTE: No independent reference available. Metrics not computed.")
        print("      Run with a simulated target or a high-dose reference in data/.")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    _save_figure(noisy, denoised, reference, ref_label, FIGURE_PATH)

    # ------------------------------------------------------------------
    # Summary for docs/denoising_comparison.md
    # ------------------------------------------------------------------
    print("\n--- Values for docs/denoising_comparison.md ---")
    if not ref_is_self:
        print(f"BM3D   PSNR = {psnr_after:.2f} dB   SSIM = {ssim_after:.4f}   "
              f"Runtime = {elapsed:.1f} s/frame")
        print(f"Input  PSNR = {psnr_before:.2f} dB   SSIM = {ssim_before:.4f}   (pre-BM3D baseline)")
    else:
        print("Run with a clean reference to get PSNR/SSIM values.")


if __name__ == "__main__":
    main()
