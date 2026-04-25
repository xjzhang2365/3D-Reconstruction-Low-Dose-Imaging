"""
Denoiser benchmark utility.

NOTE: This script measures denoiser output against an input reference
(or temporal-average proxy), not against noise-free ground truth. The
resulting metrics should be interpreted as relative comparison between
methods, not absolute quality scores. For publication-quality PSNR/SSIM,
run against simulated ground truth images.

Compares BM3D, K-SVD, and U-Net on the preprocessed averaged frame.
Results saved to outputs/benchmarks/denoising_comparison.json.

Notes on K-SVD runtime
----------------------
K-SVD uses sklearn DictionaryLearning which is O(n_patches * n_atoms * n_iter).
With stride=1 on a 512×512 image this can take hours. The benchmark uses
stride=8 (non-overlapping 8×8 grid step) which reduces patches from ~250k to
~4k and runs in minutes while still covering the full image.

Usage:
    python scripts/run_full_denoising_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from graphene3d.preprocessing.denoising import BM3DDenoiser, KSVDDenoiser, UNetDenoiser

KSVD_STRIDE = 8    # stride for benchmark; stride=1 takes hours on full images


def norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # ------------------------------------------------------------------
    # Load input frame
    # ------------------------------------------------------------------
    input_path = REPO_ROOT / "outputs" / "preprocessing" / "averaged_frame21.npy"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print("Run preprocessing first: python scripts/run_preprocessing.py")
        sys.exit(1)

    noisy = np.load(input_path).astype(np.float64)
    print(f"Input: {input_path.name}  shape={noisy.shape}  dtype=float64")

    # ------------------------------------------------------------------
    # Choose reference
    # ------------------------------------------------------------------
    ref_path = REPO_ROOT / "outputs" / "preprocessing" / "reference_clean.npy"
    if ref_path.exists():
        reference = np.load(ref_path).astype(np.float64)
        ref_source = "high-quality reference (reference_clean.npy)"
    else:
        reference = noisy.copy()
        ref_source = "5-frame temporal average (proxy; self-reference — PSNR not interpretable in absolute terms)"

    print(f"Reference: {ref_source}")

    noisy_n = norm01(noisy)
    ref_n   = norm01(reference)

    results: dict = {}

    # ------------------------------------------------------------------
    # BM3D
    # ------------------------------------------------------------------
    print("\nRunning BM3D (paper-aligned: profile='np', HARD_THRESHOLDING)...")
    t0 = time.time()
    bm3d = BM3DDenoiser()
    out_bm3d = bm3d.denoise(noisy_n)
    t_bm3d = time.time() - t0
    results["BM3D"] = {
        "psnr": float(peak_signal_noise_ratio(ref_n, out_bm3d, data_range=1.0)),
        "ssim": float(structural_similarity(ref_n, out_bm3d, data_range=1.0)),
        "runtime_s": round(t_bm3d, 3),
        "params": "sigma_psd=auto(MAD), profile='np', stage_arg=HARD_THRESHOLDING",
    }
    print(f"  PSNR={results['BM3D']['psnr']:.2f} dB  "
          f"SSIM={results['BM3D']['ssim']:.4f}  "
          f"time={t_bm3d:.2f}s")

    # ------------------------------------------------------------------
    # K-SVD  (stride=8 to keep runtime tractable)
    # ------------------------------------------------------------------
    print(f"\nRunning K-SVD (stride={KSVD_STRIDE}, n_atoms=256, n_iter=50)...")
    try:
        t0 = time.time()
        ksvd = KSVDDenoiser(stride=KSVD_STRIDE)
        out_ksvd = ksvd.denoise(noisy_n)
        t_ksvd = time.time() - t0
        results["K-SVD"] = {
            "psnr": float(peak_signal_noise_ratio(ref_n, out_ksvd, data_range=1.0)),
            "ssim": float(structural_similarity(ref_n, out_ksvd, data_range=1.0)),
            "runtime_s": round(t_ksvd, 3),
            "params": f"patch_size=8, n_atoms=256, n_iter=50, stride={KSVD_STRIDE}",
        }
        print(f"  PSNR={results['K-SVD']['psnr']:.2f} dB  "
              f"SSIM={results['K-SVD']['ssim']:.4f}  "
              f"time={t_ksvd:.2f}s")
    except Exception as exc:
        results["K-SVD"] = {"error": str(exc), "params": f"stride={KSVD_STRIDE}"}
        print(f"  K-SVD failed: {exc}")

    # ------------------------------------------------------------------
    # U-Net  (random weights — no pretrained weights in repo)
    # ------------------------------------------------------------------
    print("\nRunning U-Net (random weights — no pretrained weights in repo)...")
    try:
        t0 = time.time()
        unet = UNetDenoiser()
        out_unet = unet.denoise(noisy_n)
        t_unet = time.time() - t0
        results["U-Net"] = {
            "psnr": float(peak_signal_noise_ratio(ref_n, out_unet, data_range=1.0)),
            "ssim": float(structural_similarity(ref_n, out_unet, data_range=1.0)),
            "runtime_s": round(t_unet, 3),
            "params": "random weights (untrained); metrics not meaningful",
            "note": "Pretrained weights not included in this repository. "
                    "See thesis chapter 3 for original evaluation results.",
        }
        print(f"  PSNR={results['U-Net']['psnr']:.2f} dB  "
              f"SSIM={results['U-Net']['ssim']:.4f}  "
              f"time={t_unet:.2f}s  [random weights — metrics not meaningful]")
    except Exception as exc:
        results["U-Net"] = {
            "error": str(exc),
            "note": "Pretrained weights not included in this repository. "
                    "See thesis chapter 3 for original evaluation results.",
        }
        print(f"  U-Net failed: {exc}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_path = REPO_ROOT / "outputs" / "benchmarks" / "denoising_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_source": ref_source,
        "input": str(input_path),
        "input_shape": list(noisy.shape),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print("\n=== Denoising Benchmark Results ===")
    print(f"Reference: {ref_source}")
    print(f"{'Method':<20} {'PSNR (dB)':>10} {'SSIM':>8} {'Runtime (s)':>12}")
    print("-" * 54)
    for method, r in results.items():
        if "error" in r:
            print(f"{method:<20} {'ERROR':>10} {'—':>8}  {r['error'][:25]}")
        elif r.get("params", "").startswith("random"):
            print(
                f"{method:<20} {r['psnr']:>10.2f} {r['ssim']:>8.4f} "
                f"{r['runtime_s']:>12.2f}  [random weights]"
            )
        else:
            print(
                f"{method:<20} {r['psnr']:>10.2f} {r['ssim']:>8.4f} "
                f"{r['runtime_s']:>12.2f}"
            )


if __name__ == "__main__":
    main()
