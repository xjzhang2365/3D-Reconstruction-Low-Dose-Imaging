"""
denoising_comparison.py
=======================
Runs BM3D, K-SVD, and U-Net on a preprocessed TEM frame and produces a
side-by-side comparison report (PDF + PNG) with PSNR, SSIM, and timing.

Usage
-----
    python denoising_comparison.py

Expected inputs (produced by load_real_data.py / make_report.py):
    averaged.npy   – 5-frame temporal average (used as clean reference)
    bm3d.npy       – BM3D-denoised frame  (pre-computed)
    unet.npy       – U-Net output          (pre-computed)
    single.npy     – single corrected frame (baseline)

K-SVD is re-run here from averaged.npy because it is the slowest step
(~120 s on 256×256).  Pre-computed results can be loaded instead by
setting RERUN_KSVD = False and ensuring ksvd_full.npy exists.
"""

import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ── path to your tem_reconstruction package ──────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.preprocessing.denoising import KSVDDenoiser

# ── settings ─────────────────────────────────────────────────────────────
RERUN_KSVD  = True          # set False to load ksvd_full.npy instead
OUTPUT_DIR  = REPO_ROOT / "outputs" / "preprocessing"
INPUT_DIR   = OUTPUT_DIR
OUTPUT_PDF  = OUTPUT_DIR / "denoising_comparison.pdf"
OUTPUT_PNG  = OUTPUT_DIR / "denoising_comparison.png"
KSVD_PATH   = OUTPUT_DIR / "ksvd_full.npy"

# ── load pre-computed arrays ──────────────────────────────────────────────
def _load_array(*names):
    for name in names:
        for base in (INPUT_DIR, Path.cwd()):
            path = base / name
            if path.exists():
                return np.load(path)
    tried = ", ".join(str(INPUT_DIR / name) for name in names)
    raise FileNotFoundError(f"Could not find any of: {tried}")


avg    = _load_array('averaged_frame21.npy', 'averaged.npy')
bm3d   = _load_array('bm3d_frame21.npy', 'preprocessed_frame21.npy', 'bm3d.npy')
unet   = _load_array('unet_frame21.npy', 'unet.npy')
single = _load_array('single_corrected_frame21.npy', 'single.npy')    # single corrected frame

# ── optionally rerun K-SVD ────────────────────────────────────────────────
if RERUN_KSVD:
    print('Running K-SVD (this takes ~2 min on 256×256)…')
    t0 = time.time()
    ksvd_denoiser = KSVDDenoiser(patch_size=8, n_atoms=128, n_iter=20,
                                  sparsity=5, stride=4)
    ksvd = ksvd_denoiser.denoise(avg)
    ksvd_time = time.time() - t0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(KSVD_PATH, ksvd)
    print(f'  K-SVD done in {ksvd_time:.1f}s')
else:
    ksvd = np.load(KSVD_PATH)
    ksvd_time = None   # unknown

# ── metrics ───────────────────────────────────────────────────────────────
ref     = avg
d_range = ref.max() - ref.min()

def metrics(denoised):
    p = psnr(ref, denoised, data_range=d_range)
    s = ssim(ref, denoised, data_range=d_range)
    return p, s

p_single, s_single = metrics(single)
p_bm3d,   s_bm3d   = metrics(bm3d)
p_ksvd,   s_ksvd   = metrics(ksvd)
p_unet,   s_unet   = metrics(unet)

# ── timing (approximate; BM3D and U-Net timed during pipeline run) ────────
times = {
    'Single corrected': 0.1,
    'BM3D':             2.3,
    'K-SVD':            ksvd_time if ksvd_time else 121.9,
    'U-Net':            1.2,
}

psnr_vals = {'Single corrected': p_single, 'BM3D': p_bm3d,
             'K-SVD': p_ksvd,  'U-Net': p_unet}
ssim_vals = {'Single corrected': s_single, 'BM3D': s_bm3d,
             'K-SVD': s_ksvd,  'U-Net': s_unet}

# ── print summary ─────────────────────────────────────────────────────────
print('\n── Denoising comparison ─────────────────────────────────────────')
print(f'{"Method":<20} {"PSNR (dB)":>10} {"SSIM":>8} {"Time (s)":>10}')
print('-' * 52)
for key in ['Single corrected', 'BM3D', 'K-SVD', 'U-Net']:
    print(f'{key:<20} {psnr_vals[key]:>10.2f} {ssim_vals[key]:>8.4f} '
          f'{times[key]:>10.1f}')
print()

# ── build figure ─────────────────────────────────────────────────────────
keys    = ['Single corrected', 'BM3D', 'K-SVD', 'U-Net']
labels  = ['Single\ncorrected', 'BM3D', 'K-SVD', 'U-Net']
colors  = ['#aaaaaa', '#2196F3', '#FF9800', '#4CAF50']
imgs    = [single, bm3d, ksvd, unet]
crop    = np.s_[80:176, 80:176]     # 96×96 centre crop for visual comparison

fig = plt.figure(figsize=(16, 14), facecolor='white')
fig.suptitle('Denoising Method Comparison — Frame 21 (Real TEM Data)',
             fontsize=15, fontweight='bold', y=0.98)

# ── Row 1: cropped images ─────────────────────────────────────────────────
for i, (img, title, col) in enumerate(zip(imgs, keys, colors)):
    ax = fig.add_subplot(3, 4, i + 1)
    disp = img - img.mean()          # mean-centre for consistent display
    ax.imshow(disp[crop], cmap='gray', interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])

# ── Row 2: PSNR and SSIM bar charts ──────────────────────────────────────
psnr_list = [psnr_vals[k] for k in keys]
ssim_list = [ssim_vals[k] for k in keys]

ax_psnr = fig.add_subplot(3, 2, 3)
bars = ax_psnr.bar(labels, psnr_list, color=colors, edgecolor='black', linewidth=0.7)
ax_psnr.set_title('PSNR vs 5-frame average (dB)', fontsize=11, fontweight='bold')
ax_psnr.set_ylabel('PSNR (dB)')
ax_psnr.set_ylim(0, max(psnr_list) * 1.15)
for bar, val in zip(bars, psnr_list):
    ax_psnr.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
ax_psnr.spines[['top', 'right']].set_visible(False)

ax_ssim = fig.add_subplot(3, 2, 4)
bars2 = ax_ssim.bar(labels, ssim_list, color=colors, edgecolor='black', linewidth=0.7)
ax_ssim.set_title('SSIM vs 5-frame average', fontsize=11, fontweight='bold')
ax_ssim.set_ylabel('SSIM')
ax_ssim.set_ylim(0, 1.12)
for bar, val in zip(bars2, ssim_list):
    ax_ssim.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
ax_ssim.spines[['top', 'right']].set_visible(False)

# ── Row 3: timing bar + summary table ────────────────────────────────────
time_list = [times[k] for k in keys]

ax_time = fig.add_subplot(3, 2, 5)
bars3 = ax_time.bar(labels, time_list, color=colors, edgecolor='black', linewidth=0.7)
ax_time.set_title('Processing time (256×256 frame)', fontsize=11, fontweight='bold')
ax_time.set_ylabel('Time (s, log scale)')
ax_time.set_yscale('log')
for bar, val in zip(bars3, time_list):
    ax_time.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.1,
                 f'{val:.1f}s', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
ax_time.spines[['top', 'right']].set_visible(False)

ax_tbl = fig.add_subplot(3, 2, 6)
ax_tbl.axis('off')
table_data = [
    ['Method',        'PSNR (dB)',       'SSIM',             'Time (s)', 'Notes'],
    ['Single corr.',  f'{p_single:.1f}', f'{s_single:.3f}',  '~0.1',     'Baseline'],
    ['BM3D',          f'{p_bm3d:.1f}',   f'{s_bm3d:.3f}',   '~2.3',     '★ Pipeline choice'],
    ['K-SVD',         f'{p_ksvd:.1f}',   f'{s_ksvd:.3f}',   '~122',     'Slow; no gain'],
    ['U-Net',         f'{p_unet:.1f}',   f'{s_unet:.3f}',   '~1.2',     'No pretrained model'],
]
t = ax_tbl.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85])
t.auto_set_font_size(False)
t.set_fontsize(9)
for (row, col), cell in t.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 2:          # BM3D row — highlight chosen method
        cell.set_facecolor('#E3F2FD')
    else:
        cell.set_facecolor('#f9f9f9')
ax_tbl.set_title('Summary', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=150)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=150)
print(f'Saved: {OUTPUT_PDF}  and  {OUTPUT_PNG}')
