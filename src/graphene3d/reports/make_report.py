import numpy as np, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import fft2, fftshift
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ── Palette ──────────────────────────────────────────────────
BG   = '#ffffff'; PANEL = '#f6f8fa'; BORDER = '#d0d7de'
TXT  = '#1a1a1a'; DIM   = '#57606a'
BLUE = '#0969da'; GOLD  = '#bf8700'; GRN = '#1a7f37'; RED = '#cf222e'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': BORDER, 'text.color': TXT,
    'axes.labelcolor': TXT, 'xtick.color': DIM, 'ytick.color': DIM,
    'grid.color': BORDER, 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 9, 'axes.labelsize': 8,
})

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Load / compute data ──────────────────────────────────────
from graphene3d.preprocessing.averaging import temporal_average
from graphene3d.preprocessing.corrections import correct_flat_field_stack, remove_dead_pixels_stack
from graphene3d.preprocessing.denoising import BM3DDenoiser

FRAME_IDS = [19, 20, 21, 22, 23]
raw_frames = [tifffile.imread(REPO_ROOT / "data" / "experimental" / f"raw_{i}.tif").astype(np.float64)
              for i in FRAME_IDS]
stack_raw  = np.stack(raw_frames)
stack_norm = stack_raw / stack_raw.max()

corr_stack        = correct_flat_field_stack(stack_norm, sigma=20.0)
cleaned, masks    = remove_dead_pixels_stack(corr_stack, threshold_sigma=5.0)
single            = cleaned[2]
averaged          = temporal_average(cleaned, target_idx=2, window_size=5)
bm3d              = BM3DDenoiser().denoise(averaged)
n_dead_21         = int(masks[2].sum())

# Mean-centre for shared display scale
single_d  = single - single.mean()
bm3d_d    = bm3d   - bm3d.mean()
lim       = max(abs(single_d.min()), abs(single_d.max()))
VMIN, VMAX = -lim * 0.85, lim * 0.85

CY, CX, SZ = 128, 128, 48   # zoom: 96×96 centre crop

# ── Helpers ──────────────────────────────────────────────────
def hdr(fig, txt, y=0.96):
    fig.text(0.5, y, txt, ha='center', va='top',
             fontsize=12, fontweight='bold', color=TXT)

def footnote(fig, txt):
    fig.text(0.5, 0.01, txt, ha='center', fontsize=7.5, color=DIM)

def tag(ax, txt, color=DIM, loc='br'):
    x = 0.03 if 'l' in loc else 0.97
    y = 0.97 if 't' in loc else 0.04
    ax.text(x, y, txt, transform=ax.transAxes,
            ha='left' if 'l' in loc else 'right',
            va='top'  if 't' in loc else 'bottom',
            fontsize=7, color=color,
            bbox=dict(fc=BG, ec=color, lw=0.7, boxstyle='round,pad=0.25', alpha=0.85))

def scb(ax, im, label=''):
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.05)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=6)
    cb.outline.set_edgecolor(BORDER)
    if label: cb.set_label(label, color=DIM, fontsize=7)

def ps(img):
    p = np.abs(fftshift(fft2(img - img.mean())))**2
    p = np.log1p(p)
    H, W = p.shape
    p[H//2-4:H//2+4, W//2-4:W//2+4] = p.min()
    return p

OUTPUT_DIR = REPO_ROOT / "outputs" / "preprocessing"
OUTPUT = OUTPUT_DIR / "real_data_verification.pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with PdfPages(OUTPUT) as pdf:

    # ══════════════════════════════════════════════════════════
    # PAGE 1 — Five raw frames, single shared colorbar
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 5), facecolor=BG)
    hdr(fig, "Five Consecutive TEM Frames  ·  raw_19 – raw_23  ·  256 × 256 px  ·  uint16")
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05,
                           left=0.03, right=0.92, top=0.83, bottom=0.06)

    RMIN, RMAX = float(stack_raw.min()), float(stack_raw.max())
    last_im = None
    for col, (frame, fid) in enumerate(zip(raw_frames, FRAME_IDS)):
        ax = fig.add_subplot(gs[col])
        last_im = ax.imshow(frame, cmap='gray', origin='upper', vmin=RMIN, vmax=RMAX)
        ax.axis('off')
        is_tgt = (fid == 21)
        ec = GOLD if is_tgt else BORDER
        lw = 2.5  if is_tgt else 0.7
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(lw)
        snr = frame.mean() / frame.std()
        ax.set_title(f"raw_{fid}.tif" + ("  ★" if is_tgt else ""),
                     color=GOLD if is_tgt else TXT,
                     fontsize=8.5, fontweight='bold' if is_tgt else 'normal', pad=4)
        tag(ax, f"SNR {snr:.2f}", color=GOLD if is_tgt else DIM, loc='br')

    # Single colorbar far right
    cax = fig.add_axes([0.934, 0.06, 0.011, 0.77])
    cb  = fig.colorbar(last_im, cax=cax)
    cb.set_label('Intensity (counts)', color=DIM, fontsize=7.5)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=7)
    cb.outline.set_edgecolor(BORDER)
    footnote(fig, "All five frames share the same intensity scale  ·  Global min = 43  max = 499 counts")
    pdf.savefig(fig, bbox_inches='tight', facecolor=BG); plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # PAGE 2 — Pipeline steps
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 5.5), facecolor=BG)
    hdr(fig, "Preprocessing Pipeline  ·  Frame 21  ·  Four Steps")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.10,
                           left=0.03, right=0.97, top=0.83, bottom=0.06)

    bg21    = gaussian_filter(stack_norm[2], sigma=20.0)
    step_imgs = [
        (stack_norm[2], "① Normalised raw",          BLUE, f"SNR {stack_raw[2].mean()/stack_raw[2].std():.2f}"),
        (bg21,          "② Background (σ = 20 px)",  DIM,  "Gaussian blur"),
        (corr_stack[2], "③ Flat-field corrected",    BLUE, "subtract bg"),
        (single,        "④ Dead-pixel removed",      GRN,  f"{n_dead_21} px fixed"),
    ]
    for col, (img, title, col_c, note) in enumerate(step_imgs):
        ax = fig.add_subplot(gs[col])
        im = ax.imshow(img, cmap='gray', origin='upper',
                       vmin=float(np.percentile(img, 0.5)),
                       vmax=float(np.percentile(img, 99.5)))
        ax.axis('off')
        ax.set_title(title, color=col_c, fontsize=9, pad=5)
        tag(ax, note, color=DIM, loc='br')
        scb(ax, im)
        if col < 3:
            ax.annotate('', xy=(1.07, 0.5), xycoords='axes fraction',
                        xytext=(1.02, 0.5),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=2))
    pdf.savefig(fig, bbox_inches='tight', facecolor=BG); plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # PAGE 3 — Full frame + zoom (2 cols × 2 rows)
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(13, 10), facecolor=BG)
    hdr(fig, "Single Frame vs 5-Frame Average + BM3D  ·  Full View & Detail Zoom")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.07, hspace=0.10,
                           left=0.04, right=0.94, top=0.90, bottom=0.07)

    pairs = [
        (single_d, f"Single frame (corrected)  ·  std = {single.std():.4f}", DIM),
        (bm3d_d,   f"5-frame avg + BM3D  ·  std = {bm3d.std():.4f}",        GOLD),
    ]
    for col, (img, title, col_c) in enumerate(pairs):
        # Full frame
        ax_f = fig.add_subplot(gs[0, col])
        im   = ax_f.imshow(img, cmap='gray', origin='upper', vmin=VMIN, vmax=VMAX)
        ax_f.axis('off')
        ax_f.set_title(title, color=col_c, fontsize=9, pad=5, fontweight='bold')
        scb(ax_f, im)
        # Dashed zoom box
        rect = Rectangle((CX-SZ, CY-SZ), SZ*2, SZ*2,
                         lw=1.5, ec=BLUE, fc='none', ls='--')
        ax_f.add_patch(rect)

        # Zoom
        ax_z = fig.add_subplot(gs[1, col])
        zoom = img[CY-SZ:CY+SZ, CX-SZ:CX+SZ]
        ax_z.imshow(zoom, cmap='gray', origin='upper', vmin=VMIN, vmax=VMAX)
        ax_z.axis('off')
        ax_z.set_title(f"Detail — {SZ*2} × {SZ*2} px  (dashed region above)",
                       color=BLUE, fontsize=8, pad=4)
        scb(ax_z, im)

    imp = single.std() / bm3d.std()
    footnote(fig,
        f"Noise std:  {single.std():.4f}  →  {bm3d.std():.4f}   "
        f"({imp:.1f}× reduction  ·  theory √5 = {5**0.5:.2f}×)   "
        f"·  mean-centred for shared display scale")
    pdf.savefig(fig, bbox_inches='tight', facecolor=BG); plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # PAGE 4 — Intensity profiles & histogram
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    hdr(fig, "Intensity Comparison  ·  Single Frame vs 5-Frame Average + BM3D")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.42,
                           left=0.08, right=0.97, top=0.88, bottom=0.08)

    ax_row  = fig.add_subplot(gs[0, :])
    ax_col  = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    pxcols = np.arange(256)
    pxrows = np.arange(256)

    # — Horizontal profile —
    ax_row.plot(pxcols, single_d[CY], color=BLUE, lw=1.0, alpha=0.75, label='Single frame')
    ax_row.plot(pxcols, bm3d_d[CY],   color=GOLD, lw=2.0, alpha=0.95, label='5-frame avg + BM3D')
    ax_row.set_xlabel('Column pixel'); ax_row.set_ylabel('Mean-centred intensity')
    ax_row.set_title(f'Horizontal profile at row {CY}  ·  noise reduced, lattice signal preserved',
                     color=TXT, pad=6)
    ax_row.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, framealpha=0.9)
    ax_row.grid(True); ax_row.set_xlim(0, 255)
    ax_row.spines[['top','right']].set_visible(False)

    # — Vertical profile —
    ax_col.plot(single_d[:, CX], pxrows, color=BLUE, lw=1.0, alpha=0.75, label='Single frame')
    ax_col.plot(bm3d_d[:, CX],   pxrows, color=GOLD, lw=2.0, alpha=0.95, label='5-frame avg + BM3D')
    ax_col.invert_yaxis()
    ax_col.set_xlabel('Mean-centred intensity'); ax_col.set_ylabel('Row pixel')
    ax_col.set_title(f'Vertical profile at col {CX}', color=TXT, pad=6)
    ax_col.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, framealpha=0.9)
    ax_col.grid(True); ax_col.set_ylim(255, 0)
    ax_col.spines[['top','right']].set_visible(False)

    # — Histogram —
    bins = np.linspace(VMIN, VMAX, 80)
    ax_hist.hist(single_d.ravel(), bins=bins, color=BLUE, alpha=0.45, label='Single frame')
    ax_hist.hist(bm3d_d.ravel(),   bins=bins, color=GOLD, alpha=0.65, label='5-frame avg + BM3D')
    ax_hist.set_xlabel('Mean-centred intensity'); ax_hist.set_ylabel('Pixel count')
    ax_hist.set_title('Intensity histogram  ·  narrower distribution = less noise',
                      color=TXT, pad=6)
    ax_hist.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, framealpha=0.9)
    ax_hist.grid(True)
    ax_hist.spines[['top','right']].set_visible(False)
    # Annotate std
    for x, c, lbl in [(single_d.std(), BLUE, f'σ={single.std():.4f}'),
                       (bm3d_d.std(),   GOLD, f'σ={bm3d.std():.4f}')]:
        ax_hist.axvline(x,  color=c, ls=':', lw=1.2)
        ax_hist.axvline(-x, color=c, ls=':', lw=1.2)

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG); plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # PAGE 5 — FFT lattice check
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(15, 5.5), facecolor=BG)
    hdr(fig, "FFT Power Spectra  ·  Lattice Periodicity Check")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.07,
                           left=0.03, right=0.97, top=0.83, bottom=0.05)

    for col, (name, img, col_c) in enumerate([
        ('Raw frame 21 (normalised)', stack_norm[2], DIM),
        ('Corrected — single frame',  single,        BLUE),
        ('5-frame avg + BM3D',        bm3d,          GOLD),
    ]):
        ax = fig.add_subplot(gs[col])
        ax.imshow(ps(img), cmap='inferno', origin='upper')
        ax.axis('off')
        ax.set_title(name, color=col_c, fontsize=9, pad=5)
        tag(ax, 'off-axis peaks = lattice', color=DIM, loc='br')

    footnote(fig, "Sharper, brighter off-axis peaks indicate stronger periodic signal from the graphene lattice")
    pdf.savefig(fig, bbox_inches='tight', facecolor=BG); plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # PAGE 6 — Summary table
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(12, 5), facecolor=BG)
    hdr(fig, "Summary  ·  Stage 1 Complete  ·  Output Ready for Stage 2")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.78])
    ax.axis('off')

    imp = single.std() / bm3d.std()
    rows = [
        ['Frame',              'Raw SNR', 'Role',         'std after correction'],
        ['raw_19.tif',         '2.51',    'Window',       '—'],
        ['raw_20.tif',         '2.35',    'Window',       '—'],
        ['raw_21.tif  ★',      '2.67',    'TARGET',       f'{single.std():.4f}'],
        ['raw_22.tif',         '2.57',    'Window',       '—'],
        ['raw_23.tif',         '2.40',    'Window',       '—'],
        ['5-frame avg + BM3D', '—',       '→ Stage 2 in', f'{bm3d.std():.4f}  ({imp:.1f}× noise reduction)'],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)

    for j in range(4):
        tbl[0,j].set_facecolor('#1c2a3a')
        tbl[0,j].set_text_props(color=BLUE, fontweight='bold')
    for r in range(1, 7):
        bg = PANEL if r % 2 == 0 else BG
        for j in range(4):
            tbl[r,j].set_facecolor(bg)
            tbl[r,j].set_text_props(color=TXT)
    for j in range(4):           # target row
        tbl[3,j].set_facecolor('#2a2000')
        tbl[3,j].set_text_props(color=GOLD, fontweight='bold')
    for j in range(4):           # result row
        tbl[6,j].set_facecolor('#0d2a0d')
        tbl[6,j].set_text_props(color=GRN, fontweight='bold')
    for (r,j), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER); cell.set_linewidth(0.6)

    pdf.savefig(fig, bbox_inches='tight', facecolor=BG); plt.close(fig)

print(f"Saved → {OUTPUT}")
