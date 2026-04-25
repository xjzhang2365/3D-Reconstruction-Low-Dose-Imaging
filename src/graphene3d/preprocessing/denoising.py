"""
denoising.py
============
Three denoising methods benchmarked in the paper:

  1. BM3D       — Block-Matching 3D filtering          [chosen for production]
  2. K-SVD      — Dictionary learning + sparse coding
  3. U-Net      — Convolutional neural network

Paper reference:
  "Systematically compared three denoising methods: BM3D, Dictionary Learning
   (K-SVD), CNN (U-Net) ... Selected optimal method for production deployment
   based on quantitative metrics (PSNR, SSIM, speed)."

Each method exposes the same interface:
    denoised = method.denoise(noisy_frame)

so they can be swapped transparently in the pipeline.

Dependencies
------------
    pip install bm3d scikit-learn scikit-image torch torchvision
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Shared normalisation helpers
# ---------------------------------------------------------------------------

def _to_float01(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalise image to [0, 1] and return (normalised, min, max)."""
    mn, mx = image.min(), image.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.float64), mn, mx
    return (image.astype(np.float64) - mn) / (mx - mn), mn, mx


def _from_float01(image: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """Restore original intensity scale from [0, 1]."""
    return image * (mx - mn) + mn


# ---------------------------------------------------------------------------
# Method 1: BM3D
# ---------------------------------------------------------------------------

class BM3DDenoiser:
    """Block-Matching 3D (BM3D) denoising.

    BM3D is the standard baseline for Gaussian noise removal and was chosen
    as the production method in our pipeline due to its excellent quality /
    speed trade-off on TEM images.

    How BM3D works (two-pass algorithm)
    ------------------------------------
    Pass 1 — Hard-thresholding estimate:
      1. For each reference patch in the image, find similar patches across
         the entire image (block matching) and stack them into a 3D group.
      2. Apply a 3D transform (2D DCT spatially + 1D Haar wavelet along the
         similarity dimension).
      3. Hard-threshold the transform coefficients (zero out small values).
      4. Inverse transform → each patch gives a denoised estimate.
      5. Aggregate overlapping patches by weighted averaging.

    Pass 2 — Wiener filtering:
      1. Use the Pass 1 estimate to build a Wiener filter in transform domain.
      2. Repeat block matching and 3D grouping on the *noisy* image.
      3. Apply the Wiener filter (soft, signal-adaptive thresholding).
      4. Aggregate → final estimate.

    Why this works for TEM
    ----------------------
    TEM images of periodic materials (graphene lattice) contain many near-
    identical patches — exactly what block matching exploits.  The grouped
    patches add a redundancy dimension that makes noise easy to separate
    from signal in the transform domain.

    Parameters
    ----------
    sigma_psd : float
        Noise standard deviation estimate.  If None (default), estimated
        automatically from the image using the median absolute deviation of
        high-frequency wavelet coefficients.
    """

    def __init__(self, sigma_psd: float | None = None):
        try:
            import bm3d as _bm3d
            self._bm3d = _bm3d
        except ImportError:
            raise ImportError("Install bm3d: pip install bm3d")
        self.sigma_psd = sigma_psd

    def _estimate_sigma(self, image: np.ndarray) -> float:
        """Estimate noise σ from high-frequency wavelet coefficients (MAD).

        This is the standard robust noise estimator (Donoho & Johnstone 1994):
          σ̂ = MAD(detail coefficients) / 0.6745
        where 0.6745 = Φ⁻¹(0.75) for a standard normal distribution.
        """
        from scipy.signal import convolve2d
        # High-pass filter to isolate noise
        kernel = np.array([[-1, 2, -1],
                           [ 2,-4,  2],
                           [-1, 2, -1]], dtype=np.float64) / 8.0
        detail = convolve2d(image, kernel, mode='valid')
        mad = np.median(np.abs(detail - np.median(detail)))
        return mad / 0.6745

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Denoise a single TEM frame.

        Parameters
        ----------
        frame : np.ndarray
            2-D float array (any intensity range).

        Returns
        -------
        np.ndarray
            Denoised frame, same shape and intensity scale as input.
        """
        norm, mn, mx = _to_float01(frame)

        sigma = self.sigma_psd if self.sigma_psd is not None \
                else self._estimate_sigma(norm)

        stage_arg = self._bm3d.BM3DStages.HARD_THRESHOLDING
        denoised_norm = self._bm3d.bm3d(
            norm,
            sigma_psd=sigma,
            profile='np',
            stage_arg=stage_arg,
        )
        denoised_norm = np.clip(denoised_norm, 0.0, 1.0)

        return _from_float01(denoised_norm, mn, mx)


# ---------------------------------------------------------------------------
# Method 2: K-SVD Dictionary Learning
# ---------------------------------------------------------------------------

class KSVDDenoiser:
    """Dictionary learning denoising via K-SVD + OMP sparse coding.

    K-SVD learns an overcomplete dictionary D from the noisy image patches
    themselves, then represents each patch as a sparse linear combination of
    dictionary atoms.  The denoised image is reconstructed from these sparse
    codes.

    How K-SVD works
    ---------------
    Given a set of noisy patches Y = [y₁, y₂, ..., yₙ], K-SVD solves:

        min_{D, X}  ‖Y - DX‖²_F   subject to  ‖xᵢ‖₀ ≤ T  ∀i

    where:
      - D  ∈ ℝ^{patch_size² × n_atoms}  is the dictionary
      - X  ∈ ℝ^{n_atoms × n_patches}   holds the sparse codes
      - T  is the sparsity constraint (max non-zero coefficients per patch)

    Algorithm (alternating minimisation):
      1. Sparse coding step: fix D, solve for X using OMP (greedy pursuit).
      2. Dictionary update step: for each atom dₖ, update it via SVD of the
         error matrix restricted to patches that use dₖ.
         (This is the "K-SVD" step — SVD of a submatrix.)
      3. Repeat until convergence.

    Python implementation note
    --------------------------
    scikit-learn's MiniBatchDictionaryLearning implements a closely related
    algorithm (stochastic approximation).  True K-SVD is replicated here
    using sklearn's DictionaryLearning with the 'lars' (OMP) transform method.

    Why this works for TEM
    ----------------------
    TEM graphene images have highly structured patches (hexagonal lattice).
    A learned dictionary adapts to this structure, creating atoms that match
    the local lattice patterns.  Sparse coding then separates signal (few
    large coefficients) from noise (many small random coefficients).

    Parameters
    ----------
    patch_size : int
        Side length of square patches in pixels. Default 8.
    n_atoms : int
        Number of dictionary atoms. Should be > patch_size² (overcomplete).
        Default 256.
    n_iter : int
        Dictionary learning iterations. Default 50.
    sparsity : int
        Maximum non-zero coefficients per patch (OMP constraint). Default 10.
    stride : int
        Patch extraction stride (1 = fully overlapping). Default 1.
    """

    def __init__(
        self,
        patch_size: int = 8,
        n_atoms: int = 256,
        n_iter: int = 50,
        sparsity: int = 10,
        stride: int = 1,
    ):
        self.patch_size = patch_size
        self.n_atoms    = n_atoms
        self.n_iter     = n_iter
        self.sparsity   = sparsity
        self.stride     = stride

    def _extract_patches(self, image: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Extract all overlapping patches from image.

        Returns
        -------
        patches : np.ndarray
            Shape (n_patches, patch_size²).
        grid_shape : tuple
            (n_rows, n_cols) of patch grid, needed for reconstruction.
        """
        H, W = image.shape
        p, s = self.patch_size, self.stride
        rows = range(0, H - p + 1, s)
        cols = range(0, W - p + 1, s)
        patches = np.array([
            image[r:r+p, c:c+p].ravel()
            for r in rows
            for c in cols
        ])
        return patches, (len(list(rows)), len(list(cols)))

    def _reconstruct_from_patches(
        self,
        patches: np.ndarray,
        image_shape: tuple,
        grid_shape: tuple,
    ) -> np.ndarray:
        """Average overlapping reconstructed patches back into an image."""
        H, W = image_shape
        p, s = self.patch_size, self.stride
        rows = list(range(0, H - p + 1, s))
        cols = list(range(0, W - p + 1, s))

        accum  = np.zeros((H, W), dtype=np.float64)
        counts = np.zeros((H, W), dtype=np.float64)

        idx = 0
        for r in rows:
            for c in cols:
                accum[r:r+p, c:c+p]  += patches[idx].reshape(p, p)
                counts[r:r+p, c:c+p] += 1.0
                idx += 1

        counts[counts == 0] = 1.0   # avoid division by zero at borders
        return accum / counts

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Denoise a single TEM frame using dictionary learning.

        Parameters
        ----------
        frame : np.ndarray
            2-D float array.

        Returns
        -------
        np.ndarray
            Denoised frame, same shape and intensity scale as input.
        """
        from sklearn.decomposition import DictionaryLearning
        from sklearn.linear_model import orthogonal_mp_gram

        norm, mn, mx = _to_float01(frame)
        patches, grid_shape = self._extract_patches(norm)

        # Step 1: Learn dictionary from noisy patches
        dl = DictionaryLearning(
            n_components=self.n_atoms,
            max_iter=self.n_iter,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=self.sparsity,
            fit_algorithm='cd',         # coordinate descent (fast)
            random_state=42,
            verbose=False,
        )
        dl.fit(patches)
        D = dl.components_             # shape (n_atoms, patch_size²)

        # Step 2: Sparse-code each patch using OMP
        # Gram matrix G = D Dᵀ avoids recomputing per patch
        G = D @ D.T
        Dy = D @ patches.T             # shape (n_atoms, n_patches)
        codes = orthogonal_mp_gram(
            G, Dy,
            n_nonzero_coefs=self.sparsity,
        ).T                            # shape (n_patches, n_atoms)

        # Step 3: Reconstruct patches from sparse codes
        clean_patches = codes @ D      # shape (n_patches, patch_size²)

        # Step 4: Aggregate overlapping patches
        denoised_norm = self._reconstruct_from_patches(
            clean_patches, norm.shape, grid_shape
        )
        denoised_norm = np.clip(denoised_norm, 0.0, 1.0)

        return _from_float01(denoised_norm, mn, mx)


# ---------------------------------------------------------------------------
# Method 3: U-Net CNN
# ---------------------------------------------------------------------------

class UNetDenoiser:
    """U-Net convolutional neural network denoiser.

    Architecture
    ------------
    The U-Net (Ronneberger et al. 2015) is an encoder-decoder network with
    skip connections.  For denoising, the input and output are both single-
    channel images of the same size (no segmentation mask).

    Architecture used here (lightweight version for TEM patches):

        Input (1, H, W)
           │
        Encoder:
          Conv(1→32) → ReLU → Conv(32→32) → MaxPool  [64ch, H/2]
          Conv(32→64) → ReLU → Conv(64→64) → MaxPool  [128ch, H/4]
           │
        Bottleneck:
          Conv(64→128) → ReLU → Conv(128→128)
           │
        Decoder (with skip connections from encoder):
          Upsample → Conv(128+64→64) → ReLU → Conv(64→64)
          Upsample → Conv(64+32→32)  → ReLU → Conv(32→32)
           │
        Output:
          Conv(32→1)   [residual learning: predict noise, subtract from input]

    Residual (noise) learning
    -------------------------
    Rather than predicting the clean image directly, the network predicts the
    noise: output = input - network(input).  This is called DnCNN-style
    residual learning.  It converges faster because the residual (noise) has
    smaller magnitude than the signal.

    Training strategy
    -----------------
    For a real dataset, train on (noisy, clean) pairs generated by:
      - Simulated TEM images at known dose → apply Poisson noise → noisy
      - Original simulated image → clean
    The pre-trained weights provided here are a placeholder.  Load real
    weights with: denoiser.load_weights('path/to/weights.pth')

    Parameters
    ----------
    pretrained_path : str or None
        Path to saved model weights (.pth file). If None, random weights
        are used (for demonstration / training from scratch).
    device : str
        'cuda' or 'cpu'. Auto-detected if None.
    """

    def __init__(
        self,
        pretrained_path: str | None = None,
        device: str | None = None,
    ):
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn    = nn
        except ImportError:
            raise ImportError("Install PyTorch: pip install torch")

        if device is None:
            self.device = 'cuda' if self._torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = self._build_unet().to(self.device)
        self.model.eval()

        if pretrained_path is not None:
            self.load_weights(pretrained_path)

    def _build_unet(self):
        """Construct the U-Net architecture."""
        import torch.nn as nn

        def double_conv(in_ch, out_ch):
            """Two Conv→BN→ReLU blocks (standard U-Net building block)."""
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = double_conv(1, 32)
                self.enc2 = double_conv(32, 64)
                self.pool  = nn.MaxPool2d(2)
                # Bottleneck
                self.bottleneck = double_conv(64, 128)
                # Decoder
                self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.dec2  = double_conv(128, 64)   # 128 = 64 skip + 64 up
                self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                self.dec1  = double_conv(64, 32)    # 64  = 32 skip + 32 up
                # Output: predict residual (noise)
                self.out   = nn.Conv2d(32, 1, kernel_size=1)

            def forward(self, x):
                # Encoder path
                e1 = self.enc1(x)                   # (B, 32, H, W)
                e2 = self.enc2(self.pool(e1))        # (B, 64, H/2, W/2)
                # Bottleneck
                b  = self.bottleneck(self.pool(e2))  # (B, 128, H/4, W/4)
                # Decoder path with skip connections
                d2 = self.dec2(
                    self._cat(self.up2(b), e2)       # (B, 128, H/2, W/2)
                )
                d1 = self.dec1(
                    self._cat(self.up1(d2), e1)      # (B, 64, H, W)
                )
                noise = self.out(d1)                  # (B, 1, H, W)
                return x - noise                      # residual learning

            @staticmethod
            def _cat(a, b):
                """Concatenate along channel dim, handling size mismatch."""
                import torch
                # Crop b to match a if sizes differ (from odd-dim pooling)
                if a.shape != b.shape:
                    b = b[:, :, :a.shape[2], :a.shape[3]]
                return torch.cat([a, b], dim=1)

        return UNet()

    def load_weights(self, path: str) -> None:
        """Load pre-trained weights from a .pth file."""
        self.model.load_state_dict(
            self._torch.load(path, map_location=self.device)
        )
        self.model.eval()
        print(f"[UNetDenoiser] Loaded weights from {path}")

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Denoise a single TEM frame using the U-Net.

        Parameters
        ----------
        frame : np.ndarray
            2-D float array (any size; padding applied if needed).

        Returns
        -------
        np.ndarray
            Denoised frame, same shape and intensity scale as input.
        """
        import torch

        norm, mn, mx = _to_float01(frame)
        H, W = norm.shape

        # Pad to multiple of 4 (required by 2× pooling layers)
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            norm = np.pad(norm, ((0, pad_h), (0, pad_w)), mode='reflect')

        # (H, W) → (1, 1, H, W) tensor
        tensor = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        denoised = out.squeeze().cpu().numpy()
        denoised = denoised[:H, :W]                   # remove padding
        denoised = np.clip(denoised, 0.0, 1.0)

        return _from_float01(denoised, mn, mx)

    def train_on_pairs(
        self,
        noisy_frames: np.ndarray,
        clean_frames: np.ndarray,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        save_path: str | None = None,
    ) -> list[float]:
        """Train the U-Net on (noisy, clean) image pairs.

        Parameters
        ----------
        noisy_frames : np.ndarray
            Shape (N, H, W) — simulated low-dose TEM images.
        clean_frames : np.ndarray
            Shape (N, H, W) — corresponding high-dose / simulated ground truth.
        n_epochs : int
            Training epochs.
        learning_rate : float
            Adam optimiser learning rate.
        batch_size : int
            Mini-batch size.
        save_path : str or None
            If given, save model weights after training.

        Returns
        -------
        loss_history : list[float]
            MSE loss per epoch.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.model.train()

        # Normalise each frame independently
        X = np.stack([_to_float01(f)[0] for f in noisy_frames])
        Y = np.stack([_to_float01(f)[0] for f in clean_frames])

        X_t = torch.from_numpy(X).float().unsqueeze(1)   # (N,1,H,W)
        Y_t = torch.from_numpy(Y).float().unsqueeze(1)

        dataset    = TensorDataset(X_t, Y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimiser  = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion  = nn.MSELoss()
        loss_history = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimiser.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(x_batch)

            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}  MSE loss: {epoch_loss:.6f}")

        self.model.eval()
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"[UNetDenoiser] Weights saved to {save_path}")

        return loss_history


# ---------------------------------------------------------------------------
# Benchmarking utility
# ---------------------------------------------------------------------------

def benchmark_denoisers(
    noisy: np.ndarray,
    reference: np.ndarray,
    denoisers: dict,
) -> dict:
    """Evaluate PSNR and SSIM for each denoiser on a single frame.

    Parameters
    ----------
    noisy : np.ndarray
        Noisy input frame (2-D).
    reference : np.ndarray
        Ground-truth clean frame (2-D), same shape.
    denoisers : dict
        Mapping of name → denoiser instance (must implement .denoise()).

    Returns
    -------
    dict
        {name: {'psnr': float, 'ssim': float, 'time_s': float}}

    Example
    -------
    >>> results = benchmark_denoisers(noisy, clean, {
    ...     'BM3D':   BM3DDenoiser(),
    ...     'K-SVD':  KSVDDenoiser(),
    ...     'U-Net':  UNetDenoiser(),
    ... })
    >>> for name, m in results.items():
    ...     print(f"{name}: PSNR={m['psnr']:.1f} dB, SSIM={m['ssim']:.3f}")
    """
    import time

    results = {}
    data_range = reference.max() - reference.min()

    for name, denoiser in denoisers.items():
        t0 = time.perf_counter()
        denoised = denoiser.denoise(noisy)
        elapsed = time.perf_counter() - t0

        p = psnr(reference, denoised, data_range=data_range)
        s = ssim(reference, denoised, data_range=data_range)

        results[name] = {'psnr': p, 'ssim': s, 'time_s': elapsed}
        print(f"  {name:<8}  PSNR={p:5.2f} dB  SSIM={s:.4f}  t={elapsed:.2f}s")

    return results
