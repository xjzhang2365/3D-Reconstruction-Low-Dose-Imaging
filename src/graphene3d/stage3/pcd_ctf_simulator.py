import numpy as np


class PcdCtfSimulator:
    """
    Physics-based TEM simulator: projected atomic potential + CTF.
    Z-sensitivity comes from the depth-of-focus envelope — atoms at
    different z heights produce genuinely different projected images.
    No z-score normalization. Absolute z positions matter.
    """

    def __init__(
        self,
        pixel_size_ang: float,
        image_shape: tuple,
        voltage_kV: float = 80.0,
        defocus_ang: float = -80.0,
        Cs_mm: float = 0.001,
        dose: float = 8000.0,
        beam_convergence_mrad: float = 0.5,
        add_noise: bool = False,
    ):
        self.pixel_size = pixel_size_ang
        self.shape = image_shape
        self.defocus = defocus_ang
        self.Cs_ang = Cs_mm * 1e7
        self.dose = dose
        self.alpha = beam_convergence_mrad * 1e-3
        self.add_noise = add_noise

        # Relativistic electron wavelength (Angstroms)
        V = voltage_kV * 1000.0
        self.lambda_ang = 12.2643 / np.sqrt(V * (1.0 + V / (2.0 * 511000.0)))

        # Pre-compute frequency grid
        H, W = image_shape
        fy = np.fft.fftfreq(H, d=pixel_size_ang)
        fx = np.fft.fftfreq(W, d=pixel_size_ang)
        self.FX, self.FY = np.meshgrid(fx, fy)
        self.K2 = self.FX**2 + self.FY**2

        # Pre-compute CTF (fixed for all simulate() calls)
        self._ctf = self._build_ctf()

    def _build_ctf(self) -> np.ndarray:
        lam = self.lambda_ang
        chi = (np.pi * lam * self.defocus * self.K2 +
               0.5 * np.pi * self.Cs_ang * lam**3 * self.K2**2)
        ctf = -2.0 * np.sin(chi)

        # Spatial coherence envelope
        Es = np.exp(-(np.pi * self.alpha)**2 *
                    (self.defocus + self.Cs_ang * lam**2 * self.K2)**2 * self.K2)

        # Temporal coherence envelope
        dE_over_E = 1.0 / (80.0 * 1000.0)
        Cc_ang = 1.0e7
        Et = np.exp(-0.5 * (np.pi * lam * Cc_ang * dE_over_E)**2 * self.K2**2)

        return ctf * Es * Et

    def _projected_potential(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized projected potential with z-dependent Gaussian broadening.
        Each atom's effective defocus = global defocus + local z offset relative
        to the sheet midplane. This encodes corrugation amplitude, not absolute z.
        """
        H, W = self.shape
        V_proj = np.zeros((H, W), dtype=np.float64)

        x_px = positions[:, 0] / self.pixel_size
        y_px = positions[:, 1] / self.pixel_size

        # Center z relative to sheet midplane so sigma_eff encodes
        # corrugation amplitude, not absolute position.
        # Atoms above midplane broaden upward, below broaden downward —
        # the CTF sees different effective defocus for each atom.
        z_ang = positions[:, 2]
        z_centered = z_ang - np.mean(z_ang)   # corrugation relative to midplane

        sigma_base = 0.5  # Angstroms
        sigma_base_px = sigma_base / self.pixel_size

        # Effective defocus per atom = global defocus + local z offset
        # Positive z (above midplane) -> less underfocus -> narrower Gaussian
        # Negative z (below midplane) -> more underfocus -> broader Gaussian
        defocus_per_atom = self.defocus - z_centered   # signed, not |z|

        # Depth-of-focus broadening from effective defocus
        sigma_extra_ang = np.abs(defocus_per_atom) * self.lambda_ang / (
            4.0 * np.pi * sigma_base
        )
        sigma_eff_px = sigma_base_px + sigma_extra_ang / self.pixel_size
        sigma_eff_px = np.maximum(sigma_eff_px, 0.3)

        # Amplitude: atoms with more defocus spread more -> lower peak
        amplitude = 1.0 / (2.0 * np.pi * sigma_eff_px**2)

        for i in range(len(positions)):
            xi, yi = x_px[i], y_px[i]
            sig = sigma_eff_px[i]
            amp = amplitude[i]

            r = int(np.ceil(4.0 * sig)) + 1
            ix0 = max(0, int(xi) - r)
            ix1 = min(W, int(xi) + r + 1)
            iy0 = max(0, int(yi) - r)
            iy1 = min(H, int(yi) + r + 1)

            if ix0 >= ix1 or iy0 >= iy1:
                continue

            lx = np.arange(ix0, ix1) - xi
            ly = np.arange(iy0, iy1) - yi
            LX, LY = np.meshgrid(lx, ly)
            V_proj[iy0:iy1, ix0:ix1] += amp * np.exp(
                -(LX**2 + LY**2) / (2.0 * sig**2)
            )

        return V_proj.astype(np.float32)

    def simulate(self, positions_ang: np.ndarray) -> np.ndarray:
        """
        positions_ang: (N, 3) Angstroms — x, y, z columns.
        Returns: (H, W) float32 in electrons/pixel units.
        No min/max rescaling — that kills z-sensitivity by absorbing
        amplitude changes from defocus variation.
        """
        V_proj = self._projected_potential(positions_ang)
        V_k = np.fft.fft2(V_proj)
        I_ctf = np.real(np.fft.ifft2(self._ctf * V_k))

        # Scale by dose so units are electrons/pixel (physically meaningful).
        I_out = I_ctf * self.dose * self.pixel_size**2

        if self.add_noise:
            I_out = np.random.poisson(
                np.maximum(I_out, 0)
            ).astype(np.float32)

        return I_out.astype(np.float32)

    def chi2(self, positions_ang: np.ndarray,
             target_image: np.ndarray) -> float:
        I_sim = self.simulate(positions_ang)

        # Normalize both to zero-mean unit-variance before comparing.
        # This makes chi2 sensitive to contrast pattern (which encodes z
        # positions) rather than absolute intensity scale. Applied to the
        # whole image — completely different from the old per-atom z-score bug.
        def znorm(x):
            mu, sigma = x.mean(), x.std()
            if sigma < 1e-10:
                return x - mu
            return (x - mu) / sigma

        I_s = znorm(I_sim)
        I_t = znorm(target_image.astype(np.float32))
        return float(np.mean((I_s - I_t)**2))


class AbtemSimulator:
    """
    Physics-accurate TEM simulator using abTEM multislice.
    This correctly encodes z positions through phase contrast —
    atoms at different heights contribute phase shifts that
    produce detectable contrast differences in the image.
    """

    def __init__(
        self,
        pixel_size_ang: float,
        image_shape: tuple,
        voltage_kV: float = 80.0,
        defocus_ang: float = -80.0,
        Cs_mm: float = 0.001,
        dose: float = 8000.0,
        slice_thickness: float = 1.0,
        add_noise: bool = False,
    ):
        self.pixel_size = pixel_size_ang
        self.image_shape = image_shape
        self.voltage_kV = voltage_kV
        self.defocus_ang = defocus_ang
        self.Cs_mm = Cs_mm
        self.dose = dose
        self.slice_thickness = slice_thickness
        self.add_noise = add_noise

        # Compute field of view from image shape and pixel size
        self.fov_x = image_shape[1] * pixel_size_ang  # Angstroms
        self.fov_y = image_shape[0] * pixel_size_ang

    def _positions_to_ase(self, positions_ang: np.ndarray):
        """Convert (N,3) position array to ASE Atoms object."""
        from ase import Atoms

        # Add margin around atom cloud for periodic boundary
        margin = 5.0  # Angstroms
        x_min, y_min = positions_ang[:, 0].min(), positions_ang[:, 1].min()

        # Shift positions so minimum is at margin
        pos_shifted = positions_ang.copy()
        pos_shifted[:, 0] -= x_min - margin
        pos_shifted[:, 1] -= y_min - margin

        # Cell: fit to field of view, 20 Å vacuum in z
        cell_x = max(self.fov_x, pos_shifted[:, 0].max() + margin)
        cell_y = max(self.fov_y, pos_shifted[:, 1].max() + margin)
        cell_z = 20.0

        # Place atoms at a fixed z reference in the cell (not recentered
        # per call). This preserves translation sensitivity.
        # We anchor the INITIAL z_mean once, then keep it fixed.
        if not hasattr(self, '_z_anchor'):
            self._z_anchor = pos_shifted[:, 2].mean()

        z_center = cell_z / 2.0
        pos_shifted[:, 2] += z_center - self._z_anchor

        atoms = Atoms(
            symbols=['C'] * len(positions_ang),
            positions=pos_shifted,
            cell=[cell_x, cell_y, cell_z],
            pbc=[True, True, False],
        )
        return atoms

    def simulate(self, positions_ang: np.ndarray) -> np.ndarray:
        """
        positions_ang: (N, 3) Angstroms
        Returns: (H, W) float32, z-score normalized for chi2 comparison
        """
        import abtem
        from abtem import CTF

        atoms = self._positions_to_ase(positions_ang)

        potential = abtem.Potential(
            atoms,
            parametrization='lobato',
            slice_thickness=self.slice_thickness,
            projection='finite',
        )

        ctf = CTF(
            defocus=self.defocus_ang,
            Cs=self.Cs_mm * 1e7,
            energy=self.voltage_kV * 1e3,
        )

        wave = abtem.PlaneWave(
            energy=self.voltage_kV * 1e3,
            sampling=self.pixel_size,
        )

        exit_wave = wave.multislice(potential)
        image_wave = exit_wave.apply_ctf(ctf)
        intensity = image_wave.intensity().compute()

        # Extract array and crop/resize to match target image_shape
        arr = intensity.array
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float32)

        # Crop or pad to match required image_shape
        H, W = self.image_shape
        arr = self._match_shape(arr, H, W)

        if self.add_noise:
            counts = np.maximum(arr, 0) * self.dose * self.pixel_size**2
            arr = np.random.poisson(counts).astype(np.float32)

        return arr

    def _match_shape(self, arr: np.ndarray, H: int, W: int) -> np.ndarray:
        """Crop center or pad with mean to match target shape."""
        ah, aw = arr.shape
        # Crop center
        if ah >= H and aw >= W:
            r0 = (ah - H) // 2
            c0 = (aw - W) // 2
            return arr[r0:r0+H, c0:c0+W]
        # Pad if smaller
        out = np.full((H, W), arr.mean(), dtype=np.float32)
        r0 = (H - ah) // 2
        c0 = (W - aw) // 2
        out[r0:r0+ah, c0:c0+aw] = arr
        return out

    def chi2(self, positions_ang: np.ndarray,
             target_image: np.ndarray) -> float:
        """Z-score normalized chi2 — sensitive to contrast pattern."""
        I_sim = self.simulate(positions_ang)

        def znorm(x):
            mu, sigma = x.mean(), x.std()
            return (x - mu) / (sigma + 1e-10)

        return float(np.mean((znorm(I_sim) -
                              znorm(target_image.astype(np.float32)))**2))
