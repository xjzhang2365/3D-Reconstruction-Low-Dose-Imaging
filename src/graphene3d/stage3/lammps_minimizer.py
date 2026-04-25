"""
Thin wrapper around LAMMPSRelaxationAdapter for the SA inner loop.

Provides a clean .relax(positions_ang) -> positions_ang interface
with per-call failure handling. Reuses the existing adapter's data-file
writer, subprocess launcher, and dump parser — no format duplication.

Anisotropic displacement cap: xy_cap and z_cap are applied separately
after LAMMPS runs. This prevents the large LAMMPS xy-flattening motion
(which hurts chi2) while still allowing z bond-length corrections.
If only max_displacement_angstrom is given, it is used isotropically.
"""

import time
from pathlib import Path

import numpy as np

from graphene3d.stage3.sa_refine import LAMMPSRelaxationAdapter, MDConfig


class LammpsMinimizer:
    """Fast LAMMPS energy minimization adapter for SA inner loop."""

    def __init__(
        self,
        lammps_executable: Path,
        potential_file: Path,
        working_dir: Path,
        timeout_seconds: float = 15.0,
        boundary: str = "p p f",
        etol: float = 1e-8,
        ftol: float = 1e-10,
        maxiter: int = 5000,
        maxeval: int = 50000,
        max_displacement_angstrom: float = 0.5,
        xy_cap_angstrom: float | None = None,
        z_cap_angstrom: float | None = None,
    ):
        """
        Parameters
        ----------
        max_displacement_angstrom
            Isotropic per-atom displacement cap applied by the adapter.
            Ignored when xy_cap_angstrom and z_cap_angstrom are set
            (anisotropic mode takes over).
        xy_cap_angstrom
            Per-atom xy displacement cap applied post-hoc after LAMMPS.
            Prevents LAMMPS from flattening graphene (large xy motion)
            while preserving its z bond-length corrections.
        z_cap_angstrom
            Per-atom z displacement cap applied post-hoc after LAMMPS.
        """
        self.exe = Path(lammps_executable)
        self.pot = Path(potential_file)
        self.workdir = Path(working_dir)
        self.timeout = timeout_seconds
        self.boundary = boundary
        self.etol = etol
        self.ftol = ftol
        self.maxiter = maxiter
        self.maxeval = maxeval
        self.xy_cap = xy_cap_angstrom
        self.z_cap  = z_cap_angstrom

        # If anisotropic caps are set, let LAMMPS run uncapped internally;
        # we apply our own post-hoc anisotropic cap in relax().
        # Otherwise fall back to the adapter's isotropic cap.
        aniso = (xy_cap_angstrom is not None) or (z_cap_angstrom is not None)
        adapter_cap = None if aniso else max_displacement_angstrom

        self.workdir.mkdir(parents=True, exist_ok=True)
        self._adapter = LAMMPSRelaxationAdapter()
        self._call_count = 0
        self._fail_count = 0
        self._total_time = 0.0
        self._first_error: str | None = None

        self._md_config = MDConfig(
            enabled=True,
            backend="lammps",
            lammps_execute=True,
            lammps_executable=str(self.exe),
            lammps_working_dir=str(self.workdir),
            lammps_data_filename="structure.data",
            lammps_input_filename="minimize.in",
            lammps_dump_filename="relaxed.dump",
            lammps_log_filename="lammps.log",
            lammps_units="metal",
            lammps_atom_style="atomic",
            lammps_boundary=self.boundary,
            lammps_atom_mass=12.011,
            lammps_pair_style="tersoff",
            lammps_pair_coeff=f"* * {self.pot.name} C",
            lammps_potential_file=str(self.pot),
            lammps_box_padding_angstrom=5.0,
            lammps_minimize_etol=self.etol,
            lammps_minimize_ftol=self.ftol,
            lammps_minimize_maxiter=self.maxiter,
            lammps_minimize_maxeval=self.maxeval,
            lammps_timeout_seconds=int(self.timeout),
            max_displacement_angstrom=adapter_cap,
        )

    def _apply_anisotropic_cap(
        self, raw: np.ndarray, origin: np.ndarray
    ) -> np.ndarray:
        """Apply separate xy and z displacement caps post-hoc."""
        delta = raw - origin
        result = delta.copy()

        if self.xy_cap is not None:
            xy_disp = np.linalg.norm(delta[:, :2], axis=1)
            too_large = xy_disp > self.xy_cap
            if too_large.any():
                scale = np.ones(len(xy_disp))
                scale[too_large] = self.xy_cap / np.maximum(
                    xy_disp[too_large], 1e-12
                )
                result[:, :2] = delta[:, :2] * scale[:, None]

        if self.z_cap is not None:
            z_disp = np.abs(delta[:, 2])
            too_large = z_disp > self.z_cap
            if too_large.any():
                scale = np.ones(len(z_disp))
                scale[too_large] = self.z_cap / np.maximum(
                    z_disp[too_large], 1e-12
                )
                result[:, 2] = delta[:, 2] * scale

        return origin + result

    def relax(self, positions_ang: np.ndarray) -> np.ndarray:
        """
        Run LAMMPS energy minimization.
        Returns relaxed (N,3) positions, or input positions on any failure.
        """
        self._call_count += 1
        t0 = time.time()

        try:
            from scipy.spatial.distance import pdist
            if len(positions_ang) > 1:
                min_dist = pdist(positions_ang).min()
                if min_dist < 0.5:
                    self._fail_count += 1
                    return positions_ang

            raw_relaxed = self._adapter.relax(positions_ang, self._md_config)

            if not np.all(np.isfinite(raw_relaxed)):
                self._fail_count += 1
                return positions_ang

            # Apply anisotropic cap when configured
            if self.xy_cap is not None or self.z_cap is not None:
                relaxed = self._apply_anisotropic_cap(raw_relaxed, positions_ang)
            else:
                relaxed = raw_relaxed

            return relaxed

        except Exception as exc:
            self._fail_count += 1
            if self._first_error is None:
                self._first_error = str(exc)[:500]
            return positions_ang

        finally:
            self._total_time += time.time() - t0

    @property
    def mean_call_time(self) -> float:
        n = self._call_count - self._fail_count
        return self._total_time / max(n, 1)

    def summary(self) -> dict:
        return {
            "call_count": self._call_count,
            "fail_count": self._fail_count,
            "fail_rate": self._fail_count / max(self._call_count, 1),
            "total_time_s": self._total_time,
            "mean_call_time_s": self.mean_call_time,
            "first_error": self._first_error,
        }
