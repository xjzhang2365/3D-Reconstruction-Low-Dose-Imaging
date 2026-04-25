"""
sa_refine.py
============
Stage 3 simulated annealing refinement skeleton.

This module starts from the Stage 2 SA handoff file and runs a basic
Metropolis simulated-annealing loop over atomic coordinates.  It intentionally
does not implement full molecular dynamics or LAMMPS integration.

The objective interface is deliberately small:

    objective(coords_angstrom) -> scalar energy

By default the objective compares a target TEM image with an adapter-produced
simulated image.  The included GaussianProjectionSimulator is a fast runnable
debug proxy.  Use AbTEMSimulatorAdapter for the Python-only physics simulator
path once abTEM and ASE are installed.  A standalone LAMMPS relaxation adapter
exists for boundary testing, but the frozen SA baseline still does not invoke
real MD by default.

Current frozen pre-MD baseline:
    make_stable_stage3_baseline_config()

This baseline uses anisotropic proposals, weak z-sensitive Gaussian projection,
pre-SA sanitization, structural rejection, and a very light nearest-neighbor
structural regularizer.  It is the recommended working Stage 3 configuration
before stronger MD/LAMMPS coupling is added.

First real MD-enabled option:
    make_periodic_weak_lammps_md_config()

This opt-in preset runs real LAMMPS minimization periodically with a very small
per-atom displacement cap.  It is a structural regularization mode, not the
default image-only SA baseline.

MD/LAMMPS boundary:
    MDRelaxationAdapter.relax(coords_xyz_angstrom, md_config)

The boundary is defined now but intentionally not invoked by the frozen SA
baseline.  Future work can insert relaxation after proposal generation and
before structural rejection plus objective evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
import json
import shutil
import subprocess

import numpy as np


DEFAULT_SA_INPUT = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "stage2"
    / "validation"
    / "stage2_validation_init_sa_input.npz"
)
DEFAULT_TARGET_IMAGE = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "synthetic"
    / "img_noisy.tif"
).resolve()
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "stage3" / "runs"
DEFAULT_TUNING_DIR = Path(__file__).resolve().parents[3] / "outputs" / "stage3" / "tuning"


@dataclass
class SAConfig:
    """Parameters for the Stage 3 simulated annealing skeleton."""

    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    n_iterations: int = 1000
    step_size_xy: float = 0.03
    step_size_z: float = 0.03
    random_seed: Optional[int] = 0
    enable_structural_rejection: bool = True
    structural_min_distance_angstrom: float = 0.8
    enable_pre_sa_sanitization: bool = True
    sanitization_max_passes: int = 10
    sanitization_buffer_angstrom: float = 0.02
    enable_structural_regularization: bool = False
    structural_regularization_weight: float = 0.0
    regularization_target_nn_angstrom: float = 1.42
    regularization_neighbor_cutoff_angstrom: float = 1.9
    regularization_overlap_weight: float = 10.0
    regularization_smoothness_weight: float = 0.0

    # Image-mismatch objective.  The default simulator is a Gaussian projection
    # proxy so the SA framework is runnable before the real TEM simulator is
    # connected.  Pass a custom objective or simulator for production runs.
    target_image_path: Optional[str] = str(DEFAULT_TARGET_IMAGE)
    objective_metric: str = "normalized_mse"
    image_normalize: bool = True
    image_epsilon: float = 1e-8
    simulator_kind: str = "gaussian_projection"
    simulator_sigma_px: float = 1.2
    simulator_atom_contrast: float = 1.0
    simulator_background: float = 0.0
    simulator_z_contrast_scale: float = 0.08
    abtem_energy_ev: float = 80000.0
    abtem_sampling_angstrom: Optional[float] = None
    abtem_slice_thickness_angstrom: float = 1.0
    abtem_vacuum_angstrom: float = 5.0
    abtem_device: str = "cpu"

    output_prefix: str = "stage3_sa_refine"
    save_outputs: bool = True
    verbose: bool = True


@dataclass
class MDConfig:
    """
    Configuration placeholder for future MD/LAMMPS relaxation.

    The current Stage 3 baseline does not invoke MD.  These fields document the
    expected control surface for a later adapter while keeping today's SA
    numerical behavior unchanged.
    """

    enabled: bool = False
    backend: str = "none"
    n_steps: int = 0
    timestep_fs: float = 0.0
    temperature_k: float = 300.0
    force_field: str = ""
    # Optional per-atom cap applied after an MD adapter proposes relaxed
    # coordinates.  For LAMMPS, this lets early in-loop tests use real
    # minimization as a weak correction instead of a strong global rewrite.
    max_displacement_angstrom: Optional[float] = None
    # Optional in-loop cadence for MD relaxation.  The default of 1 preserves
    # the earlier "relax every proposal" behavior when MD is enabled.  Larger
    # values make LAMMPS a periodic structural regularization step.
    apply_every_iterations: int = 1
    fake_min_distance_angstrom: float = 0.8
    fake_target_nn_angstrom: float = 1.42
    fake_neighbor_cutoff_angstrom: float = 1.9
    fake_repulsion_strength: float = 0.5
    fake_bond_relaxation_strength: float = 0.0
    fake_max_pair_correction_angstrom: float = 0.03
    fake_max_atom_displacement_angstrom: float = 0.05

    # LAMMPS adapter settings.  Execution is opt-in via lammps_execute so the
    # frozen SA baseline and scaffold behavior remain unchanged by default.
    lammps_execute: bool = False
    lammps_executable: str = ""
    lammps_working_dir: str = "outputs/stage3/lammps_work"
    lammps_data_filename: str = "stage3_structure.data"
    lammps_input_filename: str = "minimize.in"
    lammps_dump_filename: str = "relaxed.dump"
    lammps_log_filename: str = "lammps.log"
    lammps_atom_style: str = "atomic"
    lammps_units: str = "metal"
    lammps_boundary: str = "p p p"
    lammps_atom_mass: float = 12.011
    lammps_pair_style: str = ""
    lammps_pair_coeff: str = ""
    lammps_potential_file: str = ""
    lammps_box_padding_angstrom: float = 10.0
    lammps_minimize_etol: float = 1e-8
    lammps_minimize_ftol: float = 1e-10
    lammps_minimize_maxiter: int = 1000
    lammps_minimize_maxeval: int = 10000
    lammps_timeout_seconds: int = 120


class MDRelaxationAdapter(Protocol):
    """Protocol for a future MD/LAMMPS relaxation module."""

    def relax(self,
              coords_xyz_angstrom: np.ndarray,
              md_config: MDConfig) -> np.ndarray:
        """
        Return relaxed coordinates for a proposed structure.

        Intended future placement in SA:
          proposal generation -> optional MD relax -> structural rejection
          -> image objective + structural regularization evaluation.
        """


class NoOpMDRelaxationAdapter:
    """
    Placeholder MD adapter that preserves coordinates exactly.

    This is the only MD adapter provided in the frozen pre-MD baseline.  It is
    useful for testing wiring without changing SA behavior.
    """

    def relax(self,
              coords_xyz_angstrom: np.ndarray,
              md_config: MDConfig) -> np.ndarray:
        _ = md_config
        return np.asarray(coords_xyz_angstrom, dtype=np.float64).copy()


class FakeMDRelaxationAdapter:
    """
    Lightweight pre-LAMMPS test relaxer for the MD adapter boundary.

    This is not molecular dynamics.  It applies deterministic, local coordinate
    corrections so the future SA->MD hook can be tested before a real LAMMPS
    backend exists:
      - close pairs are pushed apart along their pair direction
      - optional neighbor pairs are nudged toward a target bond length

    Use this adapter only for stress-testing the SA->MD hook under aggressive
    proposals.  The expected behavior is a structural-safety improvement that
    may trade off a small amount of image-objective progress.

    The frozen Stage 3 baseline does not call this adapter.
    """

    def __init__(self) -> None:
        self.last_summary: dict[str, Any] = {}

    def relax(self,
              coords_xyz_angstrom: np.ndarray,
              md_config: MDConfig) -> np.ndarray:
        coords = np.asarray(coords_xyz_angstrom, dtype=np.float64)
        relaxed = coords.copy()
        original = coords.copy()
        if not md_config.enabled or md_config.backend not in {"fake_md", "fake", "fake_lammps"}:
            self.last_summary = {
                "enabled": bool(md_config.enabled),
                "backend": md_config.backend,
                "applied": False,
                "reason": "MDConfig disabled or backend is not fake_md.",
            }
            return relaxed

        n_steps = max(0, int(md_config.n_steps))
        if n_steps == 0 or len(relaxed) < 2:
            self.last_summary = {
                "enabled": True,
                "backend": md_config.backend,
                "applied": False,
                "reason": "No fake-MD steps requested or fewer than two atoms.",
            }
            return relaxed

        min_distance = float(md_config.fake_min_distance_angstrom)
        target_nn = float(md_config.fake_target_nn_angstrom)
        cutoff = max(float(md_config.fake_neighbor_cutoff_angstrom), min_distance)
        repulsion_strength = max(float(md_config.fake_repulsion_strength), 0.0)
        bond_strength = max(float(md_config.fake_bond_relaxation_strength), 0.0)
        max_pair_correction = max(float(md_config.fake_max_pair_correction_angstrom), 0.0)
        max_atom_step = max(float(md_config.fake_max_atom_displacement_angstrom), 0.0)
        if md_config.max_displacement_angstrom is not None:
            max_atom_step = min(max_atom_step, max(float(md_config.max_displacement_angstrom), 0.0))

        pair_corrections = 0
        bond_corrections = 0

        for _step in range(n_steps):
            correction = np.zeros_like(relaxed)
            for i in range(len(relaxed) - 1):
                for j in range(i + 1, len(relaxed)):
                    delta = relaxed[i] - relaxed[j]
                    distance = float(np.linalg.norm(delta))
                    if distance < 1e-12:
                        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        direction = delta / distance

                    pair_shift = 0.0
                    if distance < min_distance:
                        pair_shift += repulsion_strength * (min_distance - distance)
                        pair_corrections += 1
                    elif bond_strength > 0.0 and distance <= cutoff:
                        pair_shift += bond_strength * (target_nn - distance)
                        bond_corrections += 1

                    if pair_shift == 0.0:
                        continue
                    pair_shift = float(np.clip(pair_shift, -max_pair_correction, max_pair_correction))
                    half_shift = 0.5 * pair_shift * direction
                    correction[i] += half_shift
                    correction[j] -= half_shift

            if max_atom_step > 0.0:
                norms = np.linalg.norm(correction, axis=1)
                too_large = norms > max_atom_step
                if np.any(too_large):
                    correction[too_large] *= (max_atom_step / norms[too_large])[:, None]
            relaxed += correction
            if max_atom_step > 0.0:
                total_delta = relaxed - original
                total_norms = np.linalg.norm(total_delta, axis=1)
                too_large_total = total_norms > max_atom_step
                if np.any(too_large_total):
                    relaxed[too_large_total] = (
                        original[too_large_total]
                        + total_delta[too_large_total]
                        * (max_atom_step / total_norms[too_large_total])[:, None]
                    )
            if not np.any(correction):
                break

        displacement = np.linalg.norm(relaxed - original, axis=1)
        self.last_summary = {
            "enabled": True,
            "backend": md_config.backend,
            "applied": True,
            "n_steps_requested": int(md_config.n_steps),
            "min_distance_angstrom": min_distance,
            "target_nn_angstrom": target_nn,
            "neighbor_cutoff_angstrom": cutoff,
            "repulsion_strength": repulsion_strength,
            "bond_relaxation_strength": bond_strength,
            "pair_corrections": int(pair_corrections),
            "bond_corrections": int(bond_corrections),
            "max_atom_displacement_angstrom": float(np.max(displacement)) if len(displacement) else 0.0,
            "mean_atom_displacement_angstrom": float(np.mean(displacement)) if len(displacement) else 0.0,
        }
        return relaxed


class LAMMPSRelaxationAdapter:
    """
    File/interface scaffold for future LAMMPS energy minimization.

    Chosen integration contract:
      - input: LAMMPS data file written from xyz coordinates
      - mode: energy minimization input script
      - output: dump/custom text with columns id x y z

    Execution is opt-in with MDConfig.lammps_execute=True.  With the default
    lammps_execute=False, relax() writes the data and input files, then either
    parses an existing dump/custom output or raises a clear RuntimeError.
    """

    def __init__(self) -> None:
        self.last_summary: dict[str, Any] = {}

    def _workdir(self, md_config: MDConfig) -> Path:
        return Path(md_config.lammps_working_dir)

    def data_path(self, md_config: MDConfig) -> Path:
        return self._workdir(md_config) / md_config.lammps_data_filename

    def input_path(self, md_config: MDConfig) -> Path:
        return self._workdir(md_config) / md_config.lammps_input_filename

    def dump_path(self, md_config: MDConfig) -> Path:
        return self._workdir(md_config) / md_config.lammps_dump_filename

    def _resolve_lammps_executable(self, md_config: MDConfig) -> str:
        """Resolve and validate the configured LAMMPS executable."""
        executable = md_config.lammps_executable.strip()
        if not executable:
            raise RuntimeError(
                "LAMMPS backend requested, but MDConfig.lammps_executable is empty. "
                "Set it to your LAMMPS executable path/name, for example 'lmp'."
            )
        exe_path = Path(executable)
        if exe_path.parent != Path(".") or exe_path.is_absolute():
            if not exe_path.exists():
                raise RuntimeError(
                    f"LAMMPS executable does not exist: {exe_path}. "
                    "Set MDConfig.lammps_executable to a valid executable path or command name."
                )
            return str(exe_path)
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(
                f"LAMMPS executable '{executable}' was not found on PATH. "
                "Install LAMMPS or set MDConfig.lammps_executable to the full executable path."
            )
        return resolved

    def write_lammps_data(self,
                          coords_xyz_angstrom: np.ndarray,
                          md_config: MDConfig) -> Path:
        """Write a minimal carbon-only LAMMPS data file."""
        coords = np.asarray(coords_xyz_angstrom, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N, 3), got {coords.shape}")
        workdir = self._workdir(md_config)
        workdir.mkdir(parents=True, exist_ok=True)
        data_path = self.data_path(md_config)

        if len(coords):
            mins = np.min(coords, axis=0) - float(md_config.lammps_box_padding_angstrom)
            maxs = np.max(coords, axis=0) + float(md_config.lammps_box_padding_angstrom)
        else:
            mins = np.array([-10.0, -10.0, -10.0], dtype=np.float64)
            maxs = np.array([10.0, 10.0, 10.0], dtype=np.float64)

        with open(data_path, "w", encoding="utf-8") as fh:
            fh.write("Stage 3 graphene structure for LAMMPS minimization\n\n")
            fh.write(f"{len(coords)} atoms\n")
            fh.write("1 atom types\n\n")
            fh.write(f"{mins[0]:.8f} {maxs[0]:.8f} xlo xhi\n")
            fh.write(f"{mins[1]:.8f} {maxs[1]:.8f} ylo yhi\n")
            fh.write(f"{mins[2]:.8f} {maxs[2]:.8f} zlo zhi\n\n")
            fh.write("Masses\n\n")
            fh.write(f"1 {float(md_config.lammps_atom_mass):.8f} # C\n\n")
            fh.write("Atoms # atomic\n\n")
            for atom_id, (x, y, z) in enumerate(coords, start=1):
                fh.write(f"{atom_id} 1 {x:.8f} {y:.8f} {z:.8f}\n")
        return data_path

    def write_lammps_input(self, md_config: MDConfig) -> Path:
        """Write a LAMMPS minimization input script scaffold."""
        workdir = self._workdir(md_config)
        workdir.mkdir(parents=True, exist_ok=True)
        input_path = self.input_path(md_config)
        data_file = self.data_path(md_config).name
        dump_file = self.dump_path(md_config).name
        log_file = md_config.lammps_log_filename

        pair_style = md_config.lammps_pair_style.strip()
        pair_coeff = md_config.lammps_pair_coeff.strip()
        potential_file = md_config.lammps_potential_file.strip()

        with open(input_path, "w", encoding="utf-8") as fh:
            fh.write("# Stage 3 LAMMPS minimization scaffold\n")
            fh.write("# Generated by graphene3d.stage3.sa_refine.LAMMPSRelaxationAdapter\n")
            fh.write(f"log {log_file}\n")
            fh.write(f"units {md_config.lammps_units}\n")
            fh.write(f"atom_style {md_config.lammps_atom_style}\n")
            fh.write(f"boundary {md_config.lammps_boundary}\n")
            fh.write(f"read_data {data_file}\n\n")
            if pair_style:
                fh.write(f"pair_style {pair_style}\n")
            else:
                fh.write("# TODO: set pair_style, for example: pair_style airebo 3.0\n")
            if pair_coeff:
                fh.write(f"pair_coeff {pair_coeff}\n")
            elif potential_file:
                fh.write(f"# TODO: set pair_coeff using potential file: {potential_file}\n")
            else:
                fh.write("# TODO: set pair_coeff for carbon potential\n")
            fh.write("\n")
            fh.write("neighbor 2.0 bin\n")
            fh.write("neigh_modify delay 0 every 1 check yes\n")
            fh.write("thermo 10\n")
            fh.write("thermo_style custom step pe fmax\n")
            fh.write(f"dump relaxed all custom 1 {dump_file} id x y z\n")
            fh.write("dump_modify relaxed sort id\n")
            fh.write(
                "minimize "
                f"{md_config.lammps_minimize_etol:.6e} "
                f"{md_config.lammps_minimize_ftol:.6e} "
                f"{int(md_config.lammps_minimize_maxiter)} "
                f"{int(md_config.lammps_minimize_maxeval)}\n"
            )
            fh.write(f"write_dump all custom {dump_file} id x y z modify sort id\n")
            fh.write("# NOTE: Python execution is opt-in with MDConfig.lammps_execute=True.\n")
        return input_path

    def run_lammps_subprocess(self, md_config: MDConfig) -> subprocess.CompletedProcess[str]:
        """Run LAMMPS in the configured working directory."""
        executable = self._resolve_lammps_executable(md_config)
        workdir = self._workdir(md_config)
        input_path = self.input_path(md_config)
        if not input_path.exists():
            raise FileNotFoundError(
                f"LAMMPS input script not found: {input_path}. "
                "Call write_lammps_input() before launching LAMMPS."
            )
        command = [executable, "-in", input_path.name]
        try:
            return subprocess.run(
                command,
                cwd=str(workdir),
                check=True,
                capture_output=True,
                text=True,
                timeout=int(md_config.lammps_timeout_seconds),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to launch LAMMPS executable '{executable}'. "
                "Check MDConfig.lammps_executable."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"LAMMPS timed out after {md_config.lammps_timeout_seconds} seconds. "
                f"Command: {' '.join(command)}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                "LAMMPS subprocess failed.\n"
                f"Command: {' '.join(command)}\n"
                f"Working directory: {workdir}\n"
                f"Exit code: {exc.returncode}\n"
                f"stdout:\n{stdout[-2000:]}\n"
                f"stderr:\n{stderr[-2000:]}"
            ) from exc

    def parse_dump_custom(self,
                          dump_path: str | Path,
                          expected_atoms: Optional[int] = None) -> np.ndarray:
        """
        Parse the final frame from a LAMMPS dump/custom file with id x y z.
        """
        path = Path(dump_path)
        if not path.exists():
            raise FileNotFoundError(f"LAMMPS dump/custom output not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        frames: list[list[str]] = []
        idx = 0
        while idx < len(lines):
            if not lines[idx].startswith("ITEM: TIMESTEP"):
                idx += 1
                continue
            idx += 2
            if idx >= len(lines) or not lines[idx].startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed LAMMPS dump near line {idx + 1}: expected NUMBER OF ATOMS")
            n_atoms = int(lines[idx + 1].strip())
            idx += 2
            if idx >= len(lines) or not lines[idx].startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Malformed LAMMPS dump near line {idx + 1}: expected BOX BOUNDS")
            idx += 4
            if idx >= len(lines) or not lines[idx].startswith("ITEM: ATOMS"):
                raise ValueError(f"Malformed LAMMPS dump near line {idx + 1}: expected ATOMS")
            header = lines[idx].split()[2:]
            idx += 1
            atom_lines = lines[idx:idx + n_atoms]
            idx += n_atoms
            frames.append([json.dumps(header), *atom_lines])

        if not frames:
            raise ValueError(f"No frames found in LAMMPS dump/custom file: {path}")

        final = frames[-1]
        header = json.loads(final[0])
        required = ["id", "x", "y", "z"]
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"LAMMPS dump missing required columns {missing}; found {header}")
        col = {name: header.index(name) for name in required}
        rows = []
        for line in final[1:]:
            parts = line.split()
            rows.append((
                int(float(parts[col["id"]])),
                float(parts[col["x"]]),
                float(parts[col["y"]]),
                float(parts[col["z"]]),
            ))
        rows.sort(key=lambda item: item[0])
        coords = np.array([[x, y, z] for _atom_id, x, y, z in rows], dtype=np.float64)
        if expected_atoms is not None and len(coords) != int(expected_atoms):
            raise ValueError(
                f"Expected {expected_atoms} atoms in LAMMPS dump, found {len(coords)}"
            )
        return coords

    def relax(self,
              coords_xyz_angstrom: np.ndarray,
              md_config: MDConfig) -> np.ndarray:
        coords = np.asarray(coords_xyz_angstrom, dtype=np.float64)
        if not md_config.enabled or md_config.backend != "lammps":
            self.last_summary = {
                "enabled": bool(md_config.enabled),
                "backend": md_config.backend,
                "applied": False,
                "reason": "MDConfig disabled or backend is not lammps.",
            }
            return coords.copy()

        data_path = self.write_lammps_data(coords, md_config)
        input_path = self.write_lammps_input(md_config)
        dump_path = self.dump_path(md_config)

        completed: Optional[subprocess.CompletedProcess[str]] = None
        if md_config.lammps_execute:
            if dump_path.exists():
                dump_path.unlink()
            completed = self.run_lammps_subprocess(md_config)
            if not dump_path.exists():
                self.last_summary = {
                    "enabled": True,
                    "backend": "lammps",
                    "applied": False,
                    "executed": True,
                    "data_file": str(data_path),
                    "input_script": str(input_path),
                    "expected_dump_file": str(dump_path),
                    "returncode": int(completed.returncode),
                }
                raise RuntimeError(
                    "LAMMPS subprocess completed, but the expected dump/custom "
                    f"output was not produced: {dump_path}. Check the input script, "
                    "potential settings, and LAMMPS log."
                )
        elif not dump_path.exists():
            self.last_summary = {
                "enabled": True,
                "backend": "lammps",
                "applied": False,
                "executed": False,
                "data_file": str(data_path),
                "input_script": str(input_path),
                "expected_dump_file": str(dump_path),
            }
            raise RuntimeError(
                "LAMMPSRelaxationAdapter wrote the LAMMPS data and input files, "
                "but MDConfig.lammps_execute=False and no existing dump/custom "
                f"output was found at {dump_path}. Set lammps_execute=True for "
                "standalone execution, or run LAMMPS externally and provide the dump."
            )

        raw_relaxed = self.parse_dump_custom(dump_path, expected_atoms=len(coords))
        raw_delta = raw_relaxed - coords
        raw_displacement = np.linalg.norm(raw_delta, axis=1)

        relaxed = raw_relaxed
        displacement_cap = md_config.max_displacement_angstrom
        cap_applied = False
        capped_atom_count = 0
        if displacement_cap is not None:
            cap = max(float(displacement_cap), 0.0)
            relaxed = raw_relaxed.copy()
            too_large = raw_displacement > cap
            capped_atom_count = int(np.sum(too_large))
            if capped_atom_count:
                scale = np.ones_like(raw_displacement)
                scale[too_large] = cap / np.maximum(raw_displacement[too_large], 1e-12)
                relaxed = coords + raw_delta * scale[:, None]
                cap_applied = True

        displacement = np.linalg.norm(relaxed - coords, axis=1)
        self.last_summary = {
            "enabled": True,
            "backend": "lammps",
            "applied": True,
            "executed": bool(md_config.lammps_execute),
            "data_file": str(data_path),
            "input_script": str(input_path),
            "dump_file": str(dump_path),
            "returncode": int(completed.returncode) if completed is not None else None,
            "stdout_tail": (completed.stdout or "")[-2000:] if completed is not None else "",
            "stderr_tail": (completed.stderr or "")[-2000:] if completed is not None else "",
            "raw_max_atom_displacement_angstrom": float(np.max(raw_displacement)) if len(raw_displacement) else 0.0,
            "raw_mean_atom_displacement_angstrom": float(np.mean(raw_displacement)) if len(raw_displacement) else 0.0,
            "displacement_cap_angstrom": (
                float(displacement_cap)
                if displacement_cap is not None
                else None
            ),
            "displacement_cap_applied": bool(cap_applied),
            "capped_atom_count": int(capped_atom_count),
            "max_atom_displacement_angstrom": float(np.max(displacement)) if len(displacement) else 0.0,
            "mean_atom_displacement_angstrom": float(np.mean(displacement)) if len(displacement) else 0.0,
        }
        return relaxed


def make_md_relaxation_adapter(md_config: MDConfig) -> MDRelaxationAdapter:
    """
    Create an MD relaxation adapter for future SA->MD wiring tests.

    This factory is not used by the frozen SA baseline.  It simply documents
    the adapter selection boundary for later LAMMPS integration.
    """
    if md_config.enabled and md_config.backend in {"fake_md", "fake", "fake_lammps"}:
        return FakeMDRelaxationAdapter()
    if md_config.enabled and md_config.backend == "lammps":
        return LAMMPSRelaxationAdapter()
    return NoOpMDRelaxationAdapter()


def run_lammps_adapter_standalone_test(coords_xyz_angstrom: Optional[np.ndarray] = None,
                                       md_config: Optional[MDConfig] = None) -> dict[str, Any]:
    """
    Run one standalone LAMMPS adapter relaxation outside the SA loop.

    This helper is for validating the adapter boundary only.  It writes the
    LAMMPS data/input files, launches LAMMPS when lammps_execute=True, verifies
    the expected dump/custom output, parses relaxed coordinates, and returns a
    compact diagnostic summary.
    """
    coords = (
        np.asarray(coords_xyz_angstrom, dtype=np.float64)
        if coords_xyz_angstrom is not None
        else np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]], dtype=np.float64)
    )
    if md_config is None:
        md_config = MDConfig(
            enabled=True,
            backend="lammps",
            lammps_execute=True,
        )
    adapter = make_md_relaxation_adapter(md_config)
    if not isinstance(adapter, LAMMPSRelaxationAdapter):
        raise ValueError(
            "run_lammps_adapter_standalone_test requires MDConfig(enabled=True, backend='lammps')."
        )

    relaxed = adapter.relax(coords, md_config)
    displacement = np.linalg.norm(relaxed - coords, axis=1)
    summary = {
        "status": "lammps_adapter_standalone_complete",
        "n_atoms": int(len(coords)),
        "input_shape": list(coords.shape),
        "relaxed_shape": list(relaxed.shape),
        "max_displacement_angstrom": float(np.max(displacement)) if len(displacement) else 0.0,
        "mean_displacement_angstrom": float(np.mean(displacement)) if len(displacement) else 0.0,
        "adapter_summary": adapter.last_summary,
    }
    print("LAMMPS adapter standalone test")
    print("==============================")
    print(f"atoms             : {summary['n_atoms']}")
    print(f"data file         : {adapter.last_summary.get('data_file', '')}")
    print(f"input script      : {adapter.last_summary.get('input_script', '')}")
    print(f"dump file         : {adapter.last_summary.get('dump_file', '')}")
    print(f"executed          : {adapter.last_summary.get('executed', False)}")
    print(f"max displacement  : {summary['max_displacement_angstrom']:.6f} A")
    return summary


def make_stable_stage3_baseline_config(
    target_image_path: str | Path | None = None,
    n_iterations: int = 200,
    output_prefix: str = "stage3_sa_stable_baseline",
    verbose: bool = True,
) -> SAConfig:
    """
    Return the frozen Stage 3 pre-MD baseline configuration.

    This named preset documents the current working baseline before stronger
    MD/LAMMPS integration.  It keeps the SA algorithm unchanged and simply
    centralizes the validated settings used by the 200-iteration regularized
    run in outputs/stage3/runs/sa_run_046.
    """
    return SAConfig(
        n_iterations=int(n_iterations),
        initial_temperature=1.0e-4,
        cooling_rate=0.995,
        step_size_xy=0.10,
        step_size_z=0.06,
        simulator_z_contrast_scale=0.08,
        enable_pre_sa_sanitization=True,
        sanitization_max_passes=10,
        sanitization_buffer_angstrom=0.02,
        enable_structural_rejection=True,
        structural_min_distance_angstrom=0.8,
        enable_structural_regularization=True,
        structural_regularization_weight=1.0e-3,
        regularization_target_nn_angstrom=1.42,
        regularization_neighbor_cutoff_angstrom=1.9,
        regularization_overlap_weight=10.0,
        regularization_smoothness_weight=0.0,
        target_image_path=str(target_image_path) if target_image_path is not None else str(DEFAULT_TARGET_IMAGE),
        output_prefix=output_prefix,
        save_outputs=True,
        verbose=bool(verbose),
    )


def make_periodic_weak_lammps_md_config(
    lammps_executable: str | Path = "lmp",
    lammps_potential_file: str | Path | None = None,
    lammps_working_dir: str | Path = "runs/lammps_periodic_weak",
    lammps_pair_style: str = "tersoff",
    lammps_pair_coeff: Optional[str] = None,
    potential_element: str = "C",
    apply_every_iterations: int = 5,
    max_displacement_angstrom: float = 0.001,
    timeout_seconds: int = 20,
) -> MDConfig:
    """
    Return the first validated real-MD Stage 3 configuration.

    This preset runs real LAMMPS minimization periodically as a weak structural
    regularizer inside the SA loop.  It intentionally uses loose minimization
    limits plus a small per-atom displacement cap so LAMMPS behaves like a
    conservative correction instead of rewriting each image-driven proposal.

    If pair_coeff references only a potential filename, copy that potential
    into lammps_working_dir before running, or configure LAMMPS so it can find
    the file.  The validated Windows tests used bundled BNC.tersoff copied into
    the working directory with pair_coeff="* * BNC.tersoff C".
    """
    potential_path = (
        Path(lammps_potential_file)
        if lammps_potential_file is not None
        else None
    )
    if lammps_pair_coeff is None:
        potential_name = potential_path.name if potential_path is not None else "BNC.tersoff"
        lammps_pair_coeff = f"* * {potential_name} {potential_element}"

    return MDConfig(
        enabled=True,
        backend="lammps",
        lammps_execute=True,
        lammps_executable=str(lammps_executable),
        lammps_working_dir=str(lammps_working_dir),
        lammps_potential_file=str(potential_path) if potential_path is not None else "",
        lammps_pair_style=lammps_pair_style,
        lammps_pair_coeff=lammps_pair_coeff,
        lammps_minimize_etol=1.0e-2,
        lammps_minimize_ftol=1.0e-2,
        lammps_minimize_maxiter=1,
        lammps_minimize_maxeval=5,
        lammps_timeout_seconds=int(timeout_seconds),
        max_displacement_angstrom=float(max_displacement_angstrom),
        apply_every_iterations=int(apply_every_iterations),
    )


@dataclass
class SAInput:
    """Stage 2 initializer payload used by simulated annealing."""

    atom_id: np.ndarray
    element: np.ndarray
    xyz_angstrom: np.ndarray
    xyz_pixels: np.ndarray
    atom_xy_pixels: np.ndarray
    source_label: np.ndarray
    initializer_weight: np.ndarray
    metadata: dict[str, Any]


@dataclass
class SAResult:
    """Container returned by run_sa_refinement()."""

    initial_xyz: np.ndarray
    original_xyz: np.ndarray
    current_xyz: np.ndarray
    best_xyz: np.ndarray
    atom_id: np.ndarray
    element: np.ndarray
    source_label: np.ndarray
    initializer_weight: np.ndarray
    initial_objective: float
    current_objective: float
    best_objective: float
    objective_history: np.ndarray
    acceptance_history: np.ndarray
    sanitization_summary: dict[str, Any]
    objective_component_summary: dict[str, Any]
    md_relaxation_summary: dict[str, Any]
    output_paths: dict[str, str]


@dataclass
class SARunDirectory:
    """Directory layout for one SA run."""

    run_dir: Path
    working_dir: Path
    outputs_dir: Path
    logs_dir: Path
    config_path: Path


class TEMForwardSimulator(Protocol):
    """Adapter protocol for TEM forward simulation."""

    def simulate(self, coords_angstrom: np.ndarray) -> np.ndarray:
        """Return a simulated TEM image for the current coordinates."""


def _load_image(path: str | Path) -> np.ndarray:
    """Load a target TEM image as float64."""
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        image = np.load(image_path).astype(np.float64)
    elif suffix == ".npz":
        data = np.load(image_path)
        if "image" not in data:
            raise ValueError(f"NPZ target image must contain an 'image' array: {image_path}")
        image = data["image"].astype(np.float64)
    else:
        import tifffile
        image = tifffile.imread(str(image_path)).astype(np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D target TEM image, got {image.shape}")
    return image


def _pixel_size_from_metadata(metadata: dict[str, Any], default: float = 0.183) -> float:
    """Read Stage 2 pixel size from SA metadata if available."""
    config = metadata.get("config", {})
    return float(config.get("pixel_size_angstrom", default))


def _coords_angstrom_to_pixels(coords_angstrom: np.ndarray,
                               image_shape: tuple[int, int],
                               pixel_size_angstrom: float) -> np.ndarray:
    """Convert Stage 3 Angstrom coordinates back to image x-y pixels."""
    coords = np.asarray(coords_angstrom, dtype=np.float64)
    height_px = image_shape[0]
    x_pix = coords[:, 0] / pixel_size_angstrom
    y_pix = height_px - coords[:, 1] / pixel_size_angstrom
    return np.column_stack([x_pix, y_pix])


def _normalize_image(image: np.ndarray, eps: float) -> np.ndarray:
    """Normalize an image for scale-insensitive mismatch calculations."""
    arr = np.asarray(image, dtype=np.float64)
    return (arr - float(np.mean(arr))) / (float(np.std(arr)) + eps)


class GaussianProjectionSimulator:
    """
    Runnable proxy TEM forward simulator.

    This is an adapter with the same shape as the real simulator interface, not
    a substitute for the abTEM physics model.  It projects atoms into the image
    plane as dark Gaussian spots so image-mismatch plumbing can be tested
    end-to-end.

    The optional z contrast is deliberately simple: each atom's dark Gaussian
    amplitude is weakly modulated by its centered z coordinate.  This gives the
    debug objective a controllable z response before the real abTEM path is
    used.  Set z_contrast_scale=0.0 to reproduce the old z-insensitive behavior.
    """

    def __init__(self,
                 image_shape: tuple[int, int],
                 pixel_size_angstrom: float,
                 sigma_px: float = 1.2,
                 atom_contrast: float = 1.0,
                 background: float = 0.0,
                 z_contrast_scale: float = 0.08) -> None:
        self.image_shape = tuple(int(v) for v in image_shape)
        self.pixel_size_angstrom = float(pixel_size_angstrom)
        self.sigma_px = float(sigma_px)
        self.atom_contrast = float(atom_contrast)
        self.background = float(background)
        self.z_contrast_scale = float(z_contrast_scale)

    def simulate(self, coords_angstrom: np.ndarray) -> np.ndarray:
        """Project atoms as dark Gaussian spots into the target image frame."""
        coords = np.asarray(coords_angstrom, dtype=np.float64)
        image = np.full(self.image_shape, self.background, dtype=np.float64)
        xy_pix = _coords_angstrom_to_pixels(
            coords,
            self.image_shape,
            self.pixel_size_angstrom,
        )

        sigma = max(self.sigma_px, 1e-6)
        radius = max(1, int(np.ceil(4.0 * sigma)))
        z = coords[:, 2]
        z_centered = z - float(np.mean(z)) if len(z) else z
        z_scale = float(np.std(z_centered)) + 1e-8

        h, w = self.image_shape
        for idx, (x_pix, y_pix) in enumerate(xy_pix):
            cx = int(round(x_pix))
            cy = int(round(y_pix))
            if cx < -radius or cx >= w + radius or cy < -radius or cy >= h + radius:
                continue
            c0 = max(0, cx - radius)
            c1 = min(w, cx + radius + 1)
            r0 = max(0, cy - radius)
            r1 = min(h, cy + radius + 1)
            xs = np.arange(c0, c1, dtype=np.float64)
            ys = np.arange(r0, r1, dtype=np.float64)
            X, Y = np.meshgrid(xs, ys)
            gaussian = np.exp(-((X - x_pix) ** 2 + (Y - y_pix) ** 2) / (2.0 * sigma ** 2))
            # Debug proxy only: bounded relative-z modulation of spot depth.
            # With z_contrast_scale=0.0, z_factor is exactly 1.0 and the old
            # z-insensitive Gaussian projection is recovered.
            z_relative = np.tanh(z_centered[idx] / z_scale)
            z_factor = 1.0 + self.z_contrast_scale * z_relative
            image[r0:r1, c0:c1] -= self.atom_contrast * z_factor * gaussian
        return image


def _coords_to_ase_atoms(coords_angstrom: np.ndarray,
                         elements: np.ndarray,
                         image_shape: tuple[int, int],
                         pixel_size_angstrom: float,
                         vacuum_angstrom: float):
    """
    Convert Stage 2/3 coordinates into an ASE Atoms object for abTEM.

    Stage 2 x/y are already in Angstrom in the image coordinate frame.  The
    Stage 2 z scale is an initializer height field, so it is shifted into a
    positive simulation cell with vacuum padding.
    """
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "AbTEMSimulatorAdapter requires ASE. Install it with abTEM before "
            "using simulator_kind='abtem'."
        ) from exc

    coords = np.asarray(coords_angstrom, dtype=np.float64)
    symbols = np.asarray(elements).astype(str).tolist()
    height_px, width_px = image_shape
    cell_x = width_px * pixel_size_angstrom
    cell_y = height_px * pixel_size_angstrom
    z_shifted = coords[:, 2] - float(np.min(coords[:, 2])) + 0.5 * vacuum_angstrom
    cell_z = float(np.max(z_shifted)) + 0.5 * vacuum_angstrom
    positions = np.column_stack([coords[:, 0], coords[:, 1], z_shifted])
    atoms = Atoms(symbols=symbols, positions=positions, cell=[cell_x, cell_y, cell_z], pbc=False)
    return atoms


def _abtem_to_numpy(measurement: Any) -> np.ndarray:
    """Extract a NumPy array from common abTEM lazy/eager return objects."""
    obj = measurement
    if hasattr(obj, "compute"):
        obj = obj.compute()
    if hasattr(obj, "array"):
        return np.asarray(obj.array, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def _resize_image_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize simulator output to match the target image shape if needed."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.shape == shape:
        return arr
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError(
            "Simulator image shape does not match the target and scipy is not "
            "available for resizing."
        ) from exc
    zoom_factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    resized = zoom(arr, zoom_factors, order=1)
    return resized[:shape[0], :shape[1]]


class AbTEMSimulatorAdapter:
    """
    Python-only abTEM forward-simulator adapter.

    This is the real forward-simulator integration point for Stage 3.  It
    converts Stage 2 xyz Angstrom coordinates to an ASE Atoms object, builds an
    abTEM Potential, runs a plane-wave multislice simulation, and returns a
    2-D intensity image for ImageMismatchObjective.

    TODO when abTEM is installed in the environment:
      - tune microscope parameters such as defocus, convergence/illumination,
        dose, detector/CTF settings, and scan/exit-plane handling
      - replace the simple plane-wave intensity path below if your experiment
        needs a more specific abTEM imaging pipeline
    """

    def __init__(self,
                 image_shape: tuple[int, int],
                 pixel_size_angstrom: float,
                 elements: np.ndarray,
                 energy_ev: float = 80000.0,
                 sampling_angstrom: Optional[float] = None,
                 slice_thickness_angstrom: float = 1.0,
                 vacuum_angstrom: float = 5.0,
                 device: str = "cpu") -> None:
        self.image_shape = tuple(int(v) for v in image_shape)
        self.pixel_size_angstrom = float(pixel_size_angstrom)
        self.elements = np.asarray(elements).astype(str)
        self.energy_ev = float(energy_ev)
        self.sampling_angstrom = (
            float(sampling_angstrom)
            if sampling_angstrom is not None
            else float(pixel_size_angstrom)
        )
        self.slice_thickness_angstrom = float(slice_thickness_angstrom)
        self.vacuum_angstrom = float(vacuum_angstrom)
        self.device = str(device)

    def simulate(self, coords_angstrom: np.ndarray) -> np.ndarray:
        """Run abTEM in Python and return an image matching target shape."""
        try:
            import abtem
        except ImportError as exc:
            raise ImportError(
                "simulator_kind='abtem' requires the abTEM package. Install "
                "abTEM and ASE in this Python environment, then rerun SA."
            ) from exc

        atoms = _coords_to_ase_atoms(
            coords_angstrom,
            self.elements,
            self.image_shape,
            self.pixel_size_angstrom,
            self.vacuum_angstrom,
        )
        try:
            potential = abtem.Potential(
                atoms,
                sampling=self.sampling_angstrom,
                slice_thickness=self.slice_thickness_angstrom,
            )
            wave = abtem.PlaneWave(energy=self.energy_ev)
            exit_wave = wave.multislice(potential)
            intensity = exit_wave.intensity()
        except AttributeError as exc:
            raise RuntimeError(
                "The installed abTEM API did not expose the expected "
                "Potential / PlaneWave / multislice methods. Update "
                "AbTEMSimulatorAdapter.simulate() for your abTEM version."
            ) from exc

        image = _abtem_to_numpy(intensity)
        if image.ndim > 2:
            image = np.squeeze(image)
        if image.ndim != 2:
            raise ValueError(f"abTEM returned a non-2D image with shape {image.shape}")
        return _resize_image_to_shape(image, self.image_shape)


class ImageMismatchObjective:
    """
    TEM image-mismatch objective for SA refinement.

    The simulator adapter can be the built-in GaussianProjectionSimulator or
    the Python-only AbTEMSimulatorAdapter.  Supported metrics:
      - normalized_mse: z-score both images, then mean squared difference
      - mse: raw mean squared difference
      - chi_square: sum((target - sim)^2 / max(abs(target), epsilon)) / N
    """

    def __init__(self,
                 target_image: np.ndarray,
                 simulator: TEMForwardSimulator,
                 metric: str = "normalized_mse",
                 normalize: bool = True,
                 epsilon: float = 1e-8) -> None:
        self.target_image = np.asarray(target_image, dtype=np.float64)
        self.simulator = simulator
        self.metric = metric
        self.normalize = bool(normalize)
        self.epsilon = float(epsilon)
        if self.normalize:
            self.target_for_compare = _normalize_image(self.target_image, self.epsilon)
        else:
            self.target_for_compare = self.target_image

    def __call__(self, coords_angstrom: np.ndarray) -> float:
        simulated = self.simulator.simulate(coords_angstrom)
        if simulated.shape != self.target_image.shape:
            raise ValueError(
                f"Simulated image shape {simulated.shape} does not match "
                f"target shape {self.target_image.shape}"
            )

        if self.metric == "normalized_mse":
            target = self.target_for_compare
            sim = _normalize_image(simulated, self.epsilon)
            return float(np.mean((target - sim) ** 2))
        if self.metric == "mse":
            target = self.target_for_compare
            sim = _normalize_image(simulated, self.epsilon) if self.normalize else simulated
            return float(np.mean((target - sim) ** 2))
        if self.metric == "chi_square":
            target = self.target_for_compare
            sim = _normalize_image(simulated, self.epsilon) if self.normalize else simulated
            denom = np.maximum(np.abs(target), self.epsilon)
            return float(np.mean((target - sim) ** 2 / denom))

        raise ValueError(
            "Unknown objective_metric. Use 'normalized_mse', 'mse', or 'chi_square'."
        )


class StructuralRegularizationObjective:
    """
    Lightweight MD-style structural proxy for SA refinement.

    This is not a replacement for LAMMPS or molecular dynamics.  It is a small,
    local geometry penalty that can be mixed with the image mismatch objective
    to discourage strained nearest-neighbor geometry before full MD is wired in.
    """

    def __init__(self,
                 target_nn_angstrom: float = 1.42,
                 neighbor_cutoff_angstrom: float = 1.9,
                 min_distance_angstrom: float = 0.8,
                 overlap_weight: float = 10.0,
                 smoothness_weight: float = 0.0) -> None:
        self.target_nn_angstrom = float(target_nn_angstrom)
        self.neighbor_cutoff_angstrom = float(neighbor_cutoff_angstrom)
        self.min_distance_angstrom = float(min_distance_angstrom)
        self.overlap_weight = float(overlap_weight)
        self.smoothness_weight = float(smoothness_weight)

    def components(self, coords_angstrom: np.ndarray) -> dict[str, float]:
        coords = np.asarray(coords_angstrom, dtype=np.float64)
        if len(coords) < 2:
            return {
                "raw_regularization": 0.0,
                "bond_length_penalty": 0.0,
                "overlap_penalty": 0.0,
                "smoothness_penalty": 0.0,
                "nearest_neighbor_min_angstrom": float("nan"),
                "nearest_neighbor_mean_angstrom": float("nan"),
                "close_pair_count": 0.0,
            }

        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, np.inf)
        nn = np.min(distances, axis=1)
        finite_nn = nn[np.isfinite(nn)]
        if len(finite_nn) == 0:
            return {
                "raw_regularization": 0.0,
                "bond_length_penalty": 0.0,
                "overlap_penalty": 0.0,
                "smoothness_penalty": 0.0,
                "nearest_neighbor_min_angstrom": float("nan"),
                "nearest_neighbor_mean_angstrom": float("nan"),
                "close_pair_count": 0.0,
            }

        target = max(self.target_nn_angstrom, 1e-8)
        cutoff = max(self.neighbor_cutoff_angstrom, target)
        supported = finite_nn <= cutoff
        nn_for_bonds = finite_nn[supported] if np.any(supported) else finite_nn
        bond_length_penalty = float(np.mean(((nn_for_bonds - target) / target) ** 2))

        close = finite_nn < self.min_distance_angstrom
        if np.any(close):
            overlap_penalty = float(np.mean(((self.min_distance_angstrom - finite_nn[close]) / self.min_distance_angstrom) ** 2))
        else:
            overlap_penalty = 0.0

        if self.smoothness_weight > 0.0 and len(nn_for_bonds) > 1:
            median_nn = float(np.median(nn_for_bonds))
            smoothness_penalty = float(np.mean(((nn_for_bonds - median_nn) / target) ** 2))
        else:
            smoothness_penalty = 0.0

        raw = (
            bond_length_penalty
            + self.overlap_weight * overlap_penalty
            + self.smoothness_weight * smoothness_penalty
        )
        return {
            "raw_regularization": float(raw),
            "bond_length_penalty": float(bond_length_penalty),
            "overlap_penalty": float(overlap_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "nearest_neighbor_min_angstrom": float(np.min(finite_nn)),
            "nearest_neighbor_mean_angstrom": float(np.mean(finite_nn)),
            "close_pair_count": float(len(_close_pairs(coords, self.min_distance_angstrom))),
        }

    def __call__(self, coords_angstrom: np.ndarray) -> float:
        return self.components(coords_angstrom)["raw_regularization"]


def _structural_regularizer_from_config(config: SAConfig) -> StructuralRegularizationObjective:
    """Create the lightweight structural regularizer from SA config."""
    return StructuralRegularizationObjective(
        target_nn_angstrom=config.regularization_target_nn_angstrom,
        neighbor_cutoff_angstrom=config.regularization_neighbor_cutoff_angstrom,
        min_distance_angstrom=config.structural_min_distance_angstrom,
        overlap_weight=config.regularization_overlap_weight,
        smoothness_weight=config.regularization_smoothness_weight,
    )


def _evaluate_sa_objective_components(coords_angstrom: np.ndarray,
                                      image_objective: Callable[[np.ndarray], float],
                                      regularizer: StructuralRegularizationObjective,
                                      config: SAConfig) -> dict[str, float]:
    """Evaluate image mismatch plus the optional structural regularization term."""
    image_value = float(image_objective(coords_angstrom))
    regularization_components = regularizer.components(coords_angstrom)
    structural_raw = float(regularization_components["raw_regularization"])
    structural_weighted = (
        float(config.structural_regularization_weight) * structural_raw
        if config.enable_structural_regularization
        else 0.0
    )
    result = {
        "total_objective": float(image_value + structural_weighted),
        "image_objective": image_value,
        "structural_regularization": structural_weighted,
        "structural_regularization_raw": structural_raw,
    }
    for key, value in regularization_components.items():
        result[f"structural_{key}"] = float(value)
    return result


def _log(config: SAConfig, message: str) -> None:
    if config.verbose:
        print(message)


def _next_sa_run_dir(runs_root: Path) -> Path:
    """Return the next auto-incremented runs/sa_run_XXX directory path."""
    runs_root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in runs_root.iterdir():
        if not child.is_dir() or not child.name.startswith("sa_run_"):
            continue
        suffix = child.name.removeprefix("sa_run_")
        if suffix.isdigit():
            used.append(int(suffix))
    next_idx = max(used, default=0) + 1
    return runs_root / f"sa_run_{next_idx:03d}"


def _prepare_sa_run_directory(config: SAConfig,
                              sa_input_path: str | Path,
                              output_dir: Optional[str | Path]) -> SARunDirectory:
    """
    Create the run directory layout and save the config snapshot.

    If output_dir is None, a new auto-incremented runs/sa_run_XXX folder is
    created.  If output_dir is supplied, it is treated as the run directory and
    the same working/outputs/logs layout is created inside it.
    """
    if output_dir is None:
        run_dir = _next_sa_run_dir(DEFAULT_RUNS_DIR)
    else:
        run_dir = Path(output_dir)

    working_dir = run_dir / "working"
    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"
    for path in [run_dir, working_dir, outputs_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    snapshot = {
        "config": asdict(config),
        "sa_input_path": str(Path(sa_input_path)),
        "run_dir": str(run_dir),
        "working_dir": str(working_dir),
        "outputs_dir": str(outputs_dir),
        "logs_dir": str(logs_dir),
        "note": (
            "SA run directory created automatically. All run outputs should "
            "stay inside this directory."
        ),
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2)

    return SARunDirectory(
        run_dir=run_dir,
        working_dir=working_dir,
        outputs_dir=outputs_dir,
        logs_dir=logs_dir,
        config_path=config_path,
    )


def load_stage2_sa_input(path: str | Path = DEFAULT_SA_INPUT) -> SAInput:
    """Load the Stage 2 SA handoff NPZ."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Stage 2 SA input not found: {input_path}")

    data = np.load(input_path, allow_pickle=False)
    metadata = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))

    return SAInput(
        atom_id=data["atom_id"].astype(int),
        element=data["element"].astype(str),
        xyz_angstrom=data["xyz_angstrom"].astype(np.float64),
        xyz_pixels=data["xyz_pixels"].astype(np.float64),
        atom_xy_pixels=data["atom_xy_pixels"].astype(np.float64),
        source_label=data["source_label"].astype(str),
        initializer_weight=data["initializer_weight"].astype(np.float64),
        metadata=metadata,
    )


def _default_target_image_path(sa_input: SAInput, config: SAConfig) -> Path:
    """Resolve the target image path for the default image-mismatch objective."""
    if config.target_image_path:
        return Path(config.target_image_path)
    baseline = sa_input.metadata.get("validation_baseline", {})
    if baseline.get("image"):
        return Path(__file__).resolve().parents[3] / str(baseline["image"])
    return DEFAULT_TARGET_IMAGE


def _default_simulator(sa_input: SAInput,
                       config: SAConfig,
                       target_image: np.ndarray) -> TEMForwardSimulator:
    """Create the configured TEM forward-simulator adapter."""
    if config.simulator_kind == "gaussian_projection":
        return GaussianProjectionSimulator(
            target_image.shape,
            _pixel_size_from_metadata(sa_input.metadata),
            sigma_px=config.simulator_sigma_px,
            atom_contrast=config.simulator_atom_contrast,
            background=config.simulator_background,
            z_contrast_scale=config.simulator_z_contrast_scale,
        )
    if config.simulator_kind == "abtem":
        return AbTEMSimulatorAdapter(
            target_image.shape,
            _pixel_size_from_metadata(sa_input.metadata),
            sa_input.element,
            energy_ev=config.abtem_energy_ev,
            sampling_angstrom=config.abtem_sampling_angstrom,
            slice_thickness_angstrom=config.abtem_slice_thickness_angstrom,
            vacuum_angstrom=config.abtem_vacuum_angstrom,
            device=config.abtem_device,
        )
    raise ValueError("Unknown simulator_kind. Use 'gaussian_projection' or 'abtem'.")


def _default_objective(sa_input: SAInput, config: SAConfig) -> ImageMismatchObjective:
    """Create the default TEM image-mismatch objective."""
    target_path = _default_target_image_path(sa_input, config)
    target_image = _load_image(target_path)
    simulator = _default_simulator(sa_input, config, target_image)
    return ImageMismatchObjective(
        target_image,
        simulator,
        metric=config.objective_metric,
        normalize=config.image_normalize,
        epsilon=config.image_epsilon,
    )


def _propose_single_atom_move(xyz: np.ndarray,
                              rng: np.random.Generator,
                              config: SAConfig) -> tuple[np.ndarray, int, np.ndarray]:
    """Perturb one atom with independent x/y/z Gaussian steps."""
    proposal = xyz.copy()
    atom_index = int(rng.integers(0, len(xyz)))
    delta = np.array([
        rng.normal(0.0, config.step_size_xy),
        rng.normal(0.0, config.step_size_xy),
        rng.normal(0.0, config.step_size_z),
    ], dtype=np.float64)
    proposal[atom_index] += delta
    return proposal, atom_index, delta


def _nearest_distance_for_atom(xyz: np.ndarray, atom_index: int) -> float:
    """Return the nearest-neighbor distance for one atom in Angstrom."""
    if len(xyz) <= 1:
        return np.inf
    point = xyz[atom_index]
    others = np.delete(xyz, atom_index, axis=0)
    distances = np.linalg.norm(others - point, axis=1)
    return float(np.min(distances))


def _close_pairs(xyz: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Return atom pairs closer than threshold, using simple O(N^2) checks."""
    coords = np.asarray(xyz, dtype=np.float64)
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(coords)):
        diffs = coords[i + 1:] - coords[i]
        if len(diffs) == 0:
            continue
        distances = np.linalg.norm(diffs, axis=1)
        close_local = np.where(distances < threshold)[0]
        for local_idx in close_local:
            j = i + 1 + int(local_idx)
            pairs.append((i, j, float(distances[local_idx])))
    return pairs


def _nearest_neighbor_min(xyz: np.ndarray) -> float:
    """Return the minimum nearest-neighbor distance in Angstrom."""
    if len(xyz) <= 1:
        return np.inf
    return float(min(_nearest_distance_for_atom(xyz, idx) for idx in range(len(xyz))))


def sanitize_initial_structure(xyz: np.ndarray,
                               min_distance_angstrom: float = 0.8,
                               max_passes: int = 10,
                               buffer_angstrom: float = 0.02) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Conservatively separate pre-existing close atom pairs before SA starts.

    Each pass nudges close pairs apart along their connecting vector by just
    enough to reach the configured minimum distance.  If two atoms are exactly
    coincident, a deterministic axis is used.
    """
    sanitized = np.asarray(xyz, dtype=np.float64).copy()
    target_distance = float(min_distance_angstrom + max(buffer_angstrom, 0.0))
    before_pairs = _close_pairs(sanitized, min_distance_angstrom)
    before_min = _nearest_neighbor_min(sanitized)
    total_correction = np.zeros_like(sanitized)
    n_pair_corrections = 0

    for _ in range(max_passes):
        pairs = _close_pairs(sanitized, target_distance)
        if not pairs:
            break
        pass_correction = np.zeros_like(sanitized)
        for i, j, distance in pairs:
            vector = sanitized[j] - sanitized[i]
            if distance > 1e-12:
                direction = vector / distance
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            correction = 0.5 * (target_distance - distance) * direction
            pass_correction[i] -= correction
            pass_correction[j] += correction
            n_pair_corrections += 1
        sanitized += pass_correction
        total_correction += pass_correction

    after_pairs = _close_pairs(sanitized, min_distance_angstrom)
    after_min = _nearest_neighbor_min(sanitized)
    correction_magnitudes = np.linalg.norm(total_correction, axis=1)
    summary = {
        "enabled": True,
        "threshold_angstrom": float(min_distance_angstrom),
        "target_distance_angstrom": float(target_distance),
        "buffer_angstrom": float(max(buffer_angstrom, 0.0)),
        "max_passes": int(max_passes),
        "close_pairs_before": int(len(before_pairs)),
        "close_pairs_after": int(len(after_pairs)),
        "nearest_neighbor_min_before_angstrom": float(before_min),
        "nearest_neighbor_min_after_angstrom": float(after_min),
        "pair_corrections_applied": int(n_pair_corrections),
        "max_atom_correction_angstrom": float(np.max(correction_magnitudes)) if len(correction_magnitudes) else 0.0,
        "mean_atom_correction_angstrom": float(np.mean(correction_magnitudes)) if len(correction_magnitudes) else 0.0,
    }
    return sanitized, summary


def _violates_structural_distance(current_xyz: np.ndarray,
                                  proposal_xyz: np.ndarray,
                                  atom_index: int,
                                  config: SAConfig) -> tuple[bool, float, float]:
    """
    Check whether a proposed move creates or worsens a local overlap.

    Existing close contacts are allowed to improve.  The proposal is rejected
    only if the moved atom's nearest-neighbor distance is below the configured
    threshold and is not larger than before the move.
    """
    if not config.enable_structural_rejection:
        proposal_nn = _nearest_distance_for_atom(proposal_xyz, atom_index)
        return False, _nearest_distance_for_atom(current_xyz, atom_index), proposal_nn

    current_nn = _nearest_distance_for_atom(current_xyz, atom_index)
    proposal_nn = _nearest_distance_for_atom(proposal_xyz, atom_index)
    violates = (
        proposal_nn < config.structural_min_distance_angstrom
        and proposal_nn <= current_nn
    )
    return bool(violates), current_nn, proposal_nn


def _save_best_coordinates_csv(path: Path,
                               result: SAResult) -> None:
    """Save best SA coordinates in an inspectable table."""
    header = (
        "atom_id,element,x_angstrom,y_angstrom,z_angstrom,source_label,"
        "initializer_weight"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for idx in range(len(result.best_xyz)):
            fh.write(
                f"{int(result.atom_id[idx])},{result.element[idx]},"
                f"{result.best_xyz[idx, 0]:.8f},"
                f"{result.best_xyz[idx, 1]:.8f},"
                f"{result.best_xyz[idx, 2]:.8f},"
                f"{result.source_label[idx]},"
                f"{result.initializer_weight[idx]:.3f}\n"
            )


def _save_history_csv(path: Path,
                      objective_history: np.ndarray,
                      acceptance_history: np.ndarray) -> None:
    """Save per-iteration objective and acceptance diagnostics."""
    header = (
        "iteration,temperature,objective_before,objective_after,"
        "delta_objective,current_objective,best_objective,improving,worsening,"
        "accepted,structural_rejected,objective_evaluated,atom_index,"
        "delta_x,delta_y,delta_z,proposal_size,current_nn_distance,"
        "proposal_nn_distance,image_objective_before,image_objective_after,"
        "delta_image_objective,structural_regularization_before,"
        "structural_regularization_after,delta_structural_regularization,"
        "structural_regularization_raw_before,structural_regularization_raw_after,"
        "md_relaxation_applied,md_corrected,md_correction_magnitude,"
        "md_max_atom_correction,md_pair_corrections,md_bond_corrections"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for idx in range(len(objective_history)):
            obj = objective_history[idx]
            acc = acceptance_history[idx]
            fh.write(
                f"{int(obj[0])},{obj[1]:.8f},{acc[6]:.12f},{acc[7]:.12f},"
                f"{acc[8]:.12f},{obj[2]:.12f},{obj[3]:.12f},"
                f"{int(acc[9])},{int(acc[10])},{int(acc[1])},{int(acc[12])},"
                f"{int(acc[15])},{int(acc[2])},"
                f"{acc[3]:.8f},{acc[4]:.8f},{acc[5]:.8f},{acc[11]:.8f},"
                f"{acc[13]:.8f},{acc[14]:.8f},"
                f"{acc[16]:.12f},{acc[17]:.12f},{acc[18]:.12f},"
                f"{acc[19]:.12f},{acc[20]:.12f},{acc[21]:.12f},"
                f"{acc[22]:.12f},{acc[23]:.12f},"
                f"{int(acc[24])},{int(acc[25])},{acc[26]:.12f},"
                f"{acc[27]:.12f},{int(acc[28])},{int(acc[29])}\n"
            )


def _proposal_diagnostics_summary(acceptance_history: np.ndarray) -> dict[str, float]:
    """Summarize proposal-level behavior from the extended history array."""
    if len(acceptance_history) == 0:
        return {
            "fraction_improving_proposals": 0.0,
            "fraction_worsening_proposals": 0.0,
            "fraction_worsening_proposals_accepted": 0.0,
            "mean_abs_proposal_size": 0.0,
            "mean_abs_delta_objective": 0.0,
            "structural_rejection_count": 0.0,
            "structural_rejection_fraction": 0.0,
        }

    improving = acceptance_history[:, 9].astype(bool)
    worsening = acceptance_history[:, 10].astype(bool)
    accepted = acceptance_history[:, 1].astype(bool)
    structural_rejected = (
        acceptance_history[:, 12].astype(bool)
        if acceptance_history.shape[1] > 12
        else np.zeros(len(acceptance_history), dtype=bool)
    )
    worsening_count = int(np.sum(worsening))
    worsening_accepted = int(np.sum(worsening & accepted))
    structural_rejection_count = int(np.sum(structural_rejected))

    return {
        "fraction_improving_proposals": float(np.mean(improving)),
        "fraction_worsening_proposals": float(np.mean(worsening)),
        "fraction_worsening_proposals_accepted": (
            float(worsening_accepted / worsening_count)
            if worsening_count
            else 0.0
        ),
        "mean_abs_proposal_size": float(np.mean(acceptance_history[:, 11])),
        "mean_abs_delta_objective": float(np.mean(np.abs(acceptance_history[:, 8]))),
        "structural_rejection_count": float(structural_rejection_count),
        "structural_rejection_fraction": float(structural_rejection_count / len(acceptance_history)),
    }


def _md_relaxation_diagnostics_summary(acceptance_history: np.ndarray,
                                       md_config: Optional[MDConfig]) -> dict[str, Any]:
    """Summarize optional MD adapter behavior from appended history columns."""
    base = {
        "enabled": bool(md_config.enabled) if md_config is not None else False,
        "backend": md_config.backend if md_config is not None else "none",
        "apply_every_iterations": (
            int(md_config.apply_every_iterations)
            if md_config is not None
            else 1
        ),
        "proposals_relaxed": 0,
        "proposals_corrected": 0,
        "fraction_corrected": 0.0,
        "mean_correction_magnitude_angstrom": 0.0,
        "max_correction_magnitude_angstrom": 0.0,
        "mean_max_atom_correction_angstrom": 0.0,
        "total_pair_corrections": 0,
        "total_bond_corrections": 0,
    }
    if len(acceptance_history) == 0 or acceptance_history.shape[1] < 30:
        return base

    relaxed = acceptance_history[:, 24].astype(bool)
    corrected = acceptance_history[:, 25].astype(bool)
    correction_mag = acceptance_history[:, 26]
    max_atom = acceptance_history[:, 27]
    base.update({
        "proposals_relaxed": int(np.sum(relaxed)),
        "proposals_corrected": int(np.sum(corrected)),
        "fraction_corrected": float(np.mean(corrected)),
        "mean_correction_magnitude_angstrom": (
            float(np.mean(correction_mag[corrected]))
            if np.any(corrected)
            else 0.0
        ),
        "max_correction_magnitude_angstrom": float(np.max(correction_mag)) if len(correction_mag) else 0.0,
        "mean_max_atom_correction_angstrom": (
            float(np.mean(max_atom[corrected]))
            if np.any(corrected)
            else 0.0
        ),
        "total_pair_corrections": int(np.sum(acceptance_history[:, 28])),
        "total_bond_corrections": int(np.sum(acceptance_history[:, 29])),
    })
    return base


def _should_apply_md_relaxation(iteration: int, md_config: MDConfig) -> bool:
    """Return whether the optional MD adapter should run this SA iteration."""
    if not md_config.enabled:
        return False
    cadence = int(md_config.apply_every_iterations)
    if cadence <= 1:
        return True
    return iteration % cadence == 0


def _save_sa_outputs(output_dir: Path,
                     config: SAConfig,
                     sa_input: SAInput,
                     result: SAResult,
                     run_directory: Optional[SARunDirectory] = None) -> dict[str, str]:
    """Save Stage 3 SA skeleton debug outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_prefix
    best_csv = output_dir / f"{prefix}_best_coordinates.csv"
    best_npz = output_dir / f"{prefix}_best_coordinates.npz"
    history_csv = output_dir / f"{prefix}_history.csv"
    summary_json = output_dir / f"{prefix}_summary.json"

    _save_best_coordinates_csv(best_csv, result)
    _save_history_csv(history_csv, result.objective_history, result.acceptance_history)

    np.savez_compressed(
        best_npz,
        atom_id=result.atom_id,
        element=result.element.astype(str),
        original_xyz_angstrom=result.original_xyz,
        initial_xyz_angstrom=result.initial_xyz,
        current_xyz_angstrom=result.current_xyz,
        best_xyz_angstrom=result.best_xyz,
        source_label=result.source_label.astype(str),
        initializer_weight=result.initializer_weight,
        objective_history=result.objective_history,
        acceptance_history=result.acceptance_history,
        sanitization_summary_json=json.dumps(result.sanitization_summary),
        objective_component_summary_json=json.dumps(result.objective_component_summary),
        md_relaxation_summary_json=json.dumps(result.md_relaxation_summary),
        config_json=json.dumps(asdict(config)),
        stage2_metadata_json=json.dumps(sa_input.metadata),
    )

    accepted = int(np.sum(result.acceptance_history[:, 1])) if len(result.acceptance_history) else 0
    proposal_summary = _proposal_diagnostics_summary(result.acceptance_history)
    summary = {
        "status": "sa_skeleton_complete",
        "note": (
            "Uses ImageMismatchObjective. The default simulator_kind is "
            "gaussian_projection, a proxy adapter; use simulator_kind='abtem' "
            "for the Python-only abTEM forward-simulator path."
        ),
        "n_atoms": int(len(result.best_xyz)),
        "n_iterations": int(config.n_iterations),
        "accepted_moves": accepted,
        "acceptance_rate": float(accepted / max(config.n_iterations, 1)),
        "initial_objective": result.initial_objective,
        "current_objective": result.current_objective,
        "best_objective": result.best_objective,
        "objective_components": result.objective_component_summary,
        "md_relaxation": result.md_relaxation_summary,
        "proposal_diagnostics": proposal_summary,
        "sanitization": result.sanitization_summary,
        "config": asdict(config),
        "stage2_metadata": sa_input.metadata,
    }
    if run_directory is not None:
        summary["run_directory"] = {
            "run_dir": str(run_directory.run_dir),
            "working_dir": str(run_directory.working_dir),
            "outputs_dir": str(run_directory.outputs_dir),
            "logs_dir": str(run_directory.logs_dir),
            "config_path": str(run_directory.config_path),
        }
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    paths = {
        "best_coordinates_csv": str(best_csv),
        "best_coordinates_npz": str(best_npz),
        "history_csv": str(history_csv),
        "summary_json": str(summary_json),
    }
    if run_directory is not None:
        paths.update({
            "run_dir": str(run_directory.run_dir),
            "working_dir": str(run_directory.working_dir),
            "outputs_dir": str(run_directory.outputs_dir),
            "logs_dir": str(run_directory.logs_dir),
            "config_json": str(run_directory.config_path),
        })
    return paths


def calibrate_initial_temperature(
    positions: np.ndarray,
    target_image: np.ndarray,
    simulator,
    step_xy: float = 0.08,
    step_z: float = 0.25,
    n_samples: int = 80,
    target_acceptance: float = 0.50,
    md_relaxer=None,
) -> float:
    """
    Set T0 so ~50% of random uphill single-atom moves are accepted at start.
    Uses single-atom proposals to match the actual SA move strategy.
    If md_relaxer is provided, both the baseline and each candidate are
    projected onto the relaxed manifold first — matching the inner loop's
    comparison space exactly.
    """
    baseline = positions.copy()
    if md_relaxer is not None:
        try:
            baseline = md_relaxer.relax(baseline)
        except Exception:
            pass

    chi2_base = simulator.chi2(baseline, target_image)
    uphill = []
    N = len(baseline)
    for _ in range(n_samples):
        p = baseline.copy()
        idx = np.random.randint(N)
        p[idx, 0] += np.random.normal(0, step_xy)
        p[idx, 1] += np.random.normal(0, step_xy)
        p[idx, 2] += np.random.normal(0, step_z)
        if md_relaxer is not None:
            try:
                p = md_relaxer.relax(p)
            except Exception:
                pass
        delta = simulator.chi2(p, target_image) - chi2_base
        if delta > 0:
            uphill.append(delta)
    if not uphill:
        return 1e-4
    T0 = float(np.mean(uphill) / (-np.log(target_acceptance)))
    space = "relaxed" if md_relaxer is not None else "unrelaxed"
    print(f"  Auto-calibrated T0 = {T0:.6e} from {len(uphill)} "
          f"single-atom uphill samples ({space} space)")
    return T0


def make_paper_aligned_config(
    simulator_kind: str = 'abtem',
    md_mode: str = 'none',
    max_outer_iters: int = 6,
    max_inner_steps: int = 600,
    step_size_xy_ang: float = 0.08,
    step_size_z_ang:  float = 0.15,
    initial_temperature = 'auto',
    T_final_fraction: float = 0.01,
    outer_tol: float = 0.005,
    defocus_ang: float = -80.0,
    voltage_kV: float = 80.0,
    dose: float = 8000.0,
    slice_thickness: float = 1.0,
    lammps_executable: str = None,
    potential_file: str = None,
    lammps_workdir: str = 'runs/lammps_sa',
    lammps_timeout_seconds: float = 15.0,
    lammps_max_displacement_angstrom: float = 0.5,
    lammps_xy_cap_angstrom: float | None = None,
    lammps_z_cap_angstrom: float | None = None,
) -> dict:
    """Config dict for run_sa_refinement_paper_aligned.

    lammps_xy_cap_angstrom / lammps_z_cap_angstrom
        If set, enable anisotropic displacement capping in LammpsMinimizer.
        xy_cap prevents large LAMMPS xy-flattening motion (which hurts chi2).
        z_cap allows LAMMPS bond-length corrections to act on z.
        Recommended: xy_cap=0.05, z_cap=0.5 for the per-candidate projector.
    """
    return {
        'simulator_kind': simulator_kind,
        'md_mode': md_mode,
        'max_outer_iters': max_outer_iters,
        'max_inner_steps': max_inner_steps,
        'step_size_xy_ang': step_size_xy_ang,
        'step_size_z_ang':  step_size_z_ang,
        'initial_temperature': initial_temperature,
        'T_final_fraction': T_final_fraction,
        'outer_tol': outer_tol,  # stop outer loop if |delta_chi2| < this
        'defocus_ang': defocus_ang,
        'voltage_kV':  voltage_kV,
        'dose': dose,
        'slice_thickness': slice_thickness,
        'lammps_executable': lammps_executable,
        'potential_file': potential_file,
        'lammps_workdir': lammps_workdir,
        'lammps_timeout_seconds': lammps_timeout_seconds,
        'lammps_max_displacement_angstrom': lammps_max_displacement_angstrom,
        'lammps_xy_cap_angstrom': lammps_xy_cap_angstrom,
        'lammps_z_cap_angstrom': lammps_z_cap_angstrom,
    }


def align_ground_truth_to_sa_frame(
    gt_positions: np.ndarray,
    sa_positions: np.ndarray,
) -> np.ndarray:
    """
    Align ground truth to the SA input coordinate frame using 2D
    nearest-neighbor correspondence in xy, then optimal z translation.

    Returns: (N_gt, 3) array with z shifted to match SA frame midplane.
    Assumes xy are already in similar frames (both from same image).
    If xy offsets exist, also computes and applies xy translation.
    """
    from scipy.spatial import cKDTree

    gt = gt_positions.copy()

    # Step 1: align xy via centroids (rigid translation)
    xy_shift = sa_positions[:, :2].mean(axis=0) - gt[:, :2].mean(axis=0)
    gt[:, 0] += xy_shift[0]
    gt[:, 1] += xy_shift[1]

    # Step 2: match atoms by xy nearest neighbor
    tree = cKDTree(gt[:, :2])
    dists, idx = tree.query(sa_positions[:, :2], k=1)
    good = dists < 1.0  # within 1 Å xy

    if good.sum() < 10:
        print(f"  WARNING: only {good.sum()} atoms match xy within 1 A — "
              "alignment may be poor")

    # Step 3: fit z translation so matched atoms have same mean z
    z_shift = sa_positions[good, 2].mean() - gt[idx[good], 2].mean()
    gt[:, 2] += z_shift

    print(f"  Ground truth aligned: xy_shift={xy_shift}, z_shift={z_shift:.3f} A")
    print(f"  Matched atoms for alignment: {good.sum()} / {len(sa_positions)}")
    return gt


def run_sa_refinement_paper_aligned(
    config: dict,
    initial_positions: np.ndarray,
    target_image: np.ndarray,
    simulator,
    md_relaxer=None,
    ground_truth: np.ndarray = None,
):
    """
    Two-level SA+MD refinement matching paper Figure 8 structure.
    Outer loop: 4-6 iterations tracking chi2 and z-RMSD.
    Inner loop: standard SA with anisotropic Metropolis and geometric cooling.
    MD relaxation applied to each accepted candidate (if md_relaxer provided).
    """
    positions = initial_positions.copy()
    N = len(positions)

    if md_relaxer is None and config.get('md_mode') == 'lammps':
        from graphene3d.stage3.lammps_minimizer import LammpsMinimizer
        md_relaxer = LammpsMinimizer(
            lammps_executable=Path(config['lammps_executable']),
            potential_file=Path(config['potential_file']),
            working_dir=Path(config.get('lammps_workdir', 'runs/lammps_sa')),
            timeout_seconds=float(config.get('lammps_timeout_seconds', 15.0)),
            max_displacement_angstrom=float(
                config.get('lammps_max_displacement_angstrom', 0.5)
            ),
            xy_cap_angstrom=config.get('lammps_xy_cap_angstrom'),
            z_cap_angstrom=config.get('lammps_z_cap_angstrom'),
        )
        print(f"  LammpsMinimizer: exe={config['lammps_executable']}, "
              f"pot={config['potential_file']}")

    step_xy = config.get('step_size_xy_ang', 0.08)
    step_z  = config.get('step_size_z_ang',  0.25)
    max_outer = config.get('max_outer_iters', 6)
    max_inner = config.get('max_inner_steps', 600)
    outer_tol = config.get('outer_tol', 0.3)

    T0_cfg = config.get('initial_temperature', 'auto')
    if T0_cfg == 'auto':
        T0 = calibrate_initial_temperature(
            positions, target_image, simulator,
            step_xy=step_xy, step_z=step_z,
            md_relaxer=md_relaxer,
        )
    else:
        T0 = float(T0_cfg)

    # Alpha is fixed across all outers; T0 is recalibrated per outer.
    final_fraction = config.get('T_final_fraction', 0.01)
    alpha = final_fraction ** (1.0 / max_inner)
    print(f"  alpha = {alpha:.6f} (T drops by {final_fraction*100:.0f}x "
          f"over {max_inner} steps per outer; T0 recalibrated each outer)")

    if ground_truth is not None:
        ground_truth = align_ground_truth_to_sa_frame(ground_truth, positions)

    def z_rmsd(p, gt):
        if gt is None:
            return None
        from scipy.spatial import cKDTree
        tree = cKDTree(gt[:, :2])
        _, idx = tree.query(p[:, :2], k=1)
        return float(np.sqrt(np.mean((p[:, 2] - gt[idx, 2])**2)))

    history = {
        'outer_iter': [], 'chi2': [], 'z_rmsd': [],
        'acceptance': [], 'n_md_calls': [], 'n_md_failures': [],
    }

    print(f"\nStarting SA+MD refinement: {max_outer} outer x {max_inner} inner")
    print(f"initial T0 = {T0:.3e}, alpha = {alpha:.6f}, "
          f"step_xy = {step_xy}A, step_z = {step_z}A")

    chi2_init = simulator.chi2(positions, target_image)
    print(f"Initial chi2 = {chi2_init:.6f}, "
          f"initial z-RMSD = {z_rmsd(positions, ground_truth)}")

    for outer in range(max_outer):
        n_accepted = 0
        n_md_calls = 0
        n_md_fail  = 0

        # Recalibrate T0 from the current positions, but cap at T0_initial.
        # SA temperature must never increase between outer iterations — if the
        # recalibrated value would be larger (e.g. because SA is at a local
        # chi2 minimum where all samples are uphill), keep the previous T0.
        # This prevents overheating and chi2 wandering in later outers.
        T0_outer = min(
            T0,
            calibrate_initial_temperature(
                positions, target_image, simulator,
                step_xy=step_xy, step_z=step_z,
                md_relaxer=md_relaxer,
            ),
        )
        alpha_outer = final_fraction ** (1.0 / max_inner)
        T = T0_outer
        chi2_current = simulator.chi2(positions, target_image)

        for k in range(max_inner):
            T_frac = np.sqrt(max(T / T0, 1e-6))
            # Single-atom proposal
            atom_idx = np.random.randint(N)
            candidate = positions.copy()
            candidate[atom_idx, 0] += np.random.normal(0, step_xy * T_frac)
            candidate[atom_idx, 1] += np.random.normal(0, step_xy * T_frac)
            candidate[atom_idx, 2] += np.random.normal(0, step_z  * T_frac)

            # Project candidate onto physical manifold via MD relaxation.
            # Both positions and candidate are now in relaxed space, so
            # delta = chi2_cand - chi2_current is a fair comparison.
            if md_relaxer is not None:
                n_md_calls += 1
                try:
                    candidate = md_relaxer.relax(candidate)
                except Exception:
                    n_md_fail += 1

            chi2_cand = simulator.chi2(candidate, target_image)
            delta = chi2_cand - chi2_current

            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-12)):
                positions = candidate
                chi2_current = chi2_cand
                n_accepted += 1

            T *= alpha_outer

        chi2_outer = simulator.chi2(positions, target_image)
        z_rmsd_outer = z_rmsd(positions, ground_truth)
        acceptance = n_accepted / max_inner

        history['outer_iter'].append(outer)
        history['chi2'].append(chi2_outer)
        history['z_rmsd'].append(z_rmsd_outer)
        history['acceptance'].append(acceptance)
        history['n_md_calls'].append(n_md_calls)
        history['n_md_failures'].append(n_md_fail)

        z_str = f"{z_rmsd_outer:.4f}" if z_rmsd_outer is not None else "N/A"
        print(f"  outer {outer}: chi2={chi2_outer:.6f}, z_rmsd={z_str}A, "
              f"accept={acceptance:.2f}, T0={T0_outer:.3e}, "
              f"md={n_md_calls}/{n_md_fail}fail")

        if outer > 0:
            delta_outer = abs(history['chi2'][-1] - history['chi2'][-2])
            if delta_outer < outer_tol:
                print(f"  Converged at outer iter {outer}")
                break

    return positions, history


def run_sa_refinement(sa_input_path: str | Path = DEFAULT_SA_INPUT,
                      config: Optional[SAConfig] = None,
                      objective: Optional[Callable[[np.ndarray], float]] = None,
                      md_config: Optional[MDConfig] = None,
                      md_adapter: Optional[MDRelaxationAdapter] = None,
                      output_dir: Optional[str | Path] = None) -> SAResult:
    """
    Run the Stage 3 simulated annealing skeleton.

    Parameters
    ----------
    sa_input_path
        NPZ produced by Stage 2 export_stage2_for_sa().
    config
        SA loop parameters.
    objective
        Callable objective(coords_angstrom) -> energy.  If None, this uses
        ImageMismatchObjective with the configured TEM simulator adapter.
    md_config
        Optional MD relaxation configuration.  If None or disabled, the frozen
        SA baseline behavior is unchanged.
    md_adapter
        Optional adapter implementing MDRelaxationAdapter.  If md_config is
        enabled and no adapter is supplied, make_md_relaxation_adapter() is
        used.  The current real-MD backend is not implemented.
    output_dir
        Optional run directory.  If None, creates the next
        runs/sa_run_XXX/ directory automatically.  Outputs are written inside
        that run directory's outputs/ folder.
    """
    if config is None:
        config = SAConfig()
    if md_config is None:
        md_config = MDConfig()
    if md_adapter is None:
        md_adapter = make_md_relaxation_adapter(md_config)
    run_directory = _prepare_sa_run_directory(config, sa_input_path, output_dir)

    sa_input = load_stage2_sa_input(sa_input_path)
    rng = np.random.default_rng(config.random_seed)

    if objective is None:
        objective = _default_objective(sa_input, config)
        _log(config, f"Using ImageMismatchObjective with simulator_kind={config.simulator_kind}.")
        if config.simulator_kind == "gaussian_projection":
            _log(config, "  GaussianProjectionSimulator is a fast proxy; use simulator_kind='abtem' for the Python TEM path.")
        elif config.simulator_kind == "abtem":
            _log(config, "  AbTEMSimulatorAdapter will run the Python abTEM forward simulator.")
    regularizer = _structural_regularizer_from_config(config)

    original_xyz = sa_input.xyz_angstrom.copy()
    if config.enable_pre_sa_sanitization:
        current_xyz, sanitization_summary = sanitize_initial_structure(
            original_xyz,
            min_distance_angstrom=config.structural_min_distance_angstrom,
            max_passes=config.sanitization_max_passes,
            buffer_angstrom=config.sanitization_buffer_angstrom,
        )
    else:
        current_xyz = original_xyz.copy()
        sanitization_summary = {
            "enabled": False,
            "threshold_angstrom": float(config.structural_min_distance_angstrom),
            "target_distance_angstrom": float(
                config.structural_min_distance_angstrom + max(config.sanitization_buffer_angstrom, 0.0)
            ),
            "buffer_angstrom": float(max(config.sanitization_buffer_angstrom, 0.0)),
            "max_passes": int(config.sanitization_max_passes),
            "close_pairs_before": int(len(_close_pairs(current_xyz, config.structural_min_distance_angstrom))),
            "close_pairs_after": int(len(_close_pairs(current_xyz, config.structural_min_distance_angstrom))),
            "nearest_neighbor_min_before_angstrom": float(_nearest_neighbor_min(current_xyz)),
            "nearest_neighbor_min_after_angstrom": float(_nearest_neighbor_min(current_xyz)),
            "pair_corrections_applied": 0,
            "max_atom_correction_angstrom": 0.0,
            "mean_atom_correction_angstrom": 0.0,
        }
    best_xyz = current_xyz.copy()
    initial_xyz = current_xyz.copy()
    current_components = _evaluate_sa_objective_components(
        current_xyz,
        objective,
        regularizer,
        config,
    )
    current_objective = float(current_components["total_objective"])
    initial_objective = current_objective
    best_objective = current_objective
    initial_components = dict(current_components)
    best_components = dict(current_components)

    objective_history = []
    acceptance_history = []
    temperature = float(config.initial_temperature)

    _log(config, "Stage 3 SA skeleton")
    _log(config, f"  run directory: {run_directory.run_dir}")
    _log(config, f"  atoms: {len(current_xyz)}")
    _log(
        config,
        "  sanitization close pairs: "
        f"{sanitization_summary['close_pairs_before']} -> {sanitization_summary['close_pairs_after']}"
    )
    _log(config, f"  iterations: {config.n_iterations}")
    _log(config, f"  initial objective: {initial_objective:.6f}")
    if config.enable_structural_regularization and config.structural_regularization_weight != 0.0:
        _log(
            config,
            "  structural regularization: "
            f"weight={config.structural_regularization_weight:.3e}, "
            f"initial contribution={current_components['structural_regularization']:.6f}"
        )

    for iteration in range(config.n_iterations):
        proposal, atom_index, delta = _propose_single_atom_move(current_xyz, rng, config)
        md_relaxation_applied = _should_apply_md_relaxation(iteration, md_config)
        md_pair_corrections = 0
        md_bond_corrections = 0
        if md_relaxation_applied:
            proposal_before_md = proposal.copy()
            proposal = md_adapter.relax(proposal, md_config)
            md_adapter_summary = getattr(md_adapter, "last_summary", {})
            md_pair_corrections = int(md_adapter_summary.get("pair_corrections", 0))
            md_bond_corrections = int(md_adapter_summary.get("bond_corrections", 0))
            md_correction = proposal - proposal_before_md
            md_atom_correction = np.linalg.norm(md_correction, axis=1)
            md_correction_magnitude = float(np.linalg.norm(md_correction))
            md_max_atom_correction = float(np.max(md_atom_correction)) if len(md_atom_correction) else 0.0
            md_corrected = bool(md_correction_magnitude > 1e-12)
        else:
            md_correction_magnitude = 0.0
            md_max_atom_correction = 0.0
            md_corrected = False
        objective_before = current_objective
        components_before = current_components
        structural_rejected, current_nn, proposal_nn = _violates_structural_distance(
            current_xyz,
            proposal,
            atom_index,
            config,
        )
        objective_evaluated = not structural_rejected

        if structural_rejected:
            proposal_objective = objective_before
            proposal_components = dict(components_before)
            d_obj = 0.0
            accepted = False
        else:
            proposal_components = _evaluate_sa_objective_components(
                proposal,
                objective,
                regularizer,
                config,
            )
            proposal_objective = float(proposal_components["total_objective"])
            d_obj = proposal_objective - objective_before

            if d_obj <= 0:
                accepted = True
            else:
                accept_prob = np.exp(-d_obj / max(temperature, 1e-12))
                accepted = bool(rng.random() < accept_prob)

        if accepted:
            current_xyz = proposal
            current_objective = proposal_objective
            current_components = proposal_components
            if current_objective < best_objective:
                best_objective = current_objective
                best_xyz = current_xyz.copy()
                best_components = dict(current_components)

        objective_history.append([
            float(iteration),
            temperature,
            current_objective,
            best_objective,
            current_components["image_objective"],
            best_components["image_objective"],
            current_components["structural_regularization"],
            best_components["structural_regularization"],
        ])
        acceptance_history.append([
            float(iteration),
            float(accepted),
            float(atom_index),
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
            objective_before,
            proposal_objective,
            d_obj,
            float((d_obj < 0) and objective_evaluated),
            float((d_obj > 0) and objective_evaluated),
            float(np.linalg.norm(delta)),
            float(structural_rejected),
            current_nn,
            proposal_nn,
            float(objective_evaluated),
            components_before["image_objective"],
            proposal_components["image_objective"],
            proposal_components["image_objective"] - components_before["image_objective"],
            components_before["structural_regularization"],
            proposal_components["structural_regularization"],
            proposal_components["structural_regularization"] - components_before["structural_regularization"],
            components_before["structural_regularization_raw"],
            proposal_components["structural_regularization_raw"],
            float(md_relaxation_applied),
            float(md_corrected),
            md_correction_magnitude,
            md_max_atom_correction,
            float(md_pair_corrections),
            float(md_bond_corrections),
        ])

        temperature *= config.cooling_rate

    objective_history_arr = np.array(objective_history, dtype=np.float64)
    acceptance_history_arr = np.array(acceptance_history, dtype=np.float64)
    objective_component_summary = {
        "structural_regularization_enabled": bool(config.enable_structural_regularization),
        "structural_regularization_weight": float(config.structural_regularization_weight),
        "initial": initial_components,
        "current": current_components,
        "best": best_components,
        "note": (
            "total_objective = image_objective + structural_regularization. "
            "The structural term is a lightweight nearest-neighbor geometry proxy, not MD/LAMMPS."
        ),
    }
    md_relaxation_summary = _md_relaxation_diagnostics_summary(
        acceptance_history_arr,
        md_config,
    )

    result = SAResult(
        initial_xyz=initial_xyz,
        original_xyz=original_xyz,
        current_xyz=current_xyz,
        best_xyz=best_xyz,
        atom_id=sa_input.atom_id,
        element=sa_input.element,
        source_label=sa_input.source_label,
        initializer_weight=sa_input.initializer_weight,
        initial_objective=initial_objective,
        current_objective=current_objective,
        best_objective=best_objective,
        objective_history=objective_history_arr,
        acceptance_history=acceptance_history_arr,
        sanitization_summary=sanitization_summary,
        objective_component_summary=objective_component_summary,
        md_relaxation_summary=md_relaxation_summary,
        output_paths={},
    )

    if config.save_outputs:
        result.output_paths = _save_sa_outputs(
            run_directory.outputs_dir,
            config,
            sa_input,
            result,
            run_directory=run_directory,
        )
    else:
        result.output_paths = {
            "run_dir": str(run_directory.run_dir),
            "working_dir": str(run_directory.working_dir),
            "outputs_dir": str(run_directory.outputs_dir),
            "logs_dir": str(run_directory.logs_dir),
            "config_json": str(run_directory.config_path),
        }

    accepted_count = int(np.sum(acceptance_history_arr[:, 1])) if len(acceptance_history_arr) else 0
    proposal_summary = _proposal_diagnostics_summary(acceptance_history_arr)
    _log(config, f"  best objective: {best_objective:.6f}")
    _log(config, f"  best image objective: {best_components['image_objective']:.6f}")
    _log(config, f"  best structural contribution: {best_components['structural_regularization']:.6f}")
    _log(config, f"  acceptance rate: {accepted_count / max(config.n_iterations, 1):.3f}")
    _log(config, f"  improving proposals: {proposal_summary['fraction_improving_proposals']:.3f}")
    _log(config, f"  worsening proposals accepted: {proposal_summary['fraction_worsening_proposals_accepted']:.3f}")
    _log(config, f"  mean |proposal|: {proposal_summary['mean_abs_proposal_size']:.6f}")
    _log(config, f"  mean |delta objective|: {proposal_summary['mean_abs_delta_objective']:.6f}")
    _log(config, f"  structural rejections: {int(proposal_summary['structural_rejection_count'])}")
    if md_relaxation_summary["enabled"]:
        _log(
            config,
            "  MD adapter corrections: "
            f"{md_relaxation_summary['proposals_corrected']} proposals, "
            f"mean |correction|={md_relaxation_summary['mean_correction_magnitude_angstrom']:.6f} A"
        )
    _log(config, "Stage 3 SA skeleton complete.")
    return result


def _score_tuning_row(row: dict[str, float]) -> float:
    """Score a tiny tuning run for proposal-scale informativeness."""
    acceptance = row["acceptance_rate"]
    worsening_accept = row["fraction_worsening_proposals_accepted"]
    acceptance_penalty = abs(acceptance - 0.6)
    worsening_penalty = abs(worsening_accept - 0.5)
    improvement_bonus = max(row["best_objective_improvement"], 0.0) * 1000.0
    delta_bonus = row["mean_abs_delta_objective"] * 100.0
    return improvement_bonus + delta_bonus - acceptance_penalty - worsening_penalty


def _write_tuning_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact tuning table without adding a pandas dependency."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for row in rows:
            values = []
            for field in fieldnames:
                value = row[field]
                if isinstance(value, str):
                    values.append(value.replace(",", ";"))
                elif isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.12g}")
                else:
                    values.append(str(value))
            fh.write(",".join(values) + "\n")


def run_sa_tuning_grid(sa_input_path: str | Path = DEFAULT_SA_INPUT,
                       target_image_path: str | Path | None = None,
                       n_iterations: int = 10,
                       settings: Optional[list[dict[str, float]]] = None) -> list[dict[str, float]]:
    """
    Run a tiny grid over proposal step sizes and initial temperatures.

    This helper is diagnostic only.  It reuses the normal SA loop and simulator
    path without changing SA mechanics, then reports proposal behavior for each
    setting.
    """
    if settings is None:
        settings = [
            {"step_size_xy": 0.03, "step_size_z": 0.03, "initial_temperature": 1e-4},
            {"step_size_xy": 0.06, "step_size_z": 0.06, "initial_temperature": 1e-4},
            {"step_size_xy": 0.10, "step_size_z": 0.10, "initial_temperature": 1e-4},
            {"step_size_xy": 0.10, "step_size_z": 0.10, "initial_temperature": 5e-5},
            {"step_size_xy": 0.15, "step_size_z": 0.10, "initial_temperature": 1e-4},
            {"step_size_xy": 0.20, "step_size_z": 0.10, "initial_temperature": 5e-5},
        ]

    base_config = SAConfig(
        n_iterations=n_iterations,
        output_prefix="stage3_sa_tuning_probe",
        save_outputs=False,
        verbose=False,
    )
    if target_image_path is not None:
        base_config.target_image_path = str(target_image_path)

    rows: list[dict[str, float]] = []
    for idx, setting in enumerate(settings, start=1):
        config = replace(
            base_config,
            step_size_xy=float(setting["step_size_xy"]),
            step_size_z=float(setting["step_size_z"]),
            initial_temperature=float(setting["initial_temperature"]),
            random_seed=int(setting.get("random_seed", 0)),
        )
        result = run_sa_refinement(sa_input_path=sa_input_path, config=config)
        accepted = int(np.sum(result.acceptance_history[:, 1])) if len(result.acceptance_history) else 0
        proposal_summary = _proposal_diagnostics_summary(result.acceptance_history)
        row = {
            "setting": float(idx),
            "step_size_xy": config.step_size_xy,
            "step_size_z": config.step_size_z,
            "initial_temperature": config.initial_temperature,
            "n_iterations": float(config.n_iterations),
            "acceptance_rate": float(accepted / max(config.n_iterations, 1)),
            "fraction_worsening_proposals_accepted": proposal_summary[
                "fraction_worsening_proposals_accepted"
            ],
            "mean_abs_delta_objective": proposal_summary["mean_abs_delta_objective"],
            "mean_abs_proposal_size": proposal_summary["mean_abs_proposal_size"],
            "initial_objective": float(result.initial_objective),
            "best_objective": float(result.best_objective),
            "best_objective_improvement": float(result.initial_objective - result.best_objective),
            "run_dir": result.output_paths.get("run_dir", ""),
        }
        rows.append(row)

    print("Stage 3 SA proposal tuning grid")
    print("===============================")
    print(
        "idx  step_xy  step_z   temp       acc   worsen_acc  "
        "mean|dE|     best_impr"
    )
    for row in rows:
        print(
            f"{int(row['setting']):>3}  "
            f"{row['step_size_xy']:>7.3f}  "
            f"{row['step_size_z']:>6.3f}  "
            f"{row['initial_temperature']:>9.1e}  "
            f"{row['acceptance_rate']:>5.3f}  "
            f"{row['fraction_worsening_proposals_accepted']:>10.3f}  "
            f"{row['mean_abs_delta_objective']:>10.6f}  "
            f"{row['best_objective_improvement']:>10.6f}"
        )

    recommended = max(rows, key=_score_tuning_row)
    print("")
    print("Recommended next smoke-test setting")
    print("===================================")
    print(f"step_size_xy        : {recommended['step_size_xy']:.3f}")
    print(f"step_size_z         : {recommended['step_size_z']:.3f}")
    print(f"initial_temperature : {recommended['initial_temperature']:.1e}")
    print(
        "reason              : balances lower worsening acceptance with "
        "larger objective movement in this tiny diagnostic grid"
    )
    return rows


def run_sa_anisotropic_step_tuning(
    sa_input_path: str | Path = DEFAULT_SA_INPUT,
    target_image_path: str | Path | None = None,
    n_iterations: int = 50,
    initial_temperature: float = 1.0e-4,
    simulator_z_contrast_scale: Optional[float] = None,
    settings: Optional[list[dict[str, float]]] = None,
    output_dir: str | Path = DEFAULT_TUNING_DIR,
) -> list[dict[str, Any]]:
    """
    Compare short SA runs with different x/y and z proposal scales.

    This helper is diagnostic only.  It reuses the same SA loop, temperature,
    pre-SA sanitization, and structural rejection rule while varying only
    step_size_xy and step_size_z.
    """
    if settings is None:
        settings = [
            {"step_size_xy": 0.10, "step_size_z": 0.06},
            {"step_size_xy": 0.10, "step_size_z": 0.08},
            {"step_size_xy": 0.10, "step_size_z": 0.10},
            {"step_size_xy": 0.10, "step_size_z": 0.12},
            {"step_size_xy": 0.08, "step_size_z": 0.10},
        ]

    z_scale = SAConfig().simulator_z_contrast_scale if simulator_z_contrast_scale is None else float(simulator_z_contrast_scale)
    tuning_dir = Path(output_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, setting in enumerate(settings, start=1):
        config = SAConfig(
            n_iterations=int(setting.get("n_iterations", n_iterations)),
            initial_temperature=float(setting.get("initial_temperature", initial_temperature)),
            cooling_rate=float(setting.get("cooling_rate", 0.995)),
            step_size_xy=float(setting["step_size_xy"]),
            step_size_z=float(setting["step_size_z"]),
            random_seed=int(setting.get("random_seed", 0)),
            enable_pre_sa_sanitization=True,
            enable_structural_rejection=True,
            structural_min_distance_angstrom=float(setting.get("structural_min_distance_angstrom", 0.8)),
            simulator_z_contrast_scale=float(setting.get("simulator_z_contrast_scale", z_scale)),
            output_prefix=f"stage3_sa_anisotropic_tuning_{idx:02d}",
            target_image_path=str(target_image_path) if target_image_path is not None else SAConfig.target_image_path,
            save_outputs=True,
            verbose=False,
        )
        result = run_sa_refinement(sa_input_path=sa_input_path, config=config)
        accepted = int(np.sum(result.acceptance_history[:, 1])) if len(result.acceptance_history) else 0
        proposal_summary = _proposal_diagnostics_summary(result.acceptance_history)
        nn_min = _nearest_neighbor_min(result.best_xyz)
        close_count = len(_close_pairs(result.best_xyz, config.structural_min_distance_angstrom))
        row = {
            "setting": idx,
            "step_size_xy": config.step_size_xy,
            "step_size_z": config.step_size_z,
            "initial_temperature": config.initial_temperature,
            "simulator_z_contrast_scale": config.simulator_z_contrast_scale,
            "n_iterations": config.n_iterations,
            "acceptance_rate": float(accepted / max(config.n_iterations, 1)),
            "fraction_worsening_proposals_accepted": proposal_summary[
                "fraction_worsening_proposals_accepted"
            ],
            "mean_abs_delta_objective": proposal_summary["mean_abs_delta_objective"],
            "mean_abs_proposal_size": proposal_summary["mean_abs_proposal_size"],
            "initial_objective": float(result.initial_objective),
            "best_objective": float(result.best_objective),
            "best_objective_improvement": float(result.initial_objective - result.best_objective),
            "nearest_neighbor_min_after_run": float(nn_min),
            "close_pairs_below_threshold_after_run": int(close_count),
            "run_dir": result.output_paths.get("run_dir", ""),
            "summary_json": result.output_paths.get("summary_json", ""),
        }
        rows.append(row)

    viable_rows = [row for row in rows if int(row["close_pairs_below_threshold_after_run"]) == 0]
    recommended = max(viable_rows or rows, key=_score_tuning_row)
    payload = {
        "n_iterations": int(n_iterations),
        "initial_temperature": float(initial_temperature),
        "simulator_z_contrast_scale": float(z_scale),
        "recommendation": recommended,
        "rows": rows,
        "note": (
            "Short anisotropic tuning runs compare proposal scales only. "
            "Use the recommendation as a next longer-run candidate, not as a final optimum."
        ),
        "z_sensitivity_note": (
            "GaussianProjectionSimulator z sensitivity was disabled with simulator_z_contrast_scale=0.0. "
            "Under that setting, z step-size changes mainly affect structural sanity and coordinate drift, "
            "not objective improvement."
            if float(z_scale) == 0.0
            else (
                "GaussianProjectionSimulator used weak bounded relative-z contrast. "
                "This is a debug proxy for testing SA response to z changes, not the final physics model."
            )
        ),
    }
    json_path = tuning_dir / "stage3_sa_anisotropic_step_tuning_summary.json"
    csv_path = tuning_dir / "stage3_sa_anisotropic_step_tuning_summary.csv"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _write_tuning_rows_csv(csv_path, rows)

    print("Stage 3 SA anisotropic step tuning")
    print("==================================")
    print(
        "idx  step_xy  step_z   acc   worsen_acc  best_impr   nn_min   close_pairs"
    )
    for row in rows:
        print(
            f"{int(row['setting']):>3}  "
            f"{row['step_size_xy']:>7.3f}  "
            f"{row['step_size_z']:>6.3f}  "
            f"{row['acceptance_rate']:>5.3f}  "
            f"{row['fraction_worsening_proposals_accepted']:>10.3f}  "
            f"{row['best_objective_improvement']:>10.6f}  "
            f"{row['nearest_neighbor_min_after_run']:>7.3f}  "
            f"{int(row['close_pairs_below_threshold_after_run']):>11}"
        )

    print("")
    print("Recommended anisotropic next-run setting")
    print("========================================")
    print(f"step_size_xy        : {recommended['step_size_xy']:.3f}")
    print(f"step_size_z         : {recommended['step_size_z']:.3f}")
    print(f"initial_temperature : {recommended['initial_temperature']:.1e}")
    if float(z_scale) == 0.0:
        print("z sensitivity       : disabled in the current Gaussian projection proxy")
    else:
        print(f"z sensitivity       : weak debug proxy, scale={z_scale:.3f}")
    print(f"summary JSON        : {json_path}")
    print(f"summary CSV         : {csv_path}")
    return rows


def _score_regularization_weight_row(row: dict[str, Any]) -> float:
    """Score a short structural-regularization weight probe."""
    if int(row["close_atoms_below_0p8_after_run"]) > 0:
        return -1.0e9
    image_bonus = max(float(row["image_objective_improvement"]), 0.0) * 1000.0
    total_bonus = max(float(row["best_total_objective_improvement"]), 0.0) * 250.0
    acceptance_penalty = abs(float(row["acceptance_rate"]) - 0.6)
    weight_penalty = float(row["structural_regularization_weight"]) * 10.0
    structural_fraction = abs(float(row["best_structural_contribution"])) / (
        abs(float(row["best_total_objective"])) + 1e-12
    )
    structural_penalty = structural_fraction * 100.0
    return image_bonus + total_bonus - acceptance_penalty - weight_penalty - structural_penalty


def run_sa_regularization_weight_tuning(
    sa_input_path: str | Path = DEFAULT_SA_INPUT,
    target_image_path: str | Path | None = None,
    n_iterations: int = 50,
    weights: Optional[list[float]] = None,
    step_size_xy: float = 0.10,
    step_size_z: float = 0.06,
    initial_temperature: float = 1.0e-4,
    simulator_z_contrast_scale: float = 0.08,
    output_dir: str | Path = DEFAULT_TUNING_DIR,
) -> list[dict[str, Any]]:
    """
    Compare short SA runs with different structural-regularization weights.

    This diagnostic varies only the lightweight structural regularization
    weight while keeping proposal scale, temperature, simulator, sanitization,
    and structural rejection comparable.
    """
    if weights is None:
        weights = [0.0, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2]

    tuning_dir = Path(output_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, weight in enumerate(weights, start=1):
        weight = float(weight)
        config = SAConfig(
            n_iterations=int(n_iterations),
            initial_temperature=float(initial_temperature),
            cooling_rate=0.995,
            step_size_xy=float(step_size_xy),
            step_size_z=float(step_size_z),
            simulator_z_contrast_scale=float(simulator_z_contrast_scale),
            random_seed=0,
            enable_pre_sa_sanitization=True,
            enable_structural_rejection=True,
            structural_min_distance_angstrom=0.8,
            enable_structural_regularization=weight > 0.0,
            structural_regularization_weight=weight,
            output_prefix=f"stage3_sa_regularization_weight_tuning_{idx:02d}",
            target_image_path=str(target_image_path) if target_image_path is not None else SAConfig.target_image_path,
            save_outputs=True,
            verbose=False,
        )
        result = run_sa_refinement(sa_input_path=sa_input_path, config=config)
        accepted = int(np.sum(result.acceptance_history[:, 1])) if len(result.acceptance_history) else 0
        proposal_summary = _proposal_diagnostics_summary(result.acceptance_history)
        components = result.objective_component_summary
        initial_components = components["initial"]
        best_components = components["best"]
        nn_min = _nearest_neighbor_min(result.best_xyz)
        close_count = len(_close_pairs(result.best_xyz, config.structural_min_distance_angstrom))
        row = {
            "setting": idx,
            "structural_regularization_weight": weight,
            "enabled": bool(weight > 0.0),
            "n_iterations": int(config.n_iterations),
            "step_size_xy": float(config.step_size_xy),
            "step_size_z": float(config.step_size_z),
            "initial_temperature": float(config.initial_temperature),
            "simulator_z_contrast_scale": float(config.simulator_z_contrast_scale),
            "acceptance_rate": float(accepted / max(config.n_iterations, 1)),
            "fraction_worsening_proposals_accepted": proposal_summary[
                "fraction_worsening_proposals_accepted"
            ],
            "initial_total_objective": float(result.initial_objective),
            "best_total_objective": float(result.best_objective),
            "best_total_objective_improvement": float(result.initial_objective - result.best_objective),
            "initial_image_objective": float(initial_components["image_objective"]),
            "best_image_objective": float(best_components["image_objective"]),
            "image_objective_improvement": float(
                initial_components["image_objective"] - best_components["image_objective"]
            ),
            "initial_structural_contribution": float(initial_components["structural_regularization"]),
            "best_structural_contribution": float(best_components["structural_regularization"]),
            "best_raw_structural_penalty": float(best_components["structural_regularization_raw"]),
            "nearest_neighbor_min_after_run": float(nn_min),
            "close_atoms_below_0p8_after_run": int(close_count),
            "run_dir": result.output_paths.get("run_dir", ""),
            "summary_json": result.output_paths.get("summary_json", ""),
        }
        rows.append(row)

    viable_rows = [row for row in rows if int(row["close_atoms_below_0p8_after_run"]) == 0]
    metric_recommended = max(viable_rows or rows, key=_score_regularization_weight_row)
    nonzero_viable = [
        row for row in viable_rows
        if float(row["structural_regularization_weight"]) > 0.0
    ]
    if nonzero_viable:
        best_image_improvement = max(float(row["image_objective_improvement"]) for row in viable_rows)
        near_best = [
            row for row in nonzero_viable
            if float(row["image_objective_improvement"]) >= best_image_improvement - 1e-6
        ]
        visible_regularized = [
            row for row in near_best
            if abs(float(row["best_structural_contribution"])) >= 1e-6
        ]
        recommended = min(
            visible_regularized or near_best or nonzero_viable,
            key=lambda row: float(row["structural_regularization_weight"]),
        )
    else:
        recommended = metric_recommended
    payload = {
        "n_iterations": int(n_iterations),
        "step_size_xy": float(step_size_xy),
        "step_size_z": float(step_size_z),
        "initial_temperature": float(initial_temperature),
        "simulator_z_contrast_scale": float(simulator_z_contrast_scale),
        "metric_only_recommendation": metric_recommended,
        "recommendation": recommended,
        "rows": rows,
        "note": (
            "Short regularization tuning runs compare structural weights only. "
            "The regularizer is a lightweight nearest-neighbor geometry proxy, not MD/LAMMPS. "
            "metric_only_recommendation may be 0.0 when the current structure is already safe; "
            "recommendation prefers a gentle nonzero weight if it preserves image improvement."
        ),
    }
    json_path = tuning_dir / "stage3_sa_regularization_weight_tuning_summary.json"
    csv_path = tuning_dir / "stage3_sa_regularization_weight_tuning_summary.csv"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _write_tuning_rows_csv(csv_path, rows)

    print("Stage 3 SA structural-regularization weight tuning")
    print("==================================================")
    print(
        "idx  weight     total_impr  image_impr  struct_best  acc   nn_min  close"
    )
    for row in rows:
        print(
            f"{int(row['setting']):>3}  "
            f"{row['structural_regularization_weight']:>9.1e}  "
            f"{row['best_total_objective_improvement']:>10.6f}  "
            f"{row['image_objective_improvement']:>10.6f}  "
            f"{row['best_structural_contribution']:>11.6e}  "
            f"{row['acceptance_rate']:>5.3f}  "
            f"{row['nearest_neighbor_min_after_run']:>7.3f}  "
            f"{int(row['close_atoms_below_0p8_after_run']):>5}"
        )

    print("")
    print("Recommended structural-regularization weight")
    print("============================================")
    print(f"weight       : {recommended['structural_regularization_weight']:.1e}")
    print(f"metric-only  : {metric_recommended['structural_regularization_weight']:.1e}")
    print(f"summary JSON : {json_path}")
    print(f"summary CSV  : {csv_path}")
    return rows


def _latest_sa_best_coordinates_npz(runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Find the newest SA best-coordinate NPZ under outputs/stage3/runs."""
    candidates = sorted(runs_dir.glob("sa_run_*/outputs/*_best_coordinates.npz"))
    if not candidates:
        raise FileNotFoundError(f"No SA best-coordinate NPZ files found in {runs_dir}")
    return candidates[-1]


def _history_csv_for_best_coordinates(best_coordinates_npz: Path) -> Path:
    """Infer the matching SA history CSV for a best-coordinate NPZ."""
    stem = best_coordinates_npz.stem
    suffix = "_best_coordinates"
    if stem.endswith(suffix):
        history_name = f"{stem[:-len(suffix)]}_history.csv"
    else:
        history_name = f"{stem}_history.csv"
    history_path = best_coordinates_npz.parent / history_name
    if not history_path.exists():
        raise FileNotFoundError(
            f"Could not find matching history CSV for {best_coordinates_npz}: {history_path}"
        )
    return history_path


def analyze_sa_coordinate_update(best_coordinates_npz: str | Path | None = None,
                                 close_distance_angstrom: float = 0.8,
                                 save_plot: bool = True) -> dict[str, Any]:
    """
    Analyze initial-vs-best coordinate changes for a completed SA run.

    This is a post-run diagnostic only.  It does not modify the SA result or
    enforce constraints.
    """
    npz_path = Path(best_coordinates_npz) if best_coordinates_npz is not None else _latest_sa_best_coordinates_npz()
    data = np.load(npz_path, allow_pickle=False)
    initial = np.asarray(data["initial_xyz_angstrom"], dtype=np.float64)
    best = np.asarray(data["best_xyz_angstrom"], dtype=np.float64)
    if initial.shape != best.shape or initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N, 3) coordinate arrays, got {initial.shape} and {best.shape}"
        )

    delta = best - initial
    displacement = np.linalg.norm(delta, axis=1)

    try:
        from scipy.spatial import KDTree
    except ImportError as exc:
        raise ImportError("scipy is required for nearest-neighbor coordinate diagnostics.") from exc

    tree = KDTree(best)
    nn_distances = tree.query(best, k=2)[0][:, 1] if len(best) > 1 else np.array([], dtype=np.float64)
    close_count = int(np.sum(nn_distances < close_distance_angstrom)) if len(nn_distances) else 0

    summary = {
        "best_coordinates_npz": str(npz_path),
        "n_atoms": int(len(best)),
        "mean_atomic_displacement_angstrom": float(np.mean(displacement)),
        "max_atomic_displacement_angstrom": float(np.max(displacement)),
        "mean_abs_dx_angstrom": float(np.mean(np.abs(delta[:, 0]))),
        "mean_abs_dy_angstrom": float(np.mean(np.abs(delta[:, 1]))),
        "mean_abs_dz_angstrom": float(np.mean(np.abs(delta[:, 2]))),
        "nearest_neighbor_min_angstrom": float(np.min(nn_distances)) if len(nn_distances) else np.nan,
        "nearest_neighbor_mean_angstrom": float(np.mean(nn_distances)) if len(nn_distances) else np.nan,
        "unrealistically_close_threshold_angstrom": float(close_distance_angstrom),
        "unrealistically_close_atom_count": close_count,
        "has_unrealistically_close_atoms": bool(close_count > 0),
    }

    output_dir = npz_path.parent
    summary_path = output_dir / f"{npz_path.stem}_coordinate_diagnostics.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_json"] = str(summary_path)

    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required to save displacement diagnostics plots.") from exc

        plot_path = output_dir / f"{npz_path.stem}_displacement_histogram.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(displacement, bins=30, color="tab:blue", alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Initial-to-best displacement (Angstrom)")
        ax.set_ylabel("Atom count")
        ax.set_title("Stage 3 SA Coordinate Displacements")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        summary["displacement_histogram"] = str(plot_path)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    print("Stage 3 coordinate-update diagnostics")
    print("=====================================")
    print(f"best coordinates : {npz_path}")
    print(f"atoms            : {summary['n_atoms']}")
    print(f"mean displacement: {summary['mean_atomic_displacement_angstrom']:.6f} A")
    print(f"max displacement : {summary['max_atomic_displacement_angstrom']:.6f} A")
    print(f"mean |dx|        : {summary['mean_abs_dx_angstrom']:.6f} A")
    print(f"mean |dy|        : {summary['mean_abs_dy_angstrom']:.6f} A")
    print(f"mean |dz|        : {summary['mean_abs_dz_angstrom']:.6f} A")
    print(f"NN min / mean    : {summary['nearest_neighbor_min_angstrom']:.6f} / {summary['nearest_neighbor_mean_angstrom']:.6f} A")
    print(f"close atoms < {close_distance_angstrom:.3f} A: {close_count}")
    print(f"summary          : {summary_path}")
    if save_plot:
        print(f"histogram        : {summary.get('displacement_histogram', '')}")
    return summary


def analyze_sa_coordinate_directions(best_coordinates_npz: str | Path | None = None,
                                     history_csv: str | Path | None = None,
                                     save_plot: bool = True) -> dict[str, Any]:
    """
    Analyze whether accepted SA moves are mostly x/y or z directed.

    This is a post-run diagnostic only.  Because the SA proposal perturbs x,
    y, and z together for one atom, these summaries describe directional
    proposal magnitudes rather than independent causal objective attribution.
    """
    npz_path = Path(best_coordinates_npz) if best_coordinates_npz is not None else _latest_sa_best_coordinates_npz()
    history_path = Path(history_csv) if history_csv is not None else _history_csv_for_best_coordinates(npz_path)

    data = np.load(npz_path, allow_pickle=False)
    initial = np.asarray(data["initial_xyz_angstrom"], dtype=np.float64)
    best = np.asarray(data["best_xyz_angstrom"], dtype=np.float64)
    if initial.shape != best.shape or initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N, 3) coordinate arrays, got {initial.shape} and {best.shape}"
        )

    history = np.genfromtxt(history_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    history = np.atleast_1d(history)
    required = {
        "delta_x",
        "delta_y",
        "delta_z",
        "accepted",
        "improving",
        "worsening",
        "objective_evaluated",
        "delta_objective",
    }
    missing = sorted(required - set(history.dtype.names or ()))
    if missing:
        raise ValueError(f"History CSV is missing required columns: {missing}")

    accepted = history["accepted"].astype(bool)
    improving = history["improving"].astype(bool)
    worsening = history["worsening"].astype(bool)
    objective_evaluated = history["objective_evaluated"].astype(bool)
    accepted_improving = accepted & improving & objective_evaluated
    accepted_worsening = accepted & worsening & objective_evaluated

    proposal_delta = np.column_stack([
        np.asarray(history["delta_x"], dtype=np.float64),
        np.asarray(history["delta_y"], dtype=np.float64),
        np.asarray(history["delta_z"], dtype=np.float64),
    ])
    final_delta = best - initial

    def _direction_stats(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {
                "count": 0.0,
                "mean_abs_dx_angstrom": 0.0,
                "mean_abs_dy_angstrom": 0.0,
                "mean_abs_dz_angstrom": 0.0,
                "mean_xy_magnitude_angstrom": 0.0,
                "mean_z_abs_angstrom": 0.0,
                "xy_fraction_of_abs_motion": 0.0,
                "z_fraction_of_abs_motion": 0.0,
            }
        selected = proposal_delta[mask]
        abs_selected = np.abs(selected)
        xy_mag = np.linalg.norm(selected[:, :2], axis=1)
        abs_xy_sum = float(np.sum(abs_selected[:, 0] + abs_selected[:, 1]))
        abs_z_sum = float(np.sum(abs_selected[:, 2]))
        total = abs_xy_sum + abs_z_sum
        return {
            "count": float(len(selected)),
            "mean_abs_dx_angstrom": float(np.mean(abs_selected[:, 0])),
            "mean_abs_dy_angstrom": float(np.mean(abs_selected[:, 1])),
            "mean_abs_dz_angstrom": float(np.mean(abs_selected[:, 2])),
            "mean_xy_magnitude_angstrom": float(np.mean(xy_mag)),
            "mean_z_abs_angstrom": float(np.mean(abs_selected[:, 2])),
            "xy_fraction_of_abs_motion": float(abs_xy_sum / total) if total else 0.0,
            "z_fraction_of_abs_motion": float(abs_z_sum / total) if total else 0.0,
        }

    final_abs_delta = np.abs(final_delta)
    accepted_improving_stats = _direction_stats(accepted_improving)
    xy_fraction = accepted_improving_stats["xy_fraction_of_abs_motion"]
    z_fraction = accepted_improving_stats["z_fraction_of_abs_motion"]
    if accepted_improving_stats["count"] == 0:
        improving_direction_assessment = "no accepted improving proposals"
    elif xy_fraction > z_fraction * 1.25:
        improving_direction_assessment = "accepted improving proposals are dominated by x/y motion"
    elif z_fraction > xy_fraction * 1.25:
        improving_direction_assessment = "accepted improving proposals are dominated by z motion"
    else:
        improving_direction_assessment = "accepted improving proposals are balanced between x/y and z motion"

    summary = {
        "best_coordinates_npz": str(npz_path),
        "history_csv": str(history_path),
        "n_atoms": int(len(best)),
        "n_iterations": int(len(history)),
        "final_coordinate_displacement": {
            "mean_abs_dx_angstrom": float(np.mean(final_abs_delta[:, 0])),
            "mean_abs_dy_angstrom": float(np.mean(final_abs_delta[:, 1])),
            "mean_abs_dz_angstrom": float(np.mean(final_abs_delta[:, 2])),
            "mean_xy_magnitude_angstrom": float(np.mean(np.linalg.norm(final_delta[:, :2], axis=1))),
            "mean_z_abs_angstrom": float(np.mean(final_abs_delta[:, 2])),
        },
        "accepted_proposals": _direction_stats(accepted),
        "accepted_improving_proposals": accepted_improving_stats,
        "accepted_worsening_proposals": _direction_stats(accepted_worsening),
        "all_evaluated_proposals": _direction_stats(objective_evaluated),
        "improving_direction_assessment": improving_direction_assessment,
        "note": (
            "x/y/z are proposed together for one atom, so these are directional "
            "magnitude diagnostics rather than independent objective attribution."
        ),
    }

    output_dir = npz_path.parent
    summary_path = output_dir / f"{npz_path.stem}_direction_diagnostics.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_json"] = str(summary_path)

    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required to save direction diagnostics plots.") from exc

        plot_path = output_dir / f"{npz_path.stem}_direction_displacement_histogram.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = 30
        ax.hist(final_abs_delta[:, 0], bins=bins, alpha=0.55, label="|dx|", color="tab:blue")
        ax.hist(final_abs_delta[:, 1], bins=bins, alpha=0.55, label="|dy|", color="tab:green")
        ax.hist(final_abs_delta[:, 2], bins=bins, alpha=0.55, label="|dz|", color="tab:red")
        ax.set_xlabel("Initial-to-best absolute displacement (Angstrom)")
        ax.set_ylabel("Atom count")
        ax.set_title("Stage 3 SA Directional Coordinate Displacements")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        summary["direction_displacement_histogram"] = str(plot_path)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    print("Stage 3 coordinate-direction diagnostics")
    print("========================================")
    print(f"best coordinates : {npz_path}")
    print(f"history          : {history_path}")
    print(f"iterations       : {summary['n_iterations']}")
    print(
        "final mean |dx|/|dy|/|dz|: "
        f"{summary['final_coordinate_displacement']['mean_abs_dx_angstrom']:.6f} / "
        f"{summary['final_coordinate_displacement']['mean_abs_dy_angstrom']:.6f} / "
        f"{summary['final_coordinate_displacement']['mean_abs_dz_angstrom']:.6f} A"
    )
    print(
        "accepted mean |dx|/|dy|/|dz|: "
        f"{summary['accepted_proposals']['mean_abs_dx_angstrom']:.6f} / "
        f"{summary['accepted_proposals']['mean_abs_dy_angstrom']:.6f} / "
        f"{summary['accepted_proposals']['mean_abs_dz_angstrom']:.6f} A"
    )
    print(
        "accepted improving x/y vs z fractions: "
        f"{xy_fraction:.3f} / {z_fraction:.3f}"
    )
    print(f"assessment       : {improving_direction_assessment}")
    print(f"summary          : {summary_path}")
    if save_plot:
        print(f"histogram        : {summary.get('direction_displacement_histogram', '')}")
    return summary


def main() -> None:
    """Run a small default SA skeleton job from the current Stage 2 handoff."""
    run_sa_refinement()


if __name__ == "__main__":
    main()
