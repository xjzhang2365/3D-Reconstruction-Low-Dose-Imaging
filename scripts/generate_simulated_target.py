"""
generate_simulated_target.py
============================
Loader/setup scaffold for generating a Stage 3 simulated target image.

This script intentionally stops after validating inputs and printing raw
reference-image diagnostics.  The abTEM simulation, noise addition, and
preprocessing steps will be added later.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Repo/path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.preprocessing import preprocess_stack
from graphene3d.preprocessing.denoising import BM3DDenoiser


# ---------------------------------------------------------------------------
# Configured inputs and outputs
# ---------------------------------------------------------------------------

PDB_PATH = REPO_ROOT / "data" / "simulated" / "ground_xyz.pdb"
RAW_REF_PATH = REPO_ROOT / "data" / "raw" / "raw_21.tif"

CLEAN_OUT = REPO_ROOT / "data" / "simulated" / "target_clean_abtem.tif"
NOISY_OUT = REPO_ROOT / "data" / "simulated" / "target_noisy_like_raw21.tif"
PREPROC_OUT = REPO_ROOT / "data" / "simulated" / "target_preprocessed_like_raw21.npy"
META_OUT = REPO_ROOT / "data" / "simulated" / "target_metadata.json"

ABTEM_ENERGY_EV = 80000.0
ABTEM_SLICE_THICKNESS_ANGSTROM = 1.0
ABTEM_SAMPLING_ANGSTROM = None  # If None, derive from PDB cell and raw image shape.
NOISE_RANDOM_SEED = 21


def load_raw_reference(path: Path = RAW_REF_PATH) -> np.ndarray:
    """Load the raw frame used as the noise/statistics reference."""
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D raw TIFF image, got shape {image.shape}")
    return image


def print_raw_diagnostics(image: np.ndarray, path: Path = RAW_REF_PATH) -> None:
    """Print basic diagnostics for the raw TIFF reference image."""
    image_float = image.astype(np.float64)
    print("Raw reference image diagnostics")
    print("===============================")
    print(f"path  : {path}")
    print(f"shape : {image.shape}")
    print(f"dtype : {image.dtype}")
    print(f"min   : {image_float.min():.6g}")
    print(f"max   : {image_float.max():.6g}")
    print(f"mean  : {image_float.mean():.6g}")
    print(f"std   : {image_float.std():.6g}")


def load_structure(path: Path = PDB_PATH):
    """Load the PDB atomic structure with ASE for later abTEM handoff."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError(
            "ASE is required to load the PDB structure for simulated-target "
            "generation. Install it with: pip install ase"
        ) from exc

    structure = read(path)
    if len(structure) == 0:
        raise ValueError(f"No atoms were loaded from PDB file: {path}")
    return structure


def print_structure_diagnostics(structure, path: Path = PDB_PATH) -> None:
    """Print basic coordinate diagnostics for an ASE Atoms object."""
    coords = structure.get_positions()
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    print("Atomic structure diagnostics")
    print("============================")
    print(f"path        : {path}")
    print(f"atoms       : {len(structure)}")
    print(f"coord shape : {coords.shape}")
    print(f"x range (A) : [{mins[0]:.6g}, {maxs[0]:.6g}]")
    print(f"y range (A) : [{mins[1]:.6g}, {maxs[1]:.6g}]")
    print(f"z range (A) : [{mins[2]:.6g}, {maxs[2]:.6g}]")


def _abtem_to_numpy(measurement) -> np.ndarray:
    """Extract a NumPy array from common abTEM eager/lazy objects."""
    obj = measurement
    if hasattr(obj, "compute"):
        obj = obj.compute()
    if hasattr(obj, "array"):
        return np.asarray(obj.array, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def _resize_image_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize abTEM output to the raw reference shape if sampling differs."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.shape == shape:
        return arr

    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError(
            "abTEM returned an image shape that differs from the raw reference, "
            "and scipy is not available for resizing."
        ) from exc

    zoom_factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    resized = zoom(arr, zoom_factors, order=1)
    return resized[:shape[0], :shape[1]]


def _abtem_sampling_for_shape(structure, image_shape: tuple[int, int]):
    """Derive approximate abTEM sampling from the structure cell and image size."""
    if ABTEM_SAMPLING_ANGSTROM is not None:
        return float(ABTEM_SAMPLING_ANGSTROM)

    cell_lengths = structure.cell.lengths()
    height, width = image_shape
    sampling_x = float(cell_lengths[0]) / float(width)
    sampling_y = float(cell_lengths[1]) / float(height)
    return (sampling_x, sampling_y)


def simulate_clean_abtem_image(structure,
                               image_shape: tuple[int, int],
                               energy_ev: float = ABTEM_ENERGY_EV,
                               slice_thickness_angstrom: float = ABTEM_SLICE_THICKNESS_ANGSTROM,
                               sampling=None) -> np.ndarray:
    """Generate a clean TEM image from an ASE structure using abTEM."""
    try:
        import abtem
    except ImportError as exc:
        raise ImportError(
            "abTEM is required to generate the clean simulated target image. "
            "Install it with: pip install abtem"
        ) from exc

    if sampling is None:
        sampling = _abtem_sampling_for_shape(structure, image_shape)

    try:
        potential = abtem.Potential(
            structure,
            sampling=sampling,
            slice_thickness=slice_thickness_angstrom,
        )
        wave = abtem.PlaneWave(energy=energy_ev)
        exit_wave = wave.multislice(potential)
        intensity = exit_wave.intensity()
    except AttributeError as exc:
        raise RuntimeError(
            "The installed abTEM API did not expose the expected "
            "Potential / PlaneWave / multislice methods. Update "
            "simulate_clean_abtem_image() for your abTEM version."
        ) from exc

    image = _abtem_to_numpy(intensity)
    if image.ndim > 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"abTEM returned a non-2D image with shape {image.shape}")
    return _resize_image_to_shape(image, image_shape)


def save_clean_image(image: np.ndarray, path: Path = CLEAN_OUT) -> np.ndarray:
    """Save the clean simulated image as a float32 TIFF and return saved data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = np.asarray(image, dtype=np.float32)
    tifffile.imwrite(path, saved)
    return saved


def scale_clean_to_raw_range(clean_image: np.ndarray,
                             raw_reference: np.ndarray) -> np.ndarray:
    """Map clean simulation intensities into raw-count-like units."""
    clean = np.asarray(clean_image, dtype=np.float64)
    raw = np.asarray(raw_reference, dtype=np.float64)

    clean_min = float(clean.min())
    clean_max = float(clean.max())
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    if clean_max == clean_min:
        return np.full_like(clean, raw.mean(), dtype=np.float64)

    clean_unit = (clean - clean_min) / (clean_max - clean_min)
    scaled = raw_min + clean_unit * (raw_max - raw_min)
    return scaled + (float(raw.mean()) - float(scaled.mean()))


def add_raw_calibrated_noise(clean_counts: np.ndarray,
                             raw_reference: np.ndarray,
                             seed: int = NOISE_RANDOM_SEED) -> np.ndarray:
    """
    Add Gaussian noise so the simulated image roughly matches raw std.

    The clean image is already scaled into raw-count-like units.  The noise
    standard deviation is chosen so clean variance plus noise variance matches
    the raw reference variance, when possible.
    """
    clean = np.asarray(clean_counts, dtype=np.float64)
    raw = np.asarray(raw_reference, dtype=np.float64)
    raw_std = float(raw.std())
    clean_std = float(clean.std())
    noise_std = float(np.sqrt(max(raw_std ** 2 - clean_std ** 2, 0.0)))

    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(loc=0.0, scale=noise_std, size=clean.shape)
    return noisy


def generate_noisy_like_raw(clean_image: np.ndarray,
                            raw_reference: np.ndarray,
                            seed: int = NOISE_RANDOM_SEED) -> np.ndarray:
    """Scale the clean simulation and add raw-calibrated Gaussian noise."""
    clean_counts = scale_clean_to_raw_range(clean_image, raw_reference)
    return add_raw_calibrated_noise(clean_counts, raw_reference, seed=seed)


def save_noisy_image(image: np.ndarray, path: Path = NOISY_OUT) -> np.ndarray:
    """Save the noisy simulated image as a float32 TIFF and return saved data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = np.asarray(image, dtype=np.float32)
    tifffile.imwrite(path, saved)
    return saved


def preprocess_noisy_target(noisy_image: np.ndarray) -> np.ndarray:
    """Run the existing Stage 1 preprocessing pipeline on one noisy image."""
    stack = np.asarray(noisy_image, dtype=np.float64)[None, :, :]
    result = preprocess_stack(
        stack,
        target_idx=0,
        flat_field_sigma=20.0,
        dead_pixel_sigma=5.0,
        denoiser=BM3DDenoiser(),
        window_size=1,
    )
    return np.asarray(result["averaged"], dtype=np.float64)


def save_preprocessed_image(image: np.ndarray, path: Path = PREPROC_OUT) -> np.ndarray:
    """Save the preprocessed simulated target as a NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = np.asarray(image, dtype=np.float64)
    np.save(path, saved)
    return saved


def print_image_diagnostics(image: np.ndarray, title: str, path: Path) -> None:
    """Print basic diagnostics for a generated image."""
    image_float = np.asarray(image, dtype=np.float64)
    print(title)
    print("=" * len(title))
    print(f"path  : {path}")
    print(f"shape : {image.shape}")
    print(f"dtype : {image.dtype}")
    print(f"min   : {image_float.min():.6g}")
    print(f"max   : {image_float.max():.6g}")
    print(f"mean  : {image_float.mean():.6g}")
    print(f"std   : {image_float.std():.6g}")


def _repo_relative(path: Path) -> str:
    """Return a stable repo-relative path string for metadata."""
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _image_stats(image: np.ndarray) -> dict:
    """Return JSON-serializable image statistics."""
    arr = np.asarray(image)
    arr_float = arr.astype(np.float64)
    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "min": float(arr_float.min()),
        "max": float(arr_float.max()),
        "mean": float(arr_float.mean()),
        "std": float(arr_float.std()),
    }


def write_target_metadata(raw_reference: np.ndarray,
                          clean_image: np.ndarray,
                          noisy_image: np.ndarray,
                          preprocessed_image: np.ndarray,
                          path: Path = META_OUT) -> dict:
    """Write target-generation metadata and recommended Stage 3 target."""
    metadata = {
        "recommended_stage3_target": _repo_relative(PREPROC_OUT),
        "inputs": {
            "pdb_path": _repo_relative(PDB_PATH),
            "raw_reference_image": {
                "path": _repo_relative(RAW_REF_PATH),
                "stats": _image_stats(raw_reference),
            },
        },
        "outputs": {
            "clean_simulated_image": {
                "path": _repo_relative(CLEAN_OUT),
                "stats": _image_stats(clean_image),
            },
            "noisy_simulated_image": {
                "path": _repo_relative(NOISY_OUT),
                "stats": _image_stats(noisy_image),
            },
            "preprocessed_simulated_image": {
                "path": _repo_relative(PREPROC_OUT),
                "stats": _image_stats(preprocessed_image),
            },
        },
        "simulation": {
            "abtem_energy_ev": ABTEM_ENERGY_EV,
            "abtem_slice_thickness_angstrom": ABTEM_SLICE_THICKNESS_ANGSTROM,
            "abtem_sampling_angstrom": ABTEM_SAMPLING_ANGSTROM,
        },
        "noise": {
            "model": "raw-range scaling plus zero-mean Gaussian noise",
            "random_seed": NOISE_RANDOM_SEED,
        },
        "preprocessing": {
            "stage1_preprocess_stack": True,
            "window_size": 1,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    """Validate configured paths and print raw reference diagnostics."""
    if not PDB_PATH.exists():
        raise FileNotFoundError(f"PDB file not found: {PDB_PATH}")
    if not RAW_REF_PATH.exists():
        raise FileNotFoundError(f"Raw TIFF reference not found: {RAW_REF_PATH}")

    CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)

    raw_ref = load_raw_reference(RAW_REF_PATH)
    print_raw_diagnostics(raw_ref, RAW_REF_PATH)
    print("")
    try:
        structure = load_structure(PDB_PATH)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print_structure_diagnostics(structure, PDB_PATH)
    print("")
    try:
        clean_image = simulate_clean_abtem_image(structure, raw_ref.shape)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    saved_clean_image = save_clean_image(clean_image, CLEAN_OUT)
    print_image_diagnostics(saved_clean_image, "Clean abTEM image diagnostics", CLEAN_OUT)
    print("")
    noisy_image = generate_noisy_like_raw(saved_clean_image, raw_ref)
    saved_noisy_image = save_noisy_image(noisy_image, NOISY_OUT)
    print_image_diagnostics(saved_noisy_image, "Noisy simulated image diagnostics", NOISY_OUT)
    print("")
    try:
        preprocessed_image = preprocess_noisy_target(saved_noisy_image)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    saved_preprocessed_image = save_preprocessed_image(preprocessed_image, PREPROC_OUT)
    print_image_diagnostics(
        saved_preprocessed_image,
        "Preprocessed simulated image diagnostics",
        PREPROC_OUT,
    )
    print("")
    metadata = write_target_metadata(
        raw_ref,
        saved_clean_image,
        saved_noisy_image,
        saved_preprocessed_image,
        META_OUT,
    )
    print("")
    print("Configured Stage 3 simulated target paths")
    print("=========================================")
    print(f"PDB_PATH    : {PDB_PATH}")
    print(f"CLEAN_OUT   : {CLEAN_OUT}")
    print(f"NOISY_OUT   : {NOISY_OUT}")
    print(f"PREPROC_OUT : {PREPROC_OUT}")
    print(f"META_OUT    : {META_OUT}")
    print("")
    print("Target metadata summary")
    print("=======================")
    print(f"metadata path             : {META_OUT}")
    print(f"recommended Stage 3 target: {metadata['recommended_stage3_target']}")
    print("")
    print("Clean, noisy, and preprocessed simulated target images generated.")


if __name__ == "__main__":
    main()
