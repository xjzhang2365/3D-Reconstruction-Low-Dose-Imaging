"""
stage2_pipeline.py
==================
Final Stage 2 initialization pipeline.

This module is the frozen Stage 2 handoff path before Stage 3 SA + MD
refinement.  It integrates the validated components without changing their
underlying algorithms:

  1. detect bright graphene ring-centre "holes"
  2. initialize atom x-y positions from hole triplets
  3. optionally complete unresolved interior gaps with local dark-atom checks
  4. optionally add local MAP detections in defect or fallback regions
  5. estimate raw z with the simplified PCD model
  6. smooth z with LOWESS, using frac=0.08 by default

Contrast convention:
  - graphene hole centres are bright and are detected directly
  - atom columns are dark in the original TEM contrast
  - MAP keeps its positive-Gaussian model, so dark-atom patches are inverted at
    the fitting boundary only

Stage 3 should consume the smoothed coordinates in xyz_angstrom unless it
intentionally wants to inspect raw PCD z values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
from scipy.spatial import KDTree

from graphene3d.stage2.find_xy import build_atom_positions_from_holes
from graphene3d.stage2.hole_finding import find_holes
from graphene3d.stage2.map_detection import detect_defect_atoms, detect_dark_atoms_in_local_patch
from graphene3d.stage2.pcd_z import estimate_z_pcd, lowess_smooth_z


STAGE2_VALIDATION_BASELINE: dict[str, float | int | str] = {
    "status": "local_completion_enabled",
    "image": "data/synthetic/img_noisy.tif",
    "ground_truth": "data/synthetic/ground_xyz.pdb",
    "final_atoms": 636,
    "matched_atoms": 626,
    "missing_atoms": 14,
    "extra_atoms": 10,
    "xy_rmse_px": 0.777538,
    "z_smooth_rmse_centered_angstrom": 1.420978,
}


@dataclass
class Stage2Config:
    """Parameters for the frozen Stage 2 initializer.

    Defaults are the current validated settings for the simulated graphene
    case.  Changing these values is allowed for experiments, but the Stage 3
    handoff baseline above assumes the defaults here.
    """

    # Hole detection: bright graphene centre regions
    hole_sigma: float = 2.0
    hole_connectivity: int = 2
    hole_border_margin: int = 2
    hole_min_response: Optional[float] = 0.6
    hole_min_distance: Optional[float] = 5.0

    # Hole-triplet atom x-y initialization
    nn_radius_pixels: float = 16.0
    min_triplet_angle_deg: float = 40.0
    max_triplet_side_ratio: float = 1.35
    atom_duplicate_radius: float = 1.5

    # Optional MAP in defect regions
    use_map_defects: bool = True
    map_patch_size: int = 20
    map_overlap: int = 5
    map_rho: float = 1.5
    map_n_max: int = 4
    map_duplicate_radius: float = 3.0

    # Selective MAP fallback for sparse interior regions
    use_selective_map_fallback: bool = True
    fallback_nn_radius_tolerance: float = 1.20
    fallback_max_triplet_side_ratio: float = 1.60
    fallback_existing_atom_radius: float = 3.0
    fallback_min_hole_support: int = 3
    fallback_hole_support_radius: float = 16.0
    fallback_boundary_margin: float = 8.0
    fallback_patch_radius: int = 8
    fallback_accept_radius: float = 3.5
    fallback_map_rho: float = 1.5
    fallback_map_n_max: int = 1
    fallback_min_existing_nn_ratio: float = 0.65
    fallback_max_existing_nn_ratio: float = 1.65
    fallback_min_dark_zscore: float = 0.5
    fallback_min_acceptance_score: float = 3.0

    # Local atom completion for interior hole-supported gaps
    enable_local_completion: bool = True
    completion_relaxed_nn_radius_tolerance: float = 1.45
    completion_relaxed_max_triplet_side_ratio: float = 2.00
    completion_min_triplet_angle_deg: float = 25.0
    completion_existing_atom_radius: float = 4.0
    completion_min_existing_nn_ratio: float = 0.68
    completion_max_existing_nn_ratio: float = 1.65
    completion_min_hole_support: int = 3
    completion_hole_support_radius: float = 16.0
    completion_boundary_margin: float = 8.0
    completion_patch_radius: int = 8
    completion_refine_radius: int = 2
    completion_accept_radius: float = 3.5
    completion_min_dark_zscore: float = 0.6
    completion_gaussian_dark_zscore: float = 1.25
    completion_min_acceptance_score: float = 3.5
    completion_min_nearest_hole_distance: float = 6.0
    completion_max_nearest3_hole_cv: float = 0.35
    completion_use_dark_min_candidates: bool = False
    completion_dark_min_radius: int = 2
    completion_dark_min_zscore: float = 1.0
    completion_max_dark_min_candidates: int = 80
    completion_map_if_weak: bool = True
    completion_map_rho: float = 1.5
    completion_map_n_max: int = 1

    # PCD z and LOWESS smoothing
    pcd_slope: float = 0.0049
    z_scale: float = 9.3
    pcd_search_radius: float = 13.0
    lowess_frac: float = 0.08
    lowess_iter: int = 3

    # Coordinate calibration
    pixel_size_angstrom: float = 0.183

    # Gaussian refinement (post-detection, pre-z-init)
    use_gaussian_refinement: bool = True
    gaussian_patch_radius_px: int = 4
    gaussian_min_snr: float = 0.2
    # refine_position=False: keep original position, only reject ghost atoms.
    # Set True for high-SNR images where sub-pixel refinement is reliable.
    gaussian_refine_position: bool = False
    # Lattice-completion passes: 0 disables completion (position-only mode).
    # Enable with n>=1 only after tuning min_snr for the specific image dose.
    gaussian_n_completion_passes: int = 0

    # Outputs
    output_prefix: str = "stage2_initialization"
    save_outputs: bool = True
    verbose: bool = True


@dataclass
class Stage2Result:
    """Container returned by run_stage2_initialization().

    Stage 3 should usually use xyz_angstrom, where x/y are converted from
    image pixels and z is the LOWESS-smoothed PCD estimate.  z_raw is retained
    for diagnostics and ablation studies.
    """

    holes_xy: np.ndarray
    atom_xy_pixels: np.ndarray
    z_raw: np.ndarray
    z_smooth: np.ndarray
    xyz_pixels: np.ndarray
    xyz_angstrom: np.ndarray
    source_labels: np.ndarray
    fallback_diagnostics: list[dict[str, Any]]
    completion_diagnostics: list[dict[str, Any]]
    metrics: dict[str, Any]
    output_paths: dict[str, str]


def _source_initializer_weights(source_labels: np.ndarray) -> np.ndarray:
    """
    Heuristic Stage 2 provenance weights for downstream SA bookkeeping.

    These are not probabilities and must not be interpreted as reconstruction
    likelihoods.  They only give Stage 3 a compact way to distinguish atoms
    coming from the strongest geometry path from atoms introduced by local
    completion or MAP fallback.
    """
    weights = {
        "triplet": 1.00,
        "completion_gaussian": 0.85,
        "completion_map": 0.80,
        "fallback_map": 0.75,
        "defect_map": 0.70,
    }
    labels = np.asarray(source_labels).astype(str)
    return np.array([weights.get(label, 0.50) for label in labels], dtype=np.float64)


def _sa_export_metadata(result: Stage2Result,
                        extra_metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build metadata documenting what Stage 3 should assume about Stage 2."""
    metadata: dict[str, Any] = {
        "producer": "stage2_pipeline.export_stage2_for_sa",
        "purpose": "initial structure for Stage 3 SA refinement",
        "atom_count": int(len(result.xyz_angstrom)),
        "element": "C",
        "coordinate_frame": "angstrom",
        "coordinate_fields": ["x_angstrom", "y_angstrom", "z_angstrom"],
        "z_definition": "LOWESS-smoothed raw PCD z estimate",
        "xy_definition": "image x/y converted to Angstrom by Stage2Config.pixel_size_angstrom",
        "recommended_stage3_input": "xyz_angstrom",
        "stage3_assumptions": [
            "Use x_angstrom, y_angstrom, z_angstrom as the initial SA coordinates.",
            "Treat z_raw_angstrom as a diagnostic only; z_smooth_angstrom is the initializer z.",
            "Use source_label and initializer_weight for provenance-aware diagnostics, not as hard physical constraints.",
            "Stage 2 may contain missing and extra atoms; SA/MD is expected to refine geometry rather than assume this is final truth.",
        ],
        "source_label_meaning": {
            "triplet": "atom from bright-hole triplet geometry",
            "completion_gaussian": "local completion atom accepted by dark-minimum Gaussian proxy",
            "completion_map": "local completion atom accepted by local inverted MAP",
            "fallback_map": "selective fallback atom accepted by local inverted MAP",
            "defect_map": "atom from explicit defect-region inverted MAP",
        },
        "initializer_weight_note": "Heuristic provenance weight only; not a probability.",
        "validation_baseline": STAGE2_VALIDATION_BASELINE,
        "metrics": result.metrics,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def export_stage2_for_sa(result: Stage2Result,
                         output_prefix: str | Path,
                         extra_metadata: Optional[dict[str, Any]] = None) -> dict[str, str]:
    """
    Export a completed Stage 2 result in a clean SA initializer format.

    Files written:
      - <prefix>_sa_input.csv: inspectable atom table
      - <prefix>_sa_input.npz: machine-readable arrays
      - <prefix>_sa_metadata.json: coordinate-frame and provenance contract

    The SA coordinate columns are x_angstrom, y_angstrom, z_angstrom from
    result.xyz_angstrom.  The extra columns preserve Stage 2 provenance and
    z diagnostics without changing the Stage 2 numerical pipeline.
    """
    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.parent / f"{prefix.name}_sa_input.csv"
    npz_path = prefix.parent / f"{prefix.name}_sa_input.npz"
    metadata_path = prefix.parent / f"{prefix.name}_sa_metadata.json"

    atom_ids = np.arange(len(result.xyz_angstrom), dtype=int)
    element_symbols = np.array(["C"] * len(atom_ids), dtype=object)
    source_labels = result.source_labels.astype(str)
    initializer_weights = _source_initializer_weights(source_labels)
    z_delta = result.z_smooth - result.z_raw

    header = (
        "atom_id,element,x_angstrom,y_angstrom,z_angstrom,source_label,"
        "initializer_weight,x_pixel,y_pixel,z_raw_angstrom,z_smooth_angstrom,"
        "z_delta_smooth_minus_raw_angstrom"
    )
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for idx in atom_ids:
            fh.write(
                f"{idx},C,"
                f"{result.xyz_angstrom[idx, 0]:.8f},"
                f"{result.xyz_angstrom[idx, 1]:.8f},"
                f"{result.xyz_angstrom[idx, 2]:.8f},"
                f"{source_labels[idx]},"
                f"{initializer_weights[idx]:.3f},"
                f"{result.atom_xy_pixels[idx, 0]:.8f},"
                f"{result.atom_xy_pixels[idx, 1]:.8f},"
                f"{result.z_raw[idx]:.8f},"
                f"{result.z_smooth[idx]:.8f},"
                f"{z_delta[idx]:.8f}\n"
            )

    metadata = _sa_export_metadata(result, extra_metadata)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    np.savez_compressed(
        npz_path,
        atom_id=atom_ids,
        element=element_symbols.astype(str),
        xyz_angstrom=result.xyz_angstrom,
        xyz_pixels=result.xyz_pixels,
        atom_xy_pixels=result.atom_xy_pixels,
        z_raw_angstrom=result.z_raw,
        z_smooth_angstrom=result.z_smooth,
        z_delta_smooth_minus_raw_angstrom=z_delta,
        source_label=source_labels,
        initializer_weight=initializer_weights,
        metadata_json=json.dumps(metadata),
    )

    return {
        "sa_input_csv": str(csv_path),
        "sa_input_npz": str(npz_path),
        "sa_metadata_json": str(metadata_path),
    }


def save_stage3_handoff(result: Stage2Result, path: str | Path) -> str:
    """
    Save a compact Stage 3 input file from a completed Stage 2 result.

    For SA refinement handoff, prefer export_stage2_for_sa(), which writes a
    CSV, NPZ, and explicit metadata contract.  This helper is retained as a
    compact array-only handoff file.

    The handoff file is an NPZ with:
      - xyz_angstrom: smoothed Stage 2 coordinates for SA + MD initialization
      - xyz_pixels: image-coordinate version of the same atoms
      - z_raw_angstrom: raw PCD z before LOWESS smoothing
      - z_smooth_angstrom: LOWESS-smoothed z used in xyz_angstrom
      - source_labels: atom provenance, e.g. triplet or fallback_map
      - metrics_json: run-level diagnostics as JSON
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        xyz_angstrom=result.xyz_angstrom,
        xyz_pixels=result.xyz_pixels,
        atom_xy_pixels=result.atom_xy_pixels,
        z_raw_angstrom=result.z_raw,
        z_smooth_angstrom=result.z_smooth,
        source_labels=result.source_labels.astype(str),
        metrics_json=json.dumps(result.metrics),
        coordinate_note=(
            "xyz_angstrom uses image x/y converted by pixel_size_angstrom; "
            "z is LOWESS-smoothed PCD height in Angstrom."
        ),
    )
    return str(out_path)


def _log(config: Stage2Config, message: str) -> None:
    if config.verbose:
        print(message)


def _as_image_array(image: np.ndarray | str | Path) -> np.ndarray:
    """Accept either a 2-D image array or a TIFF-like image path."""
    if isinstance(image, (str, Path)):
        import tifffile
        return tifffile.imread(str(image)).astype(np.float64)
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Stage 2 expects a 2-D preprocessed image, got {arr.shape}")
    return arr


def _build_xyz_angstrom(atom_xy: np.ndarray,
                        z_smooth: np.ndarray,
                        image_shape: tuple[int, int],
                        pixel_size_angstrom: float) -> np.ndarray:
    """Convert image pixel x/y plus z Angstrom into physical coordinates."""
    image_height_ang = image_shape[0] * pixel_size_angstrom
    x_ang = atom_xy[:, 0] * pixel_size_angstrom
    y_ang = image_height_ang - atom_xy[:, 1] * pixel_size_angstrom
    return np.column_stack([x_ang, y_ang, z_smooth])


def _append_unique_positions(existing_xy: np.ndarray,
                             existing_labels: np.ndarray,
                             new_xy: np.ndarray,
                             new_label: str,
                             radius: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Append new positions that are not within radius of existing atoms."""
    if len(new_xy) == 0:
        return existing_xy, existing_labels, 0

    accepted = []
    current = existing_xy.copy()
    for pos in np.asarray(new_xy, dtype=np.float64).reshape(-1, 2):
        if len(current) == 0:
            accepted.append(pos)
            current = np.array([pos], dtype=np.float64)
            continue
        d_min = float(np.min(np.linalg.norm(current - pos, axis=1)))
        if d_min > radius:
            accepted.append(pos)
            current = np.vstack([current, pos])

    if not accepted:
        return existing_xy, existing_labels, 0

    accepted_arr = np.array(accepted, dtype=np.float64)
    labels = np.array([new_label] * len(accepted_arr), dtype=object)
    return np.vstack([existing_xy, accepted_arr]), np.concatenate([existing_labels, labels]), len(accepted_arr)


def _append_unique_labeled_positions(existing_xy: np.ndarray,
                                     existing_labels: np.ndarray,
                                     new_xy: np.ndarray,
                                     new_labels: np.ndarray,
                                     radius: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Append new positions with one source label per candidate."""
    if len(new_xy) == 0:
        return existing_xy, existing_labels, 0

    accepted_xy = []
    accepted_labels = []
    current = existing_xy.copy()
    labels = np.asarray(new_labels, dtype=object).reshape(-1)
    for idx, pos in enumerate(np.asarray(new_xy, dtype=np.float64).reshape(-1, 2)):
        if len(current) == 0:
            accepted_xy.append(pos)
            accepted_labels.append(labels[idx])
            current = np.array([pos], dtype=np.float64)
            continue
        d_min = float(np.min(np.linalg.norm(current - pos, axis=1)))
        if d_min > radius:
            accepted_xy.append(pos)
            accepted_labels.append(labels[idx])
            current = np.vstack([current, pos])

    if not accepted_xy:
        return existing_xy, existing_labels, 0

    accepted_arr = np.array(accepted_xy, dtype=np.float64)
    accepted_label_arr = np.array(accepted_labels, dtype=object)
    return (
        np.vstack([existing_xy, accepted_arr]),
        np.concatenate([existing_labels, accepted_label_arr]),
        len(accepted_arr),
    )


def _nearest_neighbor_median(points: np.ndarray) -> float:
    """Return the median nearest-neighbour distance for current atom positions."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.nan
    distances, _ = KDTree(pts).query(pts, k=2)
    return float(np.median(distances[:, 1]))


def _min_distance_to_points(points: np.ndarray, pos: np.ndarray) -> float:
    """Minimum Euclidean distance from pos to a point cloud."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.inf
    return float(np.min(np.linalg.norm(pts - pos, axis=1)))


def _hole_support_count(holes: np.ndarray, pos: np.ndarray, radius: float) -> int:
    """Count bright hole centres supporting a local atom candidate."""
    if len(holes) == 0:
        return 0
    return int(np.sum(np.linalg.norm(holes - pos, axis=1) <= radius))


def _nearest_hole_geometry(holes: np.ndarray,
                           pos: np.ndarray) -> tuple[float, float, float]:
    """
    Summarize whether a candidate sits between nearby bright holes.

    The first value is the nearest-hole distance.  The second and third values
    describe the three closest holes: their mean distance and coefficient of
    variation.  Good atom candidates should not sit on top of a bright hole and
    should have reasonably balanced support from the nearest hole triplet.
    """
    if len(holes) < 3:
        return np.inf, np.nan, np.nan
    dists = np.sort(np.linalg.norm(holes - pos, axis=1))[:3]
    mean3 = float(np.mean(dists))
    if mean3 <= 1e-12:
        return float(dists[0]), mean3, np.inf
    return float(dists[0]), mean3, float(np.std(dists) / mean3)


def _local_dark_zscore(image: np.ndarray, pos: np.ndarray, radius: int) -> float:
    """
    Score whether a candidate is locally dark in the original TEM contrast.

    Positive values mean the candidate pixel is darker than its local median.
    This keeps the Stage 2 convention explicit: holes are bright, atoms are dark.
    """
    h, w = image.shape
    x, y = np.asarray(pos, dtype=np.float64)
    c = int(round(x))
    r = int(round(y))
    if c < 0 or c >= w or r < 0 or r >= h:
        return np.nan

    c0 = max(0, c - radius)
    c1 = min(w, c + radius + 1)
    r0 = max(0, r - radius)
    r1 = min(h, r + radius + 1)
    patch = image[r0:r1, c0:c1]
    if patch.size == 0:
        return np.nan

    local_median = float(np.median(patch))
    mad = float(np.median(np.abs(patch - local_median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-12:
        robust_sigma = float(np.std(patch))
    if robust_sigma <= 1e-12:
        return 0.0
    return float((local_median - image[r, c]) / robust_sigma)


def _inside_margin(pos: np.ndarray,
                   image_shape: tuple[int, int],
                   margin: float) -> bool:
    """Return True if an x-y position is far enough from image boundaries."""
    h, w = image_shape
    x, y = np.asarray(pos, dtype=np.float64)
    return bool(margin <= x <= w - 1 - margin and margin <= y <= h - 1 - margin)


def _merge_scored_candidates(candidates: list[dict[str, Any]],
                             radius: float) -> list[dict[str, Any]]:
    """Merge duplicate candidate positions, keeping the highest scored entry."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: row.get("proposal_score", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    kept_xy: list[np.ndarray] = []
    for row in ordered:
        pos = np.array([row["candidate_x"], row["candidate_y"]], dtype=np.float64)
        if all(np.linalg.norm(pos - other) > radius for other in kept_xy):
            row["candidate_index"] = len(kept)
            kept.append(row)
            kept_xy.append(pos)
    return kept


def _completion_geometry_candidates(holes: np.ndarray,
                                    atom_xy: np.ndarray,
                                    image_shape: tuple[int, int],
                                    nn_median: float,
                                    config: Stage2Config) -> list[dict[str, Any]]:
    """
    Propose local completion candidates by relaxing hole-triplet geometry only
    for gap discovery.  The global find_xy triplet logic is unchanged.
    """
    relaxed = build_atom_positions_from_holes(
        holes,
        nn_radius=config.nn_radius_pixels,
        min_triplet_angle_deg=config.completion_min_triplet_angle_deg,
        max_side_ratio=config.completion_relaxed_max_triplet_side_ratio,
        duplicate_radius=config.atom_duplicate_radius,
        nn_radius_tolerance=config.completion_relaxed_nn_radius_tolerance,
    )
    candidates: list[dict[str, Any]] = []
    if len(relaxed) == 0:
        return candidates

    min_existing = config.completion_existing_atom_radius
    max_existing = np.inf
    if np.isfinite(nn_median) and nn_median > 0:
        min_existing = max(min_existing, config.completion_min_existing_nn_ratio * nn_median)
        max_existing = config.completion_max_existing_nn_ratio * nn_median

    for pos in relaxed:
        nearest_existing = _min_distance_to_points(atom_xy, pos)
        support = _hole_support_count(holes, pos, config.completion_hole_support_radius)
        if nearest_existing <= min_existing:
            continue
        if nearest_existing > max_existing:
            continue
        if support < config.completion_min_hole_support:
            continue
        if not _inside_margin(pos, image_shape, config.completion_boundary_margin):
            continue
        candidates.append({
            "candidate_x": float(pos[0]),
            "candidate_y": float(pos[1]),
            "proposal_source": "hole_geometry",
            "proposal_score": float(support),
        })
    return candidates


def _completion_dark_minimum_candidates(image: np.ndarray,
                                        holes: np.ndarray,
                                        atom_xy: np.ndarray,
                                        nn_median: float,
                                        config: Stage2Config) -> list[dict[str, Any]]:
    """Propose interior gap candidates from local dark-intensity minima."""
    from scipy.ndimage import minimum_filter

    size = 2 * config.completion_dark_min_radius + 1
    local_min = image <= minimum_filter(image, size=size, mode="nearest")
    ys, xs = np.nonzero(local_min)

    min_existing = config.completion_existing_atom_radius
    max_existing = np.inf
    if np.isfinite(nn_median) and nn_median > 0:
        min_existing = max(min_existing, config.completion_min_existing_nn_ratio * nn_median)
        max_existing = config.completion_max_existing_nn_ratio * nn_median

    candidates: list[dict[str, Any]] = []
    for x, y in zip(xs, ys):
        pos = np.array([float(x), float(y)], dtype=np.float64)
        if not _inside_margin(pos, image.shape, config.completion_boundary_margin):
            continue
        nearest_existing = _min_distance_to_points(atom_xy, pos)
        if nearest_existing <= min_existing or nearest_existing > max_existing:
            continue
        support = _hole_support_count(holes, pos, config.completion_hole_support_radius)
        if support < config.completion_min_hole_support:
            continue
        dark_zscore = _local_dark_zscore(image, pos, config.completion_patch_radius)
        if dark_zscore < config.completion_dark_min_zscore:
            continue
        candidates.append({
            "candidate_x": float(pos[0]),
            "candidate_y": float(pos[1]),
            "proposal_source": "dark_minimum",
            "proposal_score": float(dark_zscore + 0.2 * support),
        })

    return sorted(
        candidates,
        key=lambda row: row["proposal_score"],
        reverse=True,
    )[:config.completion_max_dark_min_candidates]


def _completion_candidates(image: np.ndarray,
                           holes: np.ndarray,
                           atom_xy: np.ndarray,
                           config: Stage2Config) -> list[dict[str, Any]]:
    """Generate local atom-completion candidates from geometry and image gaps."""
    nn_median = _nearest_neighbor_median(atom_xy)
    candidates = _completion_geometry_candidates(
        holes,
        atom_xy,
        image.shape,
        nn_median,
        config,
    )
    if config.completion_use_dark_min_candidates and config.completion_max_dark_min_candidates > 0:
        candidates.extend(_completion_dark_minimum_candidates(
            image,
            holes,
            atom_xy,
            nn_median,
            config,
        ))
    return _merge_scored_candidates(candidates, config.atom_duplicate_radius)


def _refine_to_local_dark_minimum(image: np.ndarray,
                                  candidate_xy: np.ndarray,
                                  radius: int) -> tuple[np.ndarray, float]:
    """Move a candidate to the darkest local pixel as a simple Gaussian proxy."""
    h, w = image.shape
    x, y = np.asarray(candidate_xy, dtype=np.float64)
    c = int(round(x))
    r = int(round(y))
    c0 = max(0, c - radius)
    c1 = min(w, c + radius + 1)
    r0 = max(0, r - radius)
    r1 = min(h, r + radius + 1)
    patch = image[r0:r1, c0:c1]
    if patch.size == 0:
        return np.asarray(candidate_xy, dtype=np.float64), np.inf
    local_r, local_c = np.unravel_index(int(np.argmin(patch)), patch.shape)
    refined = np.array([c0 + local_c, r0 + local_r], dtype=np.float64)
    return refined, float(np.linalg.norm(refined - np.asarray(candidate_xy, dtype=np.float64)))


def _score_completion_detection(candidate_row: dict[str, Any],
                                detection_xy: np.ndarray,
                                method: str,
                                existing_xy: np.ndarray,
                                holes: np.ndarray,
                                image: np.ndarray,
                                nn_median: float,
                                config: Stage2Config,
                                map_n_best: int = 0,
                                map_log_prob: float = np.nan) -> dict[str, Any]:
    """Score a local completion candidate before accepting it into Stage 2."""
    candidate = np.array([candidate_row["candidate_x"], candidate_row["candidate_y"]], dtype=np.float64)
    detection = np.asarray(detection_xy, dtype=np.float64)
    candidate_shift = float(np.linalg.norm(detection - candidate))
    nearest_existing = _min_distance_to_points(existing_xy, detection)
    hole_support = _hole_support_count(holes, detection, config.completion_hole_support_radius)
    nearest_hole_dist, nearest3_hole_mean, nearest3_hole_cv = _nearest_hole_geometry(holes, detection)
    dark_zscore = _local_dark_zscore(image, detection, config.completion_patch_radius)

    min_existing = config.completion_existing_atom_radius
    max_existing = np.inf
    if np.isfinite(nn_median) and nn_median > 0:
        existing_nn_ratio = nearest_existing / nn_median
        min_existing = max(min_existing, config.completion_min_existing_nn_ratio * nn_median)
        max_existing = config.completion_max_existing_nn_ratio * nn_median
    else:
        existing_nn_ratio = np.nan

    shift_score = max(0.0, 1.0 - candidate_shift / max(config.completion_accept_radius, 1e-12))
    support_score = min(1.0, hole_support / max(float(config.completion_min_hole_support), 1.0))
    dark_score = min(1.0, max(0.0, dark_zscore) / max(config.completion_gaussian_dark_zscore, 1e-12))
    distance_score = 1.0 if nearest_existing > min_existing else 0.0
    spacing_score = 1.0 if nearest_existing <= max_existing else 0.0
    hole_geometry_score = 1.0 if (
        nearest_hole_dist >= config.completion_min_nearest_hole_distance
        and nearest3_hole_cv <= config.completion_max_nearest3_hole_cv
    ) else 0.0
    method_score = 1.0 if method == "completion_gaussian" else 0.8
    score = (
        shift_score
        + support_score
        + dark_score
        + distance_score
        + spacing_score
        + hole_geometry_score
        + method_score
    )

    reject_reasons: list[str] = []
    if candidate_shift > config.completion_accept_radius:
        reject_reasons.append("too_far_from_gap_candidate")
    if hole_support < config.completion_min_hole_support:
        reject_reasons.append("low_hole_support")
    if nearest_existing <= min_existing:
        reject_reasons.append("too_close_to_existing_atom")
    if nearest_existing > max_existing:
        reject_reasons.append("too_isolated_from_existing_atoms")
    if nearest_hole_dist < config.completion_min_nearest_hole_distance:
        reject_reasons.append("too_close_to_bright_hole")
    if nearest3_hole_cv > config.completion_max_nearest3_hole_cv:
        reject_reasons.append("unbalanced_hole_triplet_support")
    if not np.isfinite(dark_zscore) or dark_zscore < config.completion_min_dark_zscore:
        reject_reasons.append("weak_dark_atom_contrast")
    if score < config.completion_min_acceptance_score:
        reject_reasons.append("low_acceptance_score")

    return {
        "candidate_index": int(candidate_row.get("candidate_index", -1)),
        "proposal_source": str(candidate_row.get("proposal_source", "")),
        "completion_method": method,
        "candidate_x": float(candidate[0]),
        "candidate_y": float(candidate[1]),
        "detection_x": float(detection[0]),
        "detection_y": float(detection[1]),
        "map_n_best": int(map_n_best),
        "map_log_prob": float(map_log_prob),
        "candidate_shift": candidate_shift,
        "nearest_existing_atom_dist": float(nearest_existing),
        "existing_nn_ratio": float(existing_nn_ratio),
        "hole_support": int(hole_support),
        "nearest_hole_dist": float(nearest_hole_dist),
        "nearest3_hole_mean": float(nearest3_hole_mean),
        "nearest3_hole_cv": float(nearest3_hole_cv),
        "dark_zscore": float(dark_zscore),
        "shift_score": float(shift_score),
        "support_score": float(support_score),
        "dark_score": float(dark_score),
        "distance_score": float(distance_score),
        "spacing_score": float(spacing_score),
        "hole_geometry_score": float(hole_geometry_score),
        "method_score": float(method_score),
        "acceptance_score": float(score),
        "accepted": len(reject_reasons) == 0,
        "reject_reason": ";".join(reject_reasons),
    }


def _run_local_atom_completion(image: np.ndarray,
                               holes: np.ndarray,
                               atom_xy: np.ndarray,
                               config: Stage2Config
                               ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Recover interior missing atoms with local checks before z estimation.

    A candidate first gets a dark-minimum Gaussian proxy check.  If that is
    weak and MAP fallback is enabled, the same local patch is passed to the
    inverted dark-atom MAP helper.
    """
    candidates = _completion_candidates(image, holes, atom_xy, config)
    accepted: list[np.ndarray] = []
    labels: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    current_xy = np.asarray(atom_xy, dtype=np.float64).reshape(-1, 2).copy()

    for candidate_row in candidates:
        candidate_xy = np.array([candidate_row["candidate_x"], candidate_row["candidate_y"]], dtype=np.float64)
        nn_median = _nearest_neighbor_median(current_xy)
        refined_xy, _ = _refine_to_local_dark_minimum(
            image,
            candidate_xy,
            config.completion_refine_radius,
        )
        gaussian_row = _score_completion_detection(
            candidate_row,
            refined_xy,
            "completion_gaussian",
            current_xy,
            holes,
            image,
            nn_median,
            config,
        )

        accepted_row = gaussian_row
        if (
            gaussian_row["dark_zscore"] < config.completion_gaussian_dark_zscore
            and config.completion_map_if_weak
        ):
            positions, n_best, log_prob = detect_dark_atoms_in_local_patch(
                image,
                candidate_xy,
                patch_radius=config.completion_patch_radius,
                rho=config.completion_map_rho,
                N_max=config.completion_map_n_max,
                accept_radius=config.completion_accept_radius,
            )
            if len(positions):
                map_rows = [
                    _score_completion_detection(
                        candidate_row,
                        pos,
                        "completion_map",
                        current_xy,
                        holes,
                        image,
                        nn_median,
                        config,
                        map_n_best=n_best,
                        map_log_prob=log_prob,
                    )
                    for pos in positions
                ]
                accepted_row = max(map_rows, key=lambda row: row["acceptance_score"])
                gaussian_row["accepted"] = False
                gaussian_row["reject_reason"] = (
                    gaussian_row["reject_reason"] + ";superseded_by_map"
                ).strip(";")
                diagnostics.append(gaussian_row)
            else:
                gaussian_row["map_n_best"] = int(n_best)
                gaussian_row["map_log_prob"] = float(log_prob)

        if accepted_row["accepted"]:
            det = np.array([accepted_row["detection_x"], accepted_row["detection_y"]], dtype=np.float64)
            accepted.append(det)
            labels.append(str(accepted_row["completion_method"]))
            current_xy = np.vstack([current_xy, det.reshape(1, 2)])

        diagnostics.append(accepted_row)

    if not accepted:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=object),
            diagnostics,
        )
    return (
        np.array(accepted, dtype=np.float64),
        np.array(labels, dtype=object),
        diagnostics,
    )


def _selective_fallback_candidates(holes: np.ndarray,
                                   atom_xy: np.ndarray,
                                   image_shape: tuple[int, int],
                                   config: Stage2Config) -> np.ndarray:
    """
    Propose unresolved interior candidates from a relaxed hole-triplet pass.

    Candidates must be away from boundaries, away from existing atoms, and have
    enough nearby bright-hole support.  MAP is run only on these local regions.
    """
    relaxed = build_atom_positions_from_holes(
        holes,
        nn_radius=config.nn_radius_pixels,
        min_triplet_angle_deg=config.min_triplet_angle_deg,
        max_side_ratio=config.fallback_max_triplet_side_ratio,
        duplicate_radius=config.atom_duplicate_radius,
        nn_radius_tolerance=config.fallback_nn_radius_tolerance,
    )
    if len(relaxed) == 0:
        return relaxed

    keep = np.ones(len(relaxed), dtype=bool)
    if len(atom_xy):
        for i, pos in enumerate(relaxed):
            d_min = float(np.min(np.linalg.norm(atom_xy - pos, axis=1)))
            if d_min <= config.fallback_existing_atom_radius:
                keep[i] = False

    h, w = image_shape
    x, y = relaxed[:, 0], relaxed[:, 1]
    margin = config.fallback_boundary_margin
    keep &= (x >= margin) & (x <= w - 1 - margin) & (y >= margin) & (y <= h - 1 - margin)

    if len(holes):
        support = np.array([
            len(np.where(np.linalg.norm(holes - pos, axis=1) <= config.fallback_hole_support_radius)[0])
            for pos in relaxed
        ])
        keep &= support >= config.fallback_min_hole_support

    return relaxed[keep]


def _score_fallback_detection(candidate_xy: np.ndarray,
                              detection_xy: np.ndarray,
                              existing_xy: np.ndarray,
                              holes: np.ndarray,
                              image: np.ndarray,
                              nn_median: float,
                              config: Stage2Config) -> dict[str, Any]:
    """Score one local MAP fallback detection before it is merged."""
    candidate = np.asarray(candidate_xy, dtype=np.float64)
    detection = np.asarray(detection_xy, dtype=np.float64)

    candidate_shift = float(np.linalg.norm(detection - candidate))
    nearest_existing = _min_distance_to_points(existing_xy, detection)
    hole_support = _hole_support_count(
        holes,
        detection,
        config.fallback_hole_support_radius,
    )
    dark_zscore = _local_dark_zscore(image, detection, config.fallback_patch_radius)

    if np.isfinite(nn_median) and nn_median > 0:
        existing_nn_ratio = nearest_existing / nn_median
        min_existing_distance = max(
            config.fallback_existing_atom_radius,
            config.fallback_min_existing_nn_ratio * nn_median,
        )
    else:
        existing_nn_ratio = np.nan
        min_existing_distance = config.fallback_existing_atom_radius

    shift_score = max(
        0.0,
        1.0 - candidate_shift / max(config.fallback_accept_radius, 1e-12),
    )
    support_score = min(
        1.0,
        hole_support / max(float(config.fallback_min_hole_support), 1.0),
    )
    dark_score = min(
        1.0,
        max(0.0, dark_zscore) / max(config.fallback_min_dark_zscore, 1e-12),
    )
    distance_score = 1.0 if nearest_existing > min_existing_distance else 0.0
    geometry_score = 1.0
    if np.isfinite(existing_nn_ratio):
        geometry_score = float(
            config.fallback_min_existing_nn_ratio
            <= existing_nn_ratio
            <= config.fallback_max_existing_nn_ratio
        )

    score = shift_score + support_score + dark_score + distance_score + geometry_score

    reject_reasons: list[str] = []
    if candidate_shift > config.fallback_accept_radius:
        reject_reasons.append("too_far_from_relaxed_candidate")
    if hole_support < config.fallback_min_hole_support:
        reject_reasons.append("low_hole_support")
    if nearest_existing <= min_existing_distance:
        reject_reasons.append("too_close_to_existing_atom")
    if (
        np.isfinite(existing_nn_ratio)
        and existing_nn_ratio > config.fallback_max_existing_nn_ratio
    ):
        reject_reasons.append("too_isolated_from_existing_atoms")
    if not np.isfinite(dark_zscore) or dark_zscore < config.fallback_min_dark_zscore:
        reject_reasons.append("weak_dark_atom_contrast")
    if score < config.fallback_min_acceptance_score:
        reject_reasons.append("low_acceptance_score")

    return {
        "candidate_x": float(candidate[0]),
        "candidate_y": float(candidate[1]),
        "detection_x": float(detection[0]),
        "detection_y": float(detection[1]),
        "candidate_shift": candidate_shift,
        "nearest_existing_atom_dist": float(nearest_existing),
        "existing_nn_ratio": float(existing_nn_ratio),
        "hole_support": int(hole_support),
        "dark_zscore": float(dark_zscore),
        "shift_score": float(shift_score),
        "support_score": float(support_score),
        "dark_score": float(dark_score),
        "distance_score": float(distance_score),
        "geometry_score": float(geometry_score),
        "acceptance_score": float(score),
        "accepted": len(reject_reasons) == 0,
        "reject_reason": ";".join(reject_reasons),
    }


def _run_selective_map_fallback(image: np.ndarray,
                                fallback_candidates: np.ndarray,
                                holes: np.ndarray,
                                existing_xy: np.ndarray,
                                config: Stage2Config
                                ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run local dark-atom MAP and accept only well-supported gap fillers."""
    accepted: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    current_xy = np.asarray(existing_xy, dtype=np.float64).reshape(-1, 2).copy()
    nn_median = _nearest_neighbor_median(current_xy)

    for candidate_index, center_xy in enumerate(fallback_candidates):
        positions, n_best, log_prob = detect_dark_atoms_in_local_patch(
            image,
            center_xy,
            patch_radius=config.fallback_patch_radius,
            rho=config.fallback_map_rho,
            N_max=config.fallback_map_n_max,
            accept_radius=config.fallback_accept_radius,
        )
        if len(positions) == 0:
            diagnostics.append({
                "candidate_index": int(candidate_index),
                "candidate_x": float(center_xy[0]),
                "candidate_y": float(center_xy[1]),
                "detection_x": np.nan,
                "detection_y": np.nan,
                "map_n_best": int(n_best),
                "map_log_prob": float(log_prob),
                "candidate_shift": np.nan,
                "nearest_existing_atom_dist": np.nan,
                "existing_nn_ratio": np.nan,
                "hole_support": 0,
                "dark_zscore": np.nan,
                "shift_score": 0.0,
                "support_score": 0.0,
                "dark_score": 0.0,
                "distance_score": 0.0,
                "geometry_score": 0.0,
                "acceptance_score": 0.0,
                "accepted": False,
                "reject_reason": "no_local_map_detection",
            })
            continue

        for detection_index, detection_xy in enumerate(positions):
            row = _score_fallback_detection(
                center_xy,
                detection_xy,
                current_xy,
                holes,
                image,
                nn_median,
                config,
            )
            row.update({
                "candidate_index": int(candidate_index),
                "detection_index": int(detection_index),
                "map_n_best": int(n_best),
                "map_log_prob": float(log_prob),
            })

            if row["accepted"]:
                detection_arr = np.asarray(detection_xy, dtype=np.float64)
                accepted.append(detection_arr)
                current_xy = np.vstack([current_xy, detection_arr.reshape(1, 2)])

            diagnostics.append(row)

    if not accepted:
        return np.empty((0, 2), dtype=np.float64), diagnostics
    return np.array(accepted, dtype=np.float64), diagnostics


def _stage2_metrics(holes: np.ndarray,
                    atom_xy_initial: np.ndarray,
                    atom_xy_final: np.ndarray,
                    z_raw: np.ndarray,
                    z_smooth: np.ndarray,
                    n_map_atoms: int,
                    n_completion_atoms: int,
                    completion_diagnostics: list[dict[str, Any]],
                    n_fallback_candidates: int,
                    n_fallback_atoms: int,
                    fallback_diagnostics: list[dict[str, Any]],
                    n_atoms_before_merge: int,
                    config: Stage2Config) -> dict[str, Any]:
    """Collect run-level diagnostic metrics."""
    z_valid = np.isfinite(z_smooth)
    raw_valid = np.isfinite(z_raw)
    completion_candidate_ids = {
        int(row["candidate_index"])
        for row in completion_diagnostics
        if "candidate_index" in row and int(row["candidate_index"]) >= 0
    }
    n_completion_evaluated = len(completion_diagnostics)
    n_completion_accepted = sum(bool(row.get("accepted")) for row in completion_diagnostics)
    n_fallback_evaluated = len(fallback_diagnostics)
    n_fallback_accepted = sum(bool(row.get("accepted")) for row in fallback_diagnostics)
    return {
        "n_holes": int(len(holes)),
        "n_atoms_from_triplets": int(len(atom_xy_initial)),
        "n_map_atoms_added": int(n_map_atoms),
        "n_completion_candidates": int(len(completion_candidate_ids)),
        "n_completion_detections_evaluated": int(n_completion_evaluated),
        "n_completion_detections_accepted": int(n_completion_accepted),
        "n_completion_detections_rejected": int(n_completion_evaluated - n_completion_accepted),
        "n_completion_atoms_added": int(n_completion_atoms),
        "n_selective_fallback_candidates": int(n_fallback_candidates),
        "n_selective_fallback_detections_evaluated": int(n_fallback_evaluated),
        "n_selective_fallback_detections_accepted": int(n_fallback_accepted),
        "n_selective_fallback_detections_rejected": int(n_fallback_evaluated - n_fallback_accepted),
        "n_selective_fallback_atoms_added": int(n_fallback_atoms),
        "n_atoms_before_merge": int(n_atoms_before_merge),
        "n_atoms_final": int(len(atom_xy_final)),
        "raw_z_valid": int(raw_valid.sum()),
        "smooth_z_valid": int(z_valid.sum()),
        "raw_z_mean": float(np.nanmean(z_raw)) if len(z_raw) else np.nan,
        "raw_z_std": float(np.nanstd(z_raw)) if len(z_raw) else np.nan,
        "smooth_z_mean": float(np.nanmean(z_smooth)) if len(z_smooth) else np.nan,
        "smooth_z_std": float(np.nanstd(z_smooth)) if len(z_smooth) else np.nan,
        "lowess_frac": float(config.lowess_frac),
        "pcd_slope": float(config.pcd_slope),
        "z_scale": float(config.z_scale),
    }


def _save_csv(path: Path,
              atom_xy: np.ndarray,
              z_raw: np.ndarray,
              z_smooth: np.ndarray,
              xyz_angstrom: np.ndarray,
              source_labels: np.ndarray) -> None:
    """Save atom coordinates and z diagnostics as CSV."""
    header = (
        "atom_index,x_pixel,y_pixel,z_raw_angstrom,z_smooth_angstrom,"
        "x_angstrom,y_angstrom,z_angstrom,source"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for idx in range(len(atom_xy)):
            fh.write(
                f"{idx},{atom_xy[idx, 0]:.8f},{atom_xy[idx, 1]:.8f},"
                f"{z_raw[idx]:.8f},{z_smooth[idx]:.8f},"
                f"{xyz_angstrom[idx, 0]:.8f},{xyz_angstrom[idx, 1]:.8f},"
                f"{xyz_angstrom[idx, 2]:.8f},{source_labels[idx]}\n"
            )


def _csv_value(value: Any) -> str:
    """Format a simple scalar for the fallback diagnostics CSV."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{float(value):.8f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _save_fallback_diagnostics(path: Path,
                               diagnostics: list[dict[str, Any]]) -> None:
    """Save accepted/rejected fallback MAP candidates and local scores."""
    fieldnames = [
        "candidate_index",
        "detection_index",
        "candidate_x",
        "candidate_y",
        "detection_x",
        "detection_y",
        "map_n_best",
        "map_log_prob",
        "candidate_shift",
        "nearest_existing_atom_dist",
        "existing_nn_ratio",
        "hole_support",
        "dark_zscore",
        "shift_score",
        "support_score",
        "dark_score",
        "distance_score",
        "geometry_score",
        "acceptance_score",
        "accepted",
        "reject_reason",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for row in diagnostics:
            fh.write(",".join(_csv_value(row.get(name, "")) for name in fieldnames) + "\n")


def _save_completion_diagnostics(path: Path,
                                 diagnostics: list[dict[str, Any]]) -> None:
    """Save accepted/rejected local completion candidates and scores."""
    fieldnames = [
        "candidate_index",
        "proposal_source",
        "completion_method",
        "candidate_x",
        "candidate_y",
        "detection_x",
        "detection_y",
        "map_n_best",
        "map_log_prob",
        "candidate_shift",
        "nearest_existing_atom_dist",
        "existing_nn_ratio",
        "hole_support",
        "nearest_hole_dist",
        "nearest3_hole_mean",
        "nearest3_hole_cv",
        "dark_zscore",
        "shift_score",
        "support_score",
        "dark_score",
        "distance_score",
        "spacing_score",
        "hole_geometry_score",
        "method_score",
        "acceptance_score",
        "accepted",
        "reject_reason",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for row in diagnostics:
            fh.write(",".join(_csv_value(row.get(name, "")) for name in fieldnames) + "\n")


def _save_npz(path: Path,
              holes: np.ndarray,
              atom_xy: np.ndarray,
              z_raw: np.ndarray,
              z_smooth: np.ndarray,
              xyz_pixels: np.ndarray,
              xyz_angstrom: np.ndarray,
              source_labels: np.ndarray,
              fallback_diagnostics: list[dict[str, Any]],
              completion_diagnostics: list[dict[str, Any]],
              config: Stage2Config) -> None:
    """Save machine-readable Stage 2 arrays."""
    np.savez_compressed(
        path,
        holes_xy=holes,
        atom_xy_pixels=atom_xy,
        z_raw=z_raw,
        z_smooth=z_smooth,
        xyz_pixels=xyz_pixels,
        xyz_angstrom=xyz_angstrom,
        source_labels=source_labels.astype(str),
        fallback_diagnostics_json=json.dumps(fallback_diagnostics),
        completion_diagnostics_json=json.dumps(completion_diagnostics),
        config_json=json.dumps(asdict(config)),
    )


def _save_stage3_handoff_npz(path: Path,
                             atom_xy: np.ndarray,
                             z_raw: np.ndarray,
                             z_smooth: np.ndarray,
                             xyz_pixels: np.ndarray,
                             xyz_angstrom: np.ndarray,
                             source_labels: np.ndarray,
                             metrics: dict[str, Any],
                             config: Stage2Config) -> None:
    """
    Save the minimal Stage 3 initializer payload.

    This file deliberately focuses on the atomic model rather than all Stage 2
    intermediates.  Use the full NPZ if Stage 3 needs holes or fallback audit
    data for diagnosis.
    """
    np.savez_compressed(
        path,
        atom_ids=np.arange(len(xyz_angstrom), dtype=int),
        xyz_angstrom=xyz_angstrom,
        xyz_pixels=xyz_pixels,
        atom_xy_pixels=atom_xy,
        z_raw_angstrom=z_raw,
        z_smooth_angstrom=z_smooth,
        source_labels=source_labels.astype(str),
        metrics_json=json.dumps(metrics),
        config_json=json.dumps(asdict(config)),
        coordinate_note=(
            "Use xyz_angstrom as the Stage 3 initial structure. "
            "x/y are image pixels converted by pixel_size_angstrom; "
            "z is LOWESS-smoothed PCD height in Angstrom."
        ),
    )


def _save_sa_export_files(prefix: Path,
                          atom_xy: np.ndarray,
                          z_raw: np.ndarray,
                          z_smooth: np.ndarray,
                          xyz_pixels: np.ndarray,
                          xyz_angstrom: np.ndarray,
                          source_labels: np.ndarray,
                          metrics: dict[str, Any],
                          config: Stage2Config) -> dict[str, str]:
    """Write SA-specific initializer files from arrays during pipeline output."""
    result = Stage2Result(
        holes_xy=np.empty((0, 2), dtype=np.float64),
        atom_xy_pixels=atom_xy,
        z_raw=z_raw,
        z_smooth=z_smooth,
        xyz_pixels=xyz_pixels,
        xyz_angstrom=xyz_angstrom,
        source_labels=source_labels,
        fallback_diagnostics=[],
        completion_diagnostics=[],
        metrics=metrics,
        output_paths={},
    )
    return export_stage2_for_sa(
        result,
        prefix,
        extra_metadata={
            "config": asdict(config),
            "note": (
                "Generated automatically by run_stage2_initialization(). "
                "Full Stage 2 diagnostics are stored in the full NPZ and diagnostics CSV files."
            ),
        },
    )


def _save_visualizations(image: np.ndarray,
                         holes: np.ndarray,
                         atom_xy: np.ndarray,
                         z_smooth: np.ndarray,
                         source_labels: np.ndarray,
                         overlay_path: Path,
                         scatter3d_path: Path) -> None:
    """Save a 2-D overlay and 3-D scatter diagnostic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray", interpolation="nearest")
    if len(holes):
        ax.scatter(holes[:, 0], holes[:, 1], s=6, c="yellow",
                   marker=".", alpha=0.55, label=f"Holes ({len(holes)})")
    if len(atom_xy):
        labels = source_labels.astype(str)
        for label, color, marker in [
            ("triplet", "cyan", "."),
            ("defect_map", "magenta", "x"),
            ("fallback_map", "orange", "x"),
            ("completion_gaussian", "lime", "+"),
            ("completion_map", "red", "+"),
        ]:
            mask = labels == label
            if mask.any():
                ax.scatter(atom_xy[mask, 0], atom_xy[mask, 1], s=12,
                           c=color, marker=marker, alpha=0.85,
                           label=f"{label} ({int(mask.sum())})")
    ax.set_title("Stage 2 initialization overlay")
    ax.legend(fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    valid = np.isfinite(z_smooth)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    if valid.any():
        sc = ax.scatter(atom_xy[valid, 0], atom_xy[valid, 1], z_smooth[valid],
                        c=z_smooth[valid], s=10, cmap="viridis", alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="LOWESS z (Angstrom)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_zlabel("z (Angstrom)")
    ax.set_title("Stage 2 initialized 3D structure")
    fig.tight_layout()
    fig.savefig(scatter3d_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_outputs(output_dir: Path,
                  config: Stage2Config,
                  image: np.ndarray,
                  holes: np.ndarray,
                  atom_xy: np.ndarray,
                  z_raw: np.ndarray,
                  z_smooth: np.ndarray,
                  xyz_pixels: np.ndarray,
                  xyz_angstrom: np.ndarray,
                  source_labels: np.ndarray,
                  fallback_diagnostics: list[dict[str, Any]],
                  completion_diagnostics: list[dict[str, Any]],
                  metrics: dict[str, Any]) -> dict[str, str]:
    """Write CSV, NPZ, JSON metrics, and visual diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_prefix
    csv_path = output_dir / f"{prefix}.csv"
    npz_path = output_dir / f"{prefix}.npz"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    fallback_diag_path = output_dir / f"{prefix}_fallback_diagnostics.csv"
    completion_diag_path = output_dir / f"{prefix}_completion_diagnostics.csv"
    stage3_handoff_path = output_dir / f"{prefix}_stage3_handoff.npz"
    sa_export_prefix = output_dir / prefix
    overlay_path = output_dir / f"{prefix}_overlay.png"
    scatter3d_path = output_dir / f"{prefix}_3d_scatter.png"

    _save_csv(csv_path, atom_xy, z_raw, z_smooth, xyz_angstrom, source_labels)
    _save_npz(npz_path, holes, atom_xy, z_raw, z_smooth,
              xyz_pixels, xyz_angstrom, source_labels,
              fallback_diagnostics, completion_diagnostics, config)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    _save_fallback_diagnostics(fallback_diag_path, fallback_diagnostics)
    _save_completion_diagnostics(completion_diag_path, completion_diagnostics)
    _save_stage3_handoff_npz(stage3_handoff_path, atom_xy, z_raw, z_smooth,
                             xyz_pixels, xyz_angstrom, source_labels,
                             metrics, config)
    sa_paths = _save_sa_export_files(sa_export_prefix, atom_xy, z_raw, z_smooth,
                                     xyz_pixels, xyz_angstrom, source_labels,
                                     metrics, config)
    _save_visualizations(image, holes, atom_xy, z_smooth, source_labels,
                         overlay_path, scatter3d_path)

    return {
        "csv": str(csv_path),
        "npz": str(npz_path),
        "metrics_json": str(metrics_path),
        "fallback_diagnostics_csv": str(fallback_diag_path),
        "completion_diagnostics_csv": str(completion_diag_path),
        "stage3_handoff_npz": str(stage3_handoff_path),
        **sa_paths,
        "overlay_png": str(overlay_path),
        "scatter3d_png": str(scatter3d_path),
    }


def run_stage2_initialization(image: np.ndarray | str | Path,
                              config: Optional[Stage2Config] = None,
                              defect_mask: Optional[np.ndarray] = None,
                              output_dir: Optional[str | Path] = None) -> Stage2Result:
    """
    Run the final Stage 2 initialization pipeline on a preprocessed TEM image.

    Parameters
    ----------
    image : np.ndarray | str | Path
        Preprocessed 2-D TEM image or path to an image file.
    config : Stage2Config, optional
        Pipeline parameters. Defaults use the validated synthetic settings and
        LOWESS span 0.08.
    defect_mask : np.ndarray, optional
        Boolean mask for difficult defect regions. If provided and
        config.use_map_defects is True, MAP detections are added locally.
    output_dir : str | Path, optional
        Directory for CSV/NPZ/visualization outputs. If None, outputs are saved
        under outputs/stage2/ at the repository root.
    """
    if config is None:
        config = Stage2Config()

    image_arr = _as_image_array(image)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "outputs" / "stage2"
    output_dir = Path(output_dir)

    _log(config, "Stage 2 initialization")
    _log(config, f"  image shape: {image_arr.shape}")

    _log(config, "  1. Detecting bright hole centres...")
    holes = find_holes(
        image_arr,
        sigma=config.hole_sigma,
        connectivity=config.hole_connectivity,
        border_margin=config.hole_border_margin,
        min_response=config.hole_min_response,
        min_distance=config.hole_min_distance,
    )
    _log(config, f"     holes: {len(holes)}")

    _log(config, "  2. Building atom x-y candidates from hole triplets...")
    atom_xy = build_atom_positions_from_holes(
        holes,
        nn_radius=config.nn_radius_pixels,
        min_triplet_angle_deg=config.min_triplet_angle_deg,
        max_side_ratio=config.max_triplet_side_ratio,
        duplicate_radius=config.atom_duplicate_radius,
    )
    atom_xy_initial = atom_xy.copy()
    source_labels = np.array(["triplet"] * len(atom_xy), dtype=object)
    _log(config, f"     atoms from triplets: {len(atom_xy_initial)}")

    n_map_atoms = 0
    if config.use_map_defects and defect_mask is not None and np.any(defect_mask):
        _log(config, "  3. Adding MAP atoms in defect regions...")
        defect_xy = detect_defect_atoms(
            image_arr,
            defect_mask.astype(bool),
            patch_size=config.map_patch_size,
            overlap=config.map_overlap,
            rho=config.map_rho,
            N_max=config.map_n_max,
            duplicate_radius=config.map_duplicate_radius,
        )
        atom_xy, source_labels, n_map_atoms = _append_unique_positions(
            atom_xy,
            source_labels,
            defect_xy,
            "defect_map",
            config.atom_duplicate_radius,
        )
        _log(config, f"     MAP atoms added: {n_map_atoms}")
    else:
        _log(config, "  3. MAP defect step skipped.")

    n_completion_atoms = 0
    completion_diagnostics: list[dict[str, Any]] = []
    if config.enable_local_completion:
        _log(config, "  3a. Completing local interior atom gaps...")
        completion_xy, completion_labels, completion_diagnostics = _run_local_atom_completion(
            image_arr,
            holes,
            atom_xy,
            config,
        )
        atom_xy, source_labels, n_completion_atoms = _append_unique_labeled_positions(
            atom_xy,
            source_labels,
            completion_xy,
            completion_labels,
            config.atom_duplicate_radius,
        )
        n_completion_candidates = len({
            int(row["candidate_index"])
            for row in completion_diagnostics
            if "candidate_index" in row and int(row["candidate_index"]) >= 0
        })
        _log(config, f"     completion candidates: {n_completion_candidates}")
        _log(config, f"     completion detections evaluated: {len(completion_diagnostics)}")
        _log(config, f"     completion atoms added: {n_completion_atoms}")
    else:
        _log(config, "  3a. Local atom completion skipped.")

    n_fallback_candidates = 0
    n_fallback_atoms = 0
    fallback_diagnostics: list[dict[str, Any]] = []
    if config.use_selective_map_fallback:
        _log(config, "  3b. Running selective MAP fallback for sparse interior regions...")
        fallback_candidates = _selective_fallback_candidates(
            holes,
            atom_xy,
            image_arr.shape,
            config,
        )
        n_fallback_candidates = len(fallback_candidates)
        fallback_xy, fallback_diagnostics = _run_selective_map_fallback(
            image_arr,
            fallback_candidates,
            holes,
            atom_xy,
            config,
        )
        atom_xy, source_labels, n_fallback_atoms = _append_unique_positions(
            atom_xy,
            source_labels,
            fallback_xy,
            "fallback_map",
            config.atom_duplicate_radius,
        )
        _log(config, f"     fallback candidates: {n_fallback_candidates}")
        _log(config, f"     fallback detections evaluated: {len(fallback_diagnostics)}")
        _log(config, f"     fallback atoms added: {n_fallback_atoms}")
    else:
        _log(config, "  3b. Selective MAP fallback skipped.")

    n_atoms_before_merge = len(atom_xy)
    _log(config, f"     atoms after merge: {len(atom_xy)}")

    if len(atom_xy) == 0:
        raise RuntimeError("Stage 2 produced no atom x-y candidates.")

    if config.use_gaussian_refinement:
        _log(config, "  3c. Running iterative 2D Gaussian refinement...")
        from graphene3d.stage2.gaussian_refine import GaussianAtomRefiner
        refiner = GaussianAtomRefiner(
            patch_radius_px=config.gaussian_patch_radius_px,
            min_amplitude_snr=config.gaussian_min_snr,
            pixel_size_ang=config.pixel_size_angstrom,
            n_completion_passes=config.gaussian_n_completion_passes,
            refine_position=config.gaussian_refine_position,
        )
        # Pipeline uses (col, row); refiner uses (row, col) — convert
        positions_rc = atom_xy[:, ::-1].copy()
        refined_rc, _fit_quality, refine_status = refiner.refine(image_arr, positions_rc)
        _log(config, f"     {refine_status}")
        # Safety: only apply if refinement retains at least 70% of initial atoms
        if len(refined_rc) >= 0.70 * len(positions_rc):
            # Convert back from (row, col) to (col, row)
            atom_xy = refined_rc[:, ::-1].copy()
            # Rebuild source_labels: keep labels for retained atoms,
            # mark newly recovered completions as 'gaussian_refine'
            n_kept = refine_status["n_after_fit_rejection"]
            n_added = refine_status["n_added_by_completion"]
            n_final = refine_status["n_final"]
            old_labels = source_labels[:n_kept] if n_kept <= len(source_labels) else source_labels
            new_labels = np.array(["gaussian_refine"] * n_added, dtype=object)
            source_labels = np.concatenate([old_labels, new_labels])[:n_final]
        else:
            _log(config, f"     WARNING: refinement retained only {len(refined_rc)}/{len(positions_rc)} atoms "
                         f"(< 70%) — keeping original positions")

    _log(config, "  4. Estimating raw z with PCD...")
    z_raw = estimate_z_pcd(
        atom_xy,
        holes,
        image_arr,
        pcd_slope=config.pcd_slope,
        z_scale=config.z_scale,
        search_radius=config.pcd_search_radius,
    )

    _log(config, f"  5. Applying LOWESS smoothing, frac={config.lowess_frac:.2f}...")
    z_smooth = lowess_smooth_z(
        atom_xy,
        z_raw,
        frac=config.lowess_frac,
        n_iter=config.lowess_iter,
    )

    xyz_pixels = np.column_stack([atom_xy, z_smooth])
    xyz_angstrom = _build_xyz_angstrom(
        atom_xy,
        z_smooth,
        image_arr.shape,
        config.pixel_size_angstrom,
    )

    metrics = _stage2_metrics(
        holes,
        atom_xy_initial,
        atom_xy,
        z_raw,
        z_smooth,
        n_map_atoms,
        n_completion_atoms,
        completion_diagnostics,
        n_fallback_candidates,
        n_fallback_atoms,
        fallback_diagnostics,
        n_atoms_before_merge,
        config,
    )

    output_paths: dict[str, str] = {}
    if config.save_outputs:
        _log(config, "  6. Saving CSV, NPZ, metrics, and visualizations...")
        output_paths = _save_outputs(
            output_dir,
            config,
            image_arr,
            holes,
            atom_xy,
            z_raw,
            z_smooth,
            xyz_pixels,
            xyz_angstrom,
            source_labels,
            fallback_diagnostics,
            completion_diagnostics,
            metrics,
        )

    _log(config, "Stage 2 complete.")
    _log(config, f"  final atoms: {metrics['n_atoms_final']}")
    _log(config, f"  smooth z valid: {metrics['smooth_z_valid']}/{metrics['n_atoms_final']}")

    return Stage2Result(
        holes_xy=holes,
        atom_xy_pixels=atom_xy,
        z_raw=z_raw,
        z_smooth=z_smooth,
        xyz_pixels=xyz_pixels,
        xyz_angstrom=xyz_angstrom,
        source_labels=source_labels,
        fallback_diagnostics=fallback_diagnostics,
        completion_diagnostics=completion_diagnostics,
        metrics=metrics,
        output_paths=output_paths,
    )
