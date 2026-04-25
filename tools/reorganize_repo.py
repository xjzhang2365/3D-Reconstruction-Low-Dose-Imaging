"""
Safely reorganize the repository into the graphene3d package layout.

Default behavior is a dry run.  Use --apply to perform moves.

Examples
--------
Preview:
    python tools/reorganize_repo.py

Apply, refusing overwrites:
    python tools/reorganize_repo.py --apply

Apply and replace existing destination files:
    python tools/reorganize_repo.py --apply --force
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MoveSpec:
    src: str
    dst: str


@dataclass(frozen=True)
class TemplateSpec:
    dst: str
    content: str


DIRECTORIES = [
    "src/graphene3d",
    "src/graphene3d/preprocessing",
    "src/graphene3d/stage2",
    "src/graphene3d/stage3",
    "src/graphene3d/reports",
    "scripts",
    "data/preprocessed",
    "outputs/preprocessing",
]


PACKAGE_INITS = [
    "src/graphene3d/__init__.py",
    "src/graphene3d/stage2/__init__.py",
    "src/graphene3d/stage3/__init__.py",
    "src/graphene3d/reports/__init__.py",
]


MOVES = [
    MoveSpec("tem_preprocessing/averaging.py", "src/graphene3d/preprocessing/averaging.py"),
    MoveSpec("tem_preprocessing/corrections.py", "src/graphene3d/preprocessing/corrections.py"),
    MoveSpec("tem_preprocessing/denoising.py", "src/graphene3d/preprocessing/denoising.py"),
    MoveSpec("tem_preprocessing/__init__.py", "src/graphene3d/preprocessing/__init__.py"),
    MoveSpec("tem_preprocessing/make_report.py", "src/graphene3d/reports/make_report.py"),
    MoveSpec("tem_preprocessing/denoising_comparison.py", "scripts/denoising_comparison.py"),
    MoveSpec("tem_preprocessing/preprocessed_frame21.npy", "data/preprocessed/preprocessed_frame21.npy"),
    MoveSpec("stage2_atom_init/current_code/benchmark_synthetic_xy.py", "scripts/benchmark_synthetic_xy.py"),
    MoveSpec("stage2_atom_init/current_code/benchmark_raw_pcd_z.py", "scripts/benchmark_raw_pcd_z.py"),
    MoveSpec("stage2_atom_init/current_code/validate_stage2_against_ground_truth.py", "scripts/validate_stage2_against_ground_truth.py"),
    MoveSpec("stage2_atom_init/current_code/find_xy.py", "src/graphene3d/stage2/find_xy.py"),
    MoveSpec("stage2_atom_init/current_code/hole_finding.py", "src/graphene3d/stage2/hole_finding.py"),
    MoveSpec("stage2_atom_init/current_code/map_detection.py", "src/graphene3d/stage2/map_detection.py"),
    MoveSpec("stage2_atom_init/current_code/pcd_z.py", "src/graphene3d/stage2/pcd_z.py"),
    MoveSpec("stage2_atom_init/current_code/stage2_pipeline.py", "src/graphene3d/stage2/pipeline.py"),
    MoveSpec("stage2_atom_init/current_code/sa_refine.py", "src/graphene3d/stage3/sa_refine.py"),
]


TEMPLATES = [
    TemplateSpec(
        "scripts/run_stage2.py",
        '''"""Run Stage 2 initialization from the packaged source layout."""

from __future__ import annotations

from pathlib import Path

from graphene3d.stage2.pipeline import Stage2Config, run_stage2_initialization


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "stage2_atom_init" / "synthetic" / "img_noisy.tif"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "stage2"


def main() -> None:
    config = Stage2Config(output_prefix="stage2_initialization")
    run_stage2_initialization(DEFAULT_IMAGE, config=config, output_dir=DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
''',
    ),
    TemplateSpec(
        "scripts/run_stage3_smoke.py",
        '''"""Run a short Stage 3 SA smoke test from the packaged source layout."""

from __future__ import annotations

from graphene3d.stage3.sa_refine import SAConfig, run_sa_refinement


def main() -> None:
    config = SAConfig(n_iterations=25, output_prefix="stage3_sa_smoke")
    run_sa_refinement(config=config)


if __name__ == "__main__":
    main()
''',
    ),
]


def _repo_path(relative_path: str) -> Path:
    """Return a path anchored at the repository root."""
    path = REPO_ROOT / relative_path
    resolved_parent = path.parent.resolve()
    repo_root = REPO_ROOT.resolve()
    if repo_root not in (resolved_parent, *resolved_parent.parents):
        raise ValueError(f"Path escapes repository root: {relative_path}")
    return path


def _print_action(dry_run: bool, action: str, detail: str) -> None:
    prefix = "DRY-RUN" if dry_run else "APPLY"
    print(f"[{prefix}] {action}: {detail}")


def ensure_directories(dry_run: bool) -> None:
    """Create the target directory skeleton if needed."""
    for relative_dir in DIRECTORIES:
        path = _repo_path(relative_dir)
        if path.exists():
            _print_action(dry_run, "exists", relative_dir)
            continue
        _print_action(dry_run, "mkdir", relative_dir)
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)


def ensure_package_inits(dry_run: bool, force: bool) -> None:
    """Create empty package __init__.py files where no moved file provides one."""
    for relative_file in PACKAGE_INITS:
        path = _repo_path(relative_file)
        if path.exists() and not force:
            _print_action(dry_run, "exists", relative_file)
            continue
        if path.exists() and force:
            _print_action(dry_run, "overwrite-empty", relative_file)
        else:
            _print_action(dry_run, "touch", relative_file)
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")


def move_file(spec: MoveSpec, dry_run: bool, force: bool) -> str:
    """Move one file safely and return a status string."""
    src = _repo_path(spec.src)
    dst = _repo_path(spec.dst)

    if not src.exists():
        if dst.exists():
            _print_action(dry_run, "already-moved", f"{spec.src} -> {spec.dst}")
            return "already_moved"
        _print_action(dry_run, "missing-source", spec.src)
        return "missing_source"

    if src.is_dir():
        _print_action(dry_run, "skip-directory", spec.src)
        return "skip_directory"

    if dst.exists() and not force:
        _print_action(dry_run, "skip-existing-destination", f"{spec.src} -> {spec.dst}")
        return "destination_exists"

    if dst.exists() and force:
        _print_action(dry_run, "overwrite", spec.dst)

    _print_action(dry_run, "move", f"{spec.src} -> {spec.dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    return "moved"


def write_template(spec: TemplateSpec, dry_run: bool, force: bool) -> str:
    """Create a runner script if it does not already exist."""
    dst = _repo_path(spec.dst)
    if dst.exists() and not force:
        _print_action(dry_run, "skip-existing-destination", spec.dst)
        return "template_destination_exists"
    if dst.exists() and force:
        _print_action(dry_run, "overwrite-template", spec.dst)
    else:
        _print_action(dry_run, "create-template", spec.dst)

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(spec.content, encoding="utf-8")
    return "template_created"


def run(dry_run: bool, force: bool) -> int:
    """Run the repository reorganization plan."""
    print(f"Repository root: {REPO_ROOT}")
    print(f"Mode: {'dry-run' if dry_run else 'apply'}")
    print(f"Overwrite existing destinations: {force}")
    print("")

    ensure_directories(dry_run)
    ensure_package_inits(dry_run, force)

    counts: dict[str, int] = {}
    print("")
    for spec in MOVES:
        status = move_file(spec, dry_run=dry_run, force=force)
        counts[status] = counts.get(status, 0) + 1

    print("")
    for spec in TEMPLATES:
        status = write_template(spec, dry_run=dry_run, force=force)
        counts[status] = counts.get(status, 0) + 1

    print("")
    print("Summary:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    if dry_run:
        print("")
        print("Dry run only. Re-run with --apply to perform these moves.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely reorganize the repository into src/graphene3d."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the reorganization. Without this flag, the script only previews actions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files. Only meaningful with --apply.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = not args.apply
    if args.force and dry_run:
        print("Note: --force was provided, but this is still a dry run because --apply was not set.")
    return run(dry_run=dry_run, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
