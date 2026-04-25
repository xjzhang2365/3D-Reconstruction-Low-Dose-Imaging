"""Run a short Stage 3 SA smoke test from the packaged source layout."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphene3d.stage3.sa_refine import SAConfig, run_sa_refinement


METADATA_PATH = REPO_ROOT / "data" / "simulated" / "target_metadata.json"
DEFAULT_TARGET = REPO_ROOT / "data" / "simulated" / "target_preprocessed_like_raw21.npy"
DEFAULT_SA_INPUT = REPO_ROOT / "outputs" / "stage2" / "validation" / "stage2_validation_init_sa_input.npz"


def _recommended_target_from_metadata() -> Path:
    if not METADATA_PATH.exists():
        return DEFAULT_TARGET
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    recommended = metadata.get("recommended_stage3_target")
    if not recommended:
        return DEFAULT_TARGET
    return REPO_ROOT / recommended


def main() -> None:
    target_path = _recommended_target_from_metadata()
    config = SAConfig(
        n_iterations=10,
        output_prefix="stage3_sa_simulated_target_smoke",
        target_image_path=str(target_path),
    )
    result = run_sa_refinement(
        sa_input_path=DEFAULT_SA_INPUT,
        config=config,
    )
    accepted = int(result.acceptance_history[:, 1].sum()) if len(result.acceptance_history) else 0
    acceptance_rate = accepted / max(config.n_iterations, 1)
    print("")
    print("Stage 3 simulated-target smoke summary")
    print("======================================")
    print(f"SA input          : {DEFAULT_SA_INPUT}")
    print(f"target image      : {target_path}")
    print(f"initial objective : {result.initial_objective:.6f}")
    print(f"best objective    : {result.best_objective:.6f}")
    print(f"acceptance rate   : {acceptance_rate:.3f}")
    print(f"run directory     : {result.output_paths.get('run_dir', '')}")


if __name__ == "__main__":
    main()
