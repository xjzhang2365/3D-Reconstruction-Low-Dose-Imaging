# Stage 3 Notes

## Goal
Stage 3 refines the Stage 2 initialized graphene coordinates by minimizing a
TEM image-mismatch objective with simulated annealing (SA).  This stage is the
bridge between the geometry initializer and future physics-heavy refinement.

Current status: frozen no-MD SA baseline plus a first real MD-enabled option
using periodic weak LAMMPS coupling.  The no-MD baseline remains the default
reference path; the LAMMPS mode is opt-in through `MDConfig`.

## Current Working Baseline
Use the named preset in `src/graphene3d/stage3/sa_refine.py`:

```python
from graphene3d.stage3.sa_refine import make_stable_stage3_baseline_config

config = make_stable_stage3_baseline_config(
    target_image_path="data/simulated/target_preprocessed_like_raw21.npy",
    n_iterations=200,
)
```

Recommended settings:

| parameter | value |
|---|---:|
| `step_size_xy` | `0.10 A` |
| `step_size_z` | `0.06 A` |
| `initial_temperature` | `1.0e-4` |
| `cooling_rate` | `0.995` |
| `simulator_z_contrast_scale` | `0.08` |
| `enable_pre_sa_sanitization` | `True` |
| `enable_structural_rejection` | `True` |
| `structural_min_distance_angstrom` | `0.8 A` |
| `enable_structural_regularization` | `True` |
| `structural_regularization_weight` | `1.0e-3` |

Use this mode when the immediate goal is best image-objective progress on the
current debug simulator, or when comparing new ideas against the established
Stage 3 reference.

## First Real MD-Enabled Option
Use the frozen SA baseline together with the named periodic weak LAMMPS preset:

```python
from pathlib import Path
import shutil

from graphene3d.stage3.sa_refine import (
    make_periodic_weak_lammps_md_config,
    make_stable_stage3_baseline_config,
    run_sa_refinement,
)

lammps_root = Path(r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025")
workdir = Path("runs/lammps_periodic_weak")
workdir.mkdir(parents=True, exist_ok=True)
shutil.copy2(lammps_root / "Potentials" / "BNC.tersoff", workdir / "BNC.tersoff")

config = make_stable_stage3_baseline_config(
    target_image_path="data/simulated/target_preprocessed_like_raw21.npy",
    n_iterations=200,
)
md_config = make_periodic_weak_lammps_md_config(
    lammps_executable=lammps_root / "bin" / "lmp.exe",
    lammps_potential_file=lammps_root / "Potentials" / "BNC.tersoff",
    lammps_working_dir=workdir,
)

result = run_sa_refinement(config=config, md_config=md_config)
```

Preset settings:

| parameter | value |
|---|---:|
| `backend` | `lammps` |
| `lammps_execute` | `True` |
| `lammps_pair_style` | `tersoff` |
| `lammps_pair_coeff` | `* * BNC.tersoff C` |
| `apply_every_iterations` | `5` |
| `lammps_minimize_etol` | `1e-2` |
| `lammps_minimize_ftol` | `1e-2` |
| `lammps_minimize_maxiter` | `1` |
| `lammps_minimize_maxeval` | `5` |
| `lammps_timeout_seconds` | `20` |
| `max_displacement_angstrom` | `0.001 A` |

Use this mode when structural margin matters during SA refinement.  It is the
first validated real MD-enabled Stage 3 option: LAMMPS runs periodically, the
relaxed dump is parsed back into Python, and each atom's LAMMPS correction is
clipped to keep the image-driven proposal local.

## Baseline Result
Reference run:

- Run directory: `outputs/stage3/runs/sa_run_046`
- Output prefix: `stage3_sa_regularized_long`
- Iterations: `200`
- Target: `data/simulated/target_preprocessed_like_raw21.npy`

Result:

| metric | value |
|---|---:|
| initial total objective | `1.982808235607` |
| best total objective | `1.960725926140` |
| total improvement | `0.022082309466` |
| image objective improvement | `0.022083670714` |
| best structural contribution | `0.000004198848` |
| acceptance rate | `0.570` |
| worsening proposals accepted | `0.156863` |
| nearest-neighbor min after run | `0.874739 A` |
| close atoms `< 0.8 A` | `0` |

Coordinate motion:

| metric | value |
|---|---:|
| mean `|dx|` | `0.012821 A` |
| mean `|dy|` | `0.013882 A` |
| mean `|dz|` | `0.008076 A` |

## Current Components
1. Stage 2 SA input loader
2. Gaussian projection debug simulator
3. abTEM adapter scaffold
4. Image-mismatch objective
5. Pre-SA structure sanitization
6. Metropolis SA loop
7. Structural rejection for local overlaps
8. Lightweight nearest-neighbor structural regularization
9. Run-directory management and diagnostics
10. MD relaxation adapter boundary, including no-op, fake-MD, and real LAMMPS
    adapters

## Objective Definition
The current composite objective is:

```text
total_objective = image_objective + structural_regularization
```

The structural contribution is:

```text
structural_regularization =
    structural_regularization_weight * raw_nearest_neighbor_penalty
```

With the frozen baseline:

```text
structural_regularization_weight = 1e-3
```

This term is intentionally weak.  It records and gently discourages local
geometry distortion without replacing real MD.

## Simulator Status
The current working simulator mode is:

```text
simulator_kind = "gaussian_projection"
```

This is a debug proxy, not the final TEM physics model.  It uses dark Gaussian
spots and a weak bounded z-dependent contrast term:

```text
simulator_z_contrast_scale = 0.08
```

Setting `simulator_z_contrast_scale = 0.0` recovers the older z-insensitive
debug behavior.

The `AbTEMSimulatorAdapter` remains the intended Python-only path for more
physical TEM simulation.

## Inputs
Primary structure input:

- Stage 2 SA handoff NPZ
- Current default path:
  `outputs/stage2/validation/stage2_validation_init_sa_input.npz`

Required structure fields:

- `x_angstrom`
- `y_angstrom`
- `z_angstrom`
- atom IDs
- source labels
- initializer weights

Target image:

- Recommended first Stage 3 target:
  `data/simulated/target_preprocessed_like_raw21.npy`

## Outputs
Each SA run creates:

```text
outputs/stage3/runs/sa_run_XXX/
  config.json
  working/
  outputs/
  logs/
```

Important output files:

- best coordinates CSV
- best coordinates NPZ
- objective and proposal history CSV
- summary JSON
- coordinate diagnostics JSON
- direction diagnostics JSON
- displacement histograms

The history CSV records:

- total objective before/after
- image objective before/after
- structural regularization before/after
- proposal direction and size
- accept/reject decision
- structural rejection flag

## Current Limitations
- Gaussian projection is still a debug simulator.
- The z-sensitive contrast is a weak proxy, not a microscope model.
- Structural regularization is nearest-neighbor based only.
- The frozen no-MD baseline uses no force field and performs no MD relaxation.
- Periodic weak LAMMPS is available as an opt-in real-MD structural
  regularization mode, but it is not the default Stage 3 path.
- Objective weighting has only been tuned on short and 200-iteration simulated-target runs.

## MD Adapter Boundary
The code now defines a minimal future MD interface:

```python
@dataclass
class MDConfig:
    enabled: bool = False
    backend: str = "none"
    n_steps: int = 0
    timestep_fs: float = 0.0
    temperature_k: float = 300.0
    force_field: str = ""


class MDRelaxationAdapter(Protocol):
    def relax(
        self,
        coords_xyz_angstrom: np.ndarray,
        md_config: MDConfig,
    ) -> np.ndarray:
        ...
```

The provided implementation is `NoOpMDRelaxationAdapter`, which returns the
input coordinates unchanged.  A second test implementation,
`FakeMDRelaxationAdapter`, is available for checking the SA->MD boundary before
LAMMPS exists.

Frozen baseline behavior:

```text
proposal generation
-> structural rejection
-> image objective + lightweight structural regularization
-> Metropolis accept/reject
```

Optional fake-MD A/B test behavior:

```text
proposal generation
-> optional MDRelaxationAdapter.relax(proposal, md_config)
-> structural rejection
-> image objective + lightweight structural regularization
-> Metropolis accept/reject
```

This boundary is intentionally not active in the frozen baseline.  It exists so
LAMMPS or another MD backend can be added later without rewriting the image
objective or the SA proposal machinery.

To keep the frozen baseline unchanged, call `run_sa_refinement()` without
`md_config`, or pass `MDConfig(enabled=False)`.  To test the hook with the
fake relaxer, pass `MDConfig(enabled=True, backend="fake_md", ...)`.

### Fake-MD Test Relaxer
`FakeMDRelaxationAdapter` is a lightweight local-coordinate relaxer.  It is not
real molecular dynamics and should not be interpreted as a force-field result.
It only exists to test adapter plumbing before LAMMPS integration.

Behavior:

- detects atom pairs closer than `fake_min_distance_angstrom`
- pushes close pairs apart along their pair direction
- optionally nudges neighbor pairs toward `fake_target_nn_angstrom`
- clips per-pair and per-atom corrections to keep changes small
- stores a small `last_summary` after each call

Example:

```python
from graphene3d.stage3.sa_refine import (
    MDConfig,
    make_md_relaxation_adapter,
)

md_config = MDConfig(
    enabled=True,
    backend="fake_md",
    n_steps=5,
    fake_min_distance_angstrom=0.8,
    fake_target_nn_angstrom=1.42,
    fake_repulsion_strength=0.5,
    fake_bond_relaxation_strength=0.0,
)

adapter = make_md_relaxation_adapter(md_config)
relaxed = adapter.relax(proposal_coords, md_config)
```

SA A/B test usage:

```python
from graphene3d.stage3.sa_refine import (
    MDConfig,
    make_stable_stage3_baseline_config,
    run_sa_refinement,
)

config = make_stable_stage3_baseline_config(n_iterations=50)

# A: frozen behavior
baseline = run_sa_refinement(config=config)

# B: same SA settings, fake-MD hook enabled
fake_md = run_sa_refinement(
    config=config,
    md_config=MDConfig(enabled=True, backend="fake_md", n_steps=3),
)
```

Current frozen baseline:

```text
FakeMDRelaxationAdapter is not called unless md_config.enabled=True.
NoOpMDRelaxationAdapter remains the behavior-equivalent placeholder when MD is disabled.
```

### Fake-MD Stress-Test Result
Reference stress test:

- Summary: `outputs/stage3/tuning/stage3_fake_md_stress20_ab_summary.json`
- A no-MD run: `outputs/stage3/runs/sa_run_051`
- B fake-MD run: `outputs/stage3/runs/sa_run_052`
- Iterations: `20`
- Stress proposal settings:
  - `step_size_xy = 0.20 A`
  - `step_size_z = 0.12 A`
  - `fake_min_distance_angstrom = 1.0 A`
  - `fake_md_n_steps = 1`

Result:

| run | best total objective | total improvement | acceptance | NN min after run | close atoms `< 0.8 A` |
|---|---:|---:|---:|---:|---:|
| A: no MD | `1.979628207716` | `0.003180027891` | `0.400` | `0.820000 A` | `0` |
| B: fake-MD | `1.979857186016` | `0.002951049591` | `0.450` | `0.989805 A` | `0` |

Fake-MD diagnostics for run B:

| metric | value |
|---|---:|
| proposals corrected | `20 / 20` |
| mean correction magnitude | `0.020023564745 A` |
| total pair corrections | `39` |
| total bond corrections | `0` |

Conclusion:

- The SA->MD adapter hook is validated: fake-MD was called and corrected every
  proposal in the stress test.
- Under aggressive proposal settings, fake-MD improved structural safety by
  raising the nearest-neighbor minimum from about `0.82 A` to about `0.99 A`.
- The slightly smaller objective improvement in the fake-MD run is expected:
  local geometry correction constrains image-driven moves before objective
  evaluation.
- This confirms the boundary is useful for testing, but it is still not a
  substitute for real MD or LAMMPS.

## Future Stronger MD/LAMMPS Integration
Future work should keep the current Stage 3 baseline as a reproducible
reference and keep MD/LAMMPS as a separate, swappable structural module.

Suggested future boundary:

- Keep `ImageMismatchObjective` responsible only for TEM image mismatch.
- Keep the current `StructuralRegularizationObjective` as a lightweight fallback.
- Keep improving the MD/LAMMPS-backed relaxation adapter behind
  `MDRelaxationAdapter`.
- Compare new MD-assisted runs against `sa_run_046` before changing defaults.

Do not treat the current lightweight regularizer as final molecular dynamics.

## LAMMPS Adapter Scaffold
The first real LAMMPS boundary is now represented by
`LAMMPSRelaxationAdapter`.

Chosen integration design:

| boundary item | choice |
|---|---|
| input format | LAMMPS data file |
| relaxation mode | energy minimization |
| output format | `dump/custom` text |
| required output columns | `id x y z` |
| execution status | optional standalone subprocess execution |

LAMMPS-specific `MDConfig` fields include:

```python
MDConfig(
    enabled=True,
    backend="lammps",
    lammps_execute=True,
    lammps_executable="lmp",
    lammps_working_dir="outputs/stage3/lammps_work",
    lammps_data_filename="stage3_structure.data",
    lammps_input_filename="minimize.in",
    lammps_dump_filename="relaxed.dump",
    lammps_potential_file="path/to/potential",
    lammps_pair_style="...",
    lammps_pair_coeff="...",
    lammps_minimize_etol=1e-8,
    lammps_minimize_ftol=1e-10,
    lammps_minimize_maxiter=1000,
    lammps_minimize_maxeval=10000,
)
```

Current scaffold behavior:

1. `LAMMPSRelaxationAdapter.write_lammps_data(...)` writes a carbon-only
   atomic coordinate data file.
2. `LAMMPSRelaxationAdapter.write_lammps_input(...)` writes a minimization input
   script template.
3. `LAMMPSRelaxationAdapter.dump_path(...)` defines the expected dump/custom
   output path.
4. `LAMMPSRelaxationAdapter.parse_dump_custom(...)` parses the final dump frame
   and returns coordinates sorted by atom id.
5. `LAMMPSRelaxationAdapter.relax(...)` has two modes:
   - `lammps_execute=False`: write files and parse an existing dump if present.
   - `lammps_execute=True`: launch LAMMPS as a subprocess in the configured
     working directory, require the dump/custom output, then parse it.

Standalone helper:

```python
from graphene3d.stage3.sa_refine import (
    MDConfig,
    run_lammps_adapter_standalone_test,
)

summary = run_lammps_adapter_standalone_test(
    md_config=MDConfig(
        enabled=True,
        backend="lammps",
        lammps_execute=True,
        lammps_executable="lmp",
        lammps_working_dir="outputs/stage3/lammps_work",
        lammps_pair_style="...",
        lammps_pair_coeff="...",
    )
)
```

### Windows Standalone LAMMPS Test
Local Windows installation detected:

```text
C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025
```

Executable:

```text
C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\bin\lmp.exe
```

The exact requested file `CH.tersoff` was not present in the installation's
`Potentials/` folder.  If that file is added later, use this configuration and
copy `CH.tersoff` into the working directory, or make the pair coefficient use
the full potential path if the input template is updated:

```python
from graphene3d.stage3.sa_refine import MDConfig

md_config = MDConfig(
    enabled=True,
    backend="lammps",
    lammps_execute=True,
    lammps_executable=r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\bin\lmp.exe",
    lammps_working_dir="runs/lammps_adapter_test",
    lammps_potential_file=r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025\Potentials\CH.tersoff",
    lammps_pair_style="tersoff",
    lammps_pair_coeff="* * CH.tersoff C",
    lammps_minimize_etol=1e-8,
    lammps_minimize_ftol=1e-10,
    lammps_minimize_maxiter=100,
    lammps_minimize_maxeval=1000,
    lammps_timeout_seconds=60,
)
```

For the first real subprocess test, the adapter was validated with the bundled
carbon-capable Tersoff file `BNC.tersoff`:

```python
from pathlib import Path
import shutil
import numpy as np

from graphene3d.stage3.sa_refine import (
    MDConfig,
    run_lammps_adapter_standalone_test,
)

lammps_root = Path(r"C:\Users\xzhan\AppData\Local\LAMMPS 64-bit 22Jul2025")
workdir = Path("runs/lammps_adapter_test")
workdir.mkdir(parents=True, exist_ok=True)
shutil.copy2(lammps_root / "Potentials" / "BNC.tersoff", workdir / "BNC.tersoff")

coords = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.42, 0.0, 0.0],
        [0.71, 1.229756, 0.0],
    ],
    dtype=float,
)

summary = run_lammps_adapter_standalone_test(
    coords_xyz_angstrom=coords,
    md_config=MDConfig(
        enabled=True,
        backend="lammps",
        lammps_execute=True,
        lammps_executable=str(lammps_root / "bin" / "lmp.exe"),
        lammps_working_dir=str(workdir),
        lammps_potential_file=str(lammps_root / "Potentials" / "BNC.tersoff"),
        lammps_pair_style="tersoff",
        lammps_pair_coeff="* * BNC.tersoff C",
        lammps_minimize_etol=1e-8,
        lammps_minimize_ftol=1e-10,
        lammps_minimize_maxiter=100,
        lammps_minimize_maxeval=1000,
        lammps_timeout_seconds=60,
    ),
)
```

First standalone execution result:

| item | value |
|---|---:|
| working directory | `runs/lammps_adapter_test/` |
| LAMMPS return code | `0` |
| atoms | `3` |
| output dump | `runs/lammps_adapter_test/relaxed.dump` |
| summary JSON | `runs/lammps_adapter_test/standalone_summary.json` |
| max displacement | `0.152596 A` |
| mean displacement | `0.152595 A` |

Conclusion: the Python-to-LAMMPS standalone adapter boundary works on Windows.
This still is not inserted into the SA loop or made the default backend.

### First SA + LAMMPS A/B Test
Reference comparison:

- Summary: `outputs/stage3/tuning/stage3_sa_lammps_first_ab_10iter_summary.json`
- A no-MD run: `outputs/stage3/runs/sa_run_053`
- B LAMMPS run: `outputs/stage3/runs/sa_run_054`
- Iterations: `10`
- Random seed: `20260420`
- Target: `data/simulated/target_preprocessed_like_raw21.npy`
- Potential used: bundled `BNC.tersoff`
- LAMMPS working directory: `runs/lammps_sa_ab_test/lammps_work`

Shared SA settings:

| parameter | value |
|---|---:|
| `step_size_xy` | `0.10 A` |
| `step_size_z` | `0.06 A` |
| `initial_temperature` | `1.0e-4` |
| `simulator_z_contrast_scale` | `0.08` |
| `structural_regularization_weight` | `1.0e-3` |
| `enable_pre_sa_sanitization` | `True` |
| `enable_structural_rejection` | `True` |

LAMMPS settings for run B:

| parameter | value |
|---|---:|
| `backend` | `lammps` |
| `lammps_execute` | `True` |
| `lammps_pair_style` | `tersoff` |
| `lammps_pair_coeff` | `* * BNC.tersoff C` |
| `lammps_minimize_maxiter` | `2` |
| `lammps_minimize_maxeval` | `20` |
| `lammps_timeout_seconds` | `20` |

Result:

| run | initial total objective | best total objective | improvement | acceptance | NN min after run | close atoms `< 0.8 A` |
|---|---:|---:|---:|---:|---:|---:|
| A: frozen SA, no MD | `1.982808235607` | `1.981991226996` | `0.000817008611` | `0.800` | `0.820000 A` | `0` |
| B: SA + LAMMPS | `1.982808235607` | `1.982808235607` | `0.000000000000` | `0.000` | `0.820000 A` | `0` |

LAMMPS diagnostics for run B:

| metric | value |
|---|---:|
| relaxations requested | `10` |
| relaxations completed without subprocess failure | `10` |
| failures / timeouts | `0` |
| proposals corrected by LAMMPS | `10` |
| mean full-structure correction norm | `0.405172 A` |
| max full-structure correction norm | `0.417084 A` |
| mean max-atom correction | `0.206103 A` |

Conclusion:

- The real SA->LAMMPS execution path works: LAMMPS launched inside the SA loop,
  produced dump files, and the adapter parsed the relaxed coordinates.
- With even a two-iteration full-structure minimization, LAMMPS moves the
  proposal enough that all 10 tested moves were rejected by the current image
  objective and temperature.
- The frozen no-MD baseline remains unchanged and still serves as the reference.
- Next LAMMPS work should focus on making the relaxation gentler or more local
  before treating SA+LAMMPS as a productive refinement mode.

### Weakened SA + LAMMPS Coupling Test
Reference comparison:

- Summary: `outputs/stage3/tuning/stage3_sa_lammps_weakened_coupling_10iter_summary.json`
- A no-MD run: `outputs/stage3/runs/sa_run_055`
- B weakened LAMMPS probes: `outputs/stage3/runs/sa_run_056` to `sa_run_058`
- Iterations: `10`
- Random seed: `20260420`
- Potential used: bundled `BNC.tersoff`

Implementation change:

```text
LAMMPSRelaxationAdapter now supports MDConfig.max_displacement_angstrom.
When set, the adapter clips each atom's LAMMPS correction after parsing the
relaxed dump and before returning coordinates to SA.
When unset, previous LAMMPS behavior is unchanged.
```

Weakened LAMMPS settings:

| parameter | value |
|---|---:|
| `lammps_minimize_etol` | `1e-2` |
| `lammps_minimize_ftol` | `1e-2` |
| `lammps_minimize_maxiter` | `1` |
| `lammps_minimize_maxeval` | `5` |
| `lammps_timeout_seconds` | `20` |
| `max_displacement_angstrom` | swept: `0.001`, `0.002`, `0.005 A` |

Result:

| run | cap | best total objective | improvement | acceptance | successful relaxations | mean full-structure correction | mean max-atom correction | NN min after run | close atoms `< 0.8 A` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: frozen SA, no MD | none | `1.981991226996` | `0.000817008611` | `0.800` | `0` | `0.000000 A` | `0.000000 A` | `0.820000 A` | `0` |
| B: LAMMPS weak | `0.001 A` | `1.982050407005` | `0.000757828601` | `0.500` | `10` | `0.023322 A` | `0.001000 A` | `0.825723 A` | `0` |
| B: LAMMPS weak | `0.002 A` | `1.982340432771` | `0.000467802835` | `0.500` | `10` | `0.038869 A` | `0.002000 A` | `0.831457 A` | `0` |
| B: LAMMPS weak | `0.005 A` | `1.982267098394` | `0.000541137213` | `0.200` | `10` | `0.062595 A` | `0.005000 A` | `0.827156 A` | `0` |

Recommendation for the next SA+LAMMPS probe:

```python
MDConfig(
    enabled=True,
    backend="lammps",
    lammps_execute=True,
    lammps_pair_style="tersoff",
    lammps_pair_coeff="* * BNC.tersoff C",
    lammps_minimize_etol=1e-2,
    lammps_minimize_ftol=1e-2,
    lammps_minimize_maxiter=1,
    lammps_minimize_maxeval=5,
    lammps_timeout_seconds=20,
    max_displacement_angstrom=0.001,
)
```

Conclusion:

- Weak LAMMPS coupling is now usable as an in-loop test mode.
- The `0.001 A` per-atom cap recovered a meaningful acceptance rate and kept
  image-objective progress close to the no-MD baseline.
- The cap is essential for now: uncapped LAMMPS moved the proposal too far for
  the current SA temperature and image objective.
- This is still a coupling-strength test, not a final production MD setting.

### 50-Iteration Weak SA + LAMMPS A/B Test
Reference comparison:

- Summary: `outputs/stage3/tuning/stage3_sa_lammps_weak50_ab_summary.json`
- A no-MD run: `outputs/stage3/runs/sa_run_059`
- B weak LAMMPS run: `outputs/stage3/runs/sa_run_060`
- Iterations: `50`
- Random seed: `20260420`
- Potential used: bundled `BNC.tersoff`

Shared SA settings remained the frozen stable baseline.  Weak LAMMPS settings
used the recommended coupling from the 10-iteration probe:

| parameter | value |
|---|---:|
| `lammps_minimize_etol` | `1e-2` |
| `lammps_minimize_ftol` | `1e-2` |
| `lammps_minimize_maxiter` | `1` |
| `lammps_minimize_maxeval` | `5` |
| `lammps_timeout_seconds` | `20` |
| `max_displacement_angstrom` | `0.001 A` |

Result:

| run | best total objective | improvement | acceptance | NN min after run | close atoms `< 0.8 A` |
|---|---:|---:|---:|---:|---:|
| A: frozen SA, no MD | `1.974631771228` | `0.008176464379` | `0.740` | `0.820000 A` | `0` |
| B: weak SA + LAMMPS | `1.977391747357` | `0.005416488250` | `0.520` | `0.857406 A` | `0` |

LAMMPS diagnostics for run B:

| metric | value |
|---|---:|
| successful relaxations | `50` |
| failures / timeouts | `0` |
| mean full-structure correction norm | `0.021901 A` |
| mean max-atom correction | `0.001000 A` |

Conclusion:

- Weak real LAMMPS coupling remains stable over a 50-iteration controlled run.
- It improves structural margin relative to no-MD: nearest-neighbor minimum
  increased from about `0.820 A` to about `0.857 A`.
- It is still conservative for image refinement: acceptance and objective
  improvement are lower than the no-MD frozen baseline.
- Keep this setting as the current safe LAMMPS coupling baseline, but compare
  against no-MD before using it for longer production refinement.

### 50-Iteration Periodic Weak SA + LAMMPS A/B Test
Reference comparison:

- Summary: `outputs/stage3/tuning/stage3_sa_lammps_periodic50_every5_ab_summary.json`
- A no-MD run: `outputs/stage3/runs/sa_run_061`
- B periodic weak LAMMPS run: `outputs/stage3/runs/sa_run_062`
- Iterations: `50`
- Random seed: `20260420`
- Potential used: bundled `BNC.tersoff`

Implementation change:

```text
MDConfig.apply_every_iterations controls how often the MD adapter runs.
The default is 1, preserving the previous every-proposal behavior when MD is
enabled.  Setting apply_every_iterations=5 runs LAMMPS on iterations
0, 5, 10, ...
```

Periodic weak LAMMPS settings:

| parameter | value |
|---|---:|
| `apply_every_iterations` | `5` |
| `lammps_minimize_etol` | `1e-2` |
| `lammps_minimize_ftol` | `1e-2` |
| `lammps_minimize_maxiter` | `1` |
| `lammps_minimize_maxeval` | `5` |
| `lammps_timeout_seconds` | `20` |
| `max_displacement_angstrom` | `0.001 A` |

Result:

| run | best total objective | improvement | acceptance | NN min after run | close atoms `< 0.8 A` |
|---|---:|---:|---:|---:|---:|
| A: frozen SA, no MD | `1.974631771228` | `0.008176464379` | `0.740` | `0.820000 A` | `0` |
| B: periodic weak SA + LAMMPS | `1.975271441021` | `0.007536794585` | `0.740` | `0.827155 A` | `0` |

LAMMPS diagnostics for run B:

| metric | value |
|---|---:|
| LAMMPS cadence | every `5` iterations |
| relaxations executed | `10` |
| failures / timeouts | `0` |
| mean full-structure correction norm | `0.023518 A` |
| mean max-atom correction | `0.001000 A` |

Conclusion:

- Periodic weak LAMMPS is a better coupling pattern than every-proposal LAMMPS
  for the current image-driven SA loop.
- Acceptance matched the no-MD baseline while retaining some structural
  regularization benefit.
- Image-objective progress remained slightly lower than no-MD, but much closer
  than the every-proposal weak LAMMPS test.
- Current preferred real-LAMMPS test setting: `apply_every_iterations=5`,
  `max_displacement_angstrom=0.001`, and the same weak minimization limits.

### 200-Iteration Periodic Weak SA + LAMMPS A/B Test
Reference comparison:

- Summary: `outputs/stage3/tuning/stage3_sa_lammps_periodic200_every5_ab_summary.json`
- A no-MD run: `outputs/stage3/runs/sa_run_063`
- B periodic weak LAMMPS run: `outputs/stage3/runs/sa_run_064`
- Iterations: `200`
- Random seed: `20260420`
- Potential used: bundled `BNC.tersoff`

Shared SA settings remained the frozen stable baseline.  The periodic LAMMPS
branch used the same weak coupling settings:

| parameter | value |
|---|---:|
| `apply_every_iterations` | `5` |
| `lammps_minimize_etol` | `1e-2` |
| `lammps_minimize_ftol` | `1e-2` |
| `lammps_minimize_maxiter` | `1` |
| `lammps_minimize_maxeval` | `5` |
| `lammps_timeout_seconds` | `20` |
| `max_displacement_angstrom` | `0.001 A` |

Result:

| run | best total objective | improvement | acceptance | NN min after run | close atoms `< 0.8 A` |
|---|---:|---:|---:|---:|---:|
| A: frozen SA, no MD | `1.956583814385` | `0.026224421222` | `0.585` | `0.820000 A` | `0` |
| B: periodic weak SA + LAMMPS | `1.956987693586` | `0.025820542021` | `0.535` | `0.840083 A` | `0` |

LAMMPS diagnostics for run B:

| metric | value |
|---|---:|
| LAMMPS cadence | every `5` iterations |
| relaxations executed | `40` |
| failures / timeouts | `0` |
| mean full-structure correction norm | `0.023205 A` |
| mean max-atom correction | `0.001000 A` |

Conclusion:

- Periodic weak real LAMMPS remains stable over the first 200-iteration test.
- Image-objective progress is now very close to the no-MD baseline while still
  improving structural margin.
- The periodic LAMMPS branch raised nearest-neighbor minimum from about
  `0.820 A` to about `0.840 A` without introducing close atoms.
- Current practical conclusion: periodic weak LAMMPS is a viable structural
  regularization mode, but the no-MD frozen baseline still gives the slightly
  better pure image objective on this debug simulator.

Expected future active boundary:

```text
proposal generation
-> LAMMPSRelaxationAdapter writes data/input
-> standalone subprocess LAMMPS minimization, or future SA-managed call
-> parse dump/custom id x y z
-> structural rejection
-> image objective + lightweight structural regularization
-> Metropolis accept/reject
```

Current limitation:

```text
LAMMPS execution is available for standalone adapter tests and opt-in periodic
weak in-loop regularization.  It is not the default Stage 3 path, and stronger
LAMMPS coupling still needs more validation before production use.
```

## Superseded Real LAMMPS integration plan
The open questions below are retained as historical notes.  The active scaffold
choices are listed in the `LAMMPS Adapter Scaffold` section above.

### Goal
Use a real MD relaxation step inside the Stage 3 refinement loop.

### Input to LAMMPS
- atomic coordinates in Angstrom
- atom types (carbon only for now)
- simulation cell / box
- MD settings:
  - potential
  - number of relaxation steps
  - optional temperature / minimization mode

### Output from LAMMPS
- relaxed atomic coordinates in Angstrom

### Python ↔ LAMMPS boundary
1. Python writes a temporary structure file
2. Python calls LAMMPS
3. LAMMPS writes relaxed coordinates
4. Python reads relaxed coordinates back
5. SA continues with:
   proposal -> LAMMPS relax -> structural rejection -> objective

### Questions to resolve
- What structure file format should Python write?
- What LAMMPS input script template should be used?
- What potential file is needed?
- What output coordinate format should LAMMPS write?
- How many MD / minimization steps should be used per proposal?
