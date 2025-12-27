# Methodology Overview

Detailed technical explanation of the reconstruction approach.

---

## 1. Problem Formulation

**Goal**: Reconstruct 3D atomic structure **S** from 2D projection image **I**

**Mathematical Framework:**
```
Given: I(x,y) = ∫ ρ(x,y,z) dz + noise
Find: S = {(x_i, y_i, z_i) | i=1..N}

Where:
- I(x,y): Observed 2D image (noisy, low-dose)
- ρ(x,y,z): 3D atomic potential
- S: Set of N atomic positions
- noise: Poisson (shot) + Gaussian (readout)
```

**Challenges:**
1. **Ill-posed inverse problem**: Infinite 3D structures can produce similar 2D projections
2. **Low signal-to-noise**: SNR < 5 in low-dose conditions
3. **No prior knowledge**: Unknown structure, unknown defects
4. **Real-time requirement**: Need fast reconstruction for dynamics

---

## 2. Preprocessing Pipeline

### 2.1 Temporal Averaging

**Purpose**: Reduce noise while maintaining temporal information

**Method**:
```python
I_avg(t) = (1/N) Σ I(t + Δt_i)    where i ∈ [-w/2, w/2]

Parameters:
- Window size N = 5 frames
- Frame interval Δt = 1ms
- Effective resolution: 5ms
```

**Theory**: For Poisson noise, SNR improves as √N

**Results**: 
- SNR improvement: ~2.2× (close to theoretical √5 = 2.24)
- Temporal resolution maintained: 5ms (sufficient for ms-scale dynamics)

### 2.2 BM3D Denoising

**Purpose**: Further noise reduction via advanced denoising

**Algorithm**:
1. **Block matching**: Find similar patches across image
2. **Collaborative filtering**: Denoise patches together in 3D transform domain
3. **Aggregation**: Combine denoised patches

**Parameters**:
```python
sigma_psd = auto-estimated from image
stage = 'all' (hard thresholding + Wiener filter)
```

**Performance**: Additional 1.5-2× noise reduction

**Total preprocessing improvement**: ~3-4× SNR increase

---

## 3. Structure Estimation

### 3.1 Gaussian Fitting (x,y positions)

**Model**: Each atomic column as 2D Gaussian
```
I(x,y) = A·exp(-[(x-x₀)²/σ_x² + (y-y₀)²/σ_y²]) + B

Parameters to fit:
- (x₀, y₀): Center position
- A: Amplitude
- σ_x, σ_y: Widths
- B: Background offset
```

**Method**:
1. Peak detection (local maxima)
2. Non-linear least squares fitting
3. Quality filtering

**Accuracy**: 
- Typical precision: 0.05-0.1 Å in x,y
- Depends on SNR and peak separation

### 3.2 PCD Method (z-heights)

**Principle**: Image intensity ∝ projected charge density
```
I(x,y) ∝ ∫ ρ(x,y,z) dz

Approximation:
z_estimate ∝ I(x,y) - I_background
```

**Process**:
1. Extract local intensities at each (x,y) position
2. Integrate intensity in small window
3. Convert to relative z-height using calibration

**Limitations**:
- Only provides relative z
- Approximate (assumes linear relationship)
- Serves as initial guess for optimization

### 3.3 Bayesian Refinement

**Framework**:
```
P(S|I) ∝ P(I|S)·P(S)

Where:
- P(S|I): Posterior (what we want)
- P(I|S): Likelihood (image given structure)
- P(S): Prior (expected structure properties)
```

**Priors**:
- Lattice structure (hexagonal for graphene)
- Bond length distribution
- Planarity constraints

**Inference**: Maximum A Posteriori (MAP) estimation

⚠️ **Note**: Specific prior formulations and likelihood functions are proprietary.

---

## 4. Optimization (Simulated Annealing)

### 4.1 Energy Function

**Conceptual Form**:
```
E(S) = w₁·E_image(S) + w₂·E_physics(S) + w₃·E_reg(S)

Components:
1. E_image: χ² between simulated and target image
2. E_physics: Physical constraint violations
3. E_reg: Regularization terms
```

**Image Matching Term**:
```
E_image = (1/N_pixels) Σ [I_sim(x,y) - I_target(x,y)]²

where I_sim = ForwardModel(S)
```

**Physics Term** (conceptual):
```
E_physics = Σ penalty(bond_length_i, expected_length)
          + Σ penalty(bond_angle_i, expected_angle)
          + planarity_deviation
```

⚠️ **Proprietary**: Exact formulations and weights (w₁, w₂, w₃)

### 4.2 Simulated Annealing Algorithm

**Pseudocode**:
```
Initialize:
  S = S_initial (from estimation)
  T = T_initial
  E_current = Energy(S)

Repeat:
  For k = 1 to iterations_per_temp:
    S_new = GenerateNeighbor(S)
    E_new = Energy(S_new)
    ΔE = E_new - E_current
    
    If ΔE < 0:
      S = S_new
      E_current = E_new
    Else:
      If random() < exp(-ΔE/T):
        S = S_new  (accept worse solution)
        E_current = E_new
    
    If k % validation_interval == 0:
      S = MDValidate(S)  (ensure physical plausibility)
  
  T = T × cooling_rate
  
Until: convergence or max_iterations
```

**Key Parameters** (proprietary):
- T_initial: Starting temperature
- cooling_rate: Temperature reduction factor
- iterations_per_temp: Steps per temperature
- convergence_threshold: Stopping criterion

### 4.3 Forward Model

**Purpose**: Simulate TEM image from atomic structure

**Method**: Multislice algorithm
```
For each slice z_i:
  1. Project atoms onto slice
  2. Calculate phase shift: φ(x,y) = σ·V(x,y)
  3. Apply transmission function: t(x,y) = exp(iφ(x,y))
  4. Propagate to next slice

Final: I(x,y) = |ψ(x,y)|²
```

**Software**:
- Original work: Tempas (commercial)
- Alternative: abTEM (open-source)

**Parameters**:
- Acceleration voltage: 80 kV
- Spherical aberration: Cs value
- Defocus: Δf value
- Pixel size: calibrated

---

## 5. Validation

### 5.1 Molecular Dynamics Check

**Purpose**: Ensure atomic positions are physically plausible

**Method**:
1. Take structure from SA iteration
2. Run short MD simulation (50ps)
3. Relax to energy minimum
4. Check if relaxation is small

**MD Parameters**:
```
Potential: Tersoff (C-C interactions)
Ensemble: NVT (Nose-Hoover)
Temperature: 300K
Timestep: 1 fs
Duration: 50 ps
```

**Acceptance Criterion**:
- If RMS displacement < threshold: Accept
- If large displacement: Reject (unphysical structure)

### 5.2 Convergence Metrics

**Track during optimization**:
1. χ² (energy)
2. Bond length distribution
3. Planarity measure
4. Temperature schedule

**Stopping Criteria**:
- Energy change < 0.1% for 3 consecutive iterations
- Or: Maximum iterations reached
- Or: MD validation fails repeatedly

---

## 6. Performance Analysis

### 6.1 Accuracy Metrics

**Root Mean Square Deviation (RMSD)**:
```
RMSD = √[(1/N) Σ (r_i - r_i_true)²]

Separate for x, y, z:
RMSD_z = √[(1/N) Σ (z_i - z_i_true)²]
```

**Results**:
- RMSD_x: 0.08 Å
- RMSD_y: 0.09 Å  
- RMSD_z: 0.45 Å
- Overall: 0.46 Å

### 6.2 Dose Dependence

**Tested doses**: 2.3×10³ to 2.7×10⁴ e⁻/Å²

**Findings**:
1. **Threshold**: 4.6×10³ e⁻/Å² (below = poor reconstruction)
2. **Optimal**: 6.4-9.1×10³ e⁻/Å² (best accuracy/dose tradeoff)
3. **Saturation**: >2×10⁴ e⁻/Å² (minimal improvement)



---

## 7. Extensions & Future Work

### 7.1 Medical Imaging Adaptation

**Required modifications**:
1. Replace TEM forward model → CT/MRI forward model
2. Adapt energy function → Anatomical constraints
3. Adjust noise model → Medical imaging noise characteristics

**Same framework applies**:
- SA optimization structure
- Physics-based validation
- Multi-scale approach

### 7.2 Real-Time Implementation

**Optimizations**:
- GPU acceleration of forward model
- Parallel tempering (multiple chains)
- Adaptive cooling schedule
- Smart initialization from previous frames

**Target**: <30 seconds per frame for real-time applications

---

## References

**Simulated Annealing**:
- Kirkpatrick et al., Science, 1983

**Gaussian Fitting, MAP rule**:
- Van Aert et al., Ultramicroscopy, 2005
- J Fatermans, PhysRevLett, 2018

**BM3D Denoising**:
- Dabov et al., IEEE TIP, 2007

**TEM Simulation**:
- Kirkland, "Advanced Computing in Electron Microscopy", 2020

**Full methodology**:
- Zhang et al., "Revealing 3D Atomic Dynamics..." (2025, in preparation)