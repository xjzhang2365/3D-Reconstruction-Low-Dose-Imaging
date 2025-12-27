# Comprehensive Preprocessing Pipeline

**Industrial-scale data processing for 50,000+ TEM images**

---

## Overview

The raw TEM data requires extensive preprocessing before structure estimation. This multi-stage pipeline transforms noisy, artifact-laden raw images into analysis-ready data suitable for high-precision 3D reconstruction.

**Scale of operation:**
- **50,000+ images** processed
- **Multiple correction stages** applied to each
- **Automated quality control** throughout pipeline
- **Parallel processing** for computational efficiency

---

## Pipeline Architecture
```
RAW TEM DATA (50,000+ images)
│
├─> STAGE 1: Image Quality Correction
│   ├─> Flat-field correction
│   └─> Dead pixel removal (statistical outlier detection)
│
├─> STAGE 2: Noise Reduction
│   ├─> Temporal averaging (5-frame window)
│   └─> Advanced denoising (BM3D / Dictionary Learning / CNN)
│
├─> STAGE 3: Initial Structure Estimation
│   ├─> Model-based estimation (Gaussian mixture models)
│   └─> PCD method (z-coordinate estimation)
│
└─> ANALYSIS-READY DATA
    ├─> Quality metrics validated
    └─> Ready for optimization
```

---

## Stage 1: Image Quality Correction

### 1.1 Flat-Field Correction

**Purpose**: Correct for non-uniform illumination and detector response

**Problem**: 
- TEM illumination not perfectly uniform
- Detector sensitivity varies across sensor
- Results in intensity gradients unrelated to sample

**Method**:
```
I_corrected(x,y) = [I_raw(x,y) - I_dark(x,y)] / [I_flat(x,y) - I_dark(x,y)]

Where:
- I_raw: Raw image with sample
- I_dark: Dark reference (no beam)
- I_flat: Flat-field reference (no sample)
```

**Implementation**:
- Acquired dark and flat references before imaging
- Applied correction to all 50,000+ images
- Polynomial fitting used when references unavailable

**Validation**:
- Background variance reduced by ~60%
- Intensity profiles across field of view flattened

### 1.2 Dead Pixel Removal

**Purpose**: Detect and correct malfunctioning detector pixels

**Problem**:
- Stuck pixels (always high/low)
- Hot pixels (random high values)
- Dead pixels (always zero)

**Statistical Detection Method**:
```python
# Outlier detection algorithm (simplified)

For each pixel position (i,j):
    1. Collect values across all 50,000+ images: V[i,j,:]
    2. Calculate statistics:
       - median: M[i,j] = median(V[i,j,:])
       - MAD: mad[i,j] = median(|V[i,j,:] - M[i,j]|)
       - z-score: z[i,j,k] = |V[i,j,k] - M[i,j]| / (1.4826 * mad[i,j])
    
    3. Flag outliers:
       IF z[i,j,k] > threshold (e.g., 5):
           Mark as bad pixel
    
    4. Correction:
       - Replace with median of neighboring pixels
       - Use inpainting for clusters
```

**Key Features**:
- **Robust statistics** (median/MAD instead of mean/std)
- **Per-pixel analysis** across entire dataset
- **Temporal consistency** check
- **Spatial correlation** for validation

**Results**:
- Detected ~150-200 problematic pixels per detector
- Automatic correction applied to all images
- Improved image quality consistency across dataset

**Computational Challenge**:
- 50,000 images × 2048×2048 pixels = ~200 billion pixel values
- Required efficient parallel processing
- Used chunked processing to manage memory

---

## Stage 2: Advanced Noise Reduction

### 2.1 Temporal Averaging

**Methodology**: Same as previously described
- 5-frame sliding window
- Reduces Poisson noise by √5 ≈ 2.24×
- Maintains 1ms temporal resolution

### 2.2 Spatial Denoising - Comparative Study

**Three methods tested on all images:**

#### Method A: BM3D (Block-Matching 3D)

**Algorithm**:
```
1. Block matching: Find similar patches across image
2. 3D transform: Stack similar patches → 3D array
3. Collaborative filtering: Denoise in transform domain
4. Aggregation: Combine denoised patches
```

**Performance**:
- PSNR improvement: ~3-5 dB
- Processing time: ~2-3 seconds per image
- Best for: General noise reduction

**Advantages**:
- Non-local approach (uses image self-similarity)
- Preserves fine details well
- Parameter-free (auto noise estimation)

#### Method B: Dictionary Learning

**Algorithm**:
```
1. Learn sparse dictionary from image patches
2. Represent each patch as sparse combination of dictionary atoms
3. Denoise by sparse coding with regularization
4. Reconstruct image from denoised sparse codes
```

**Implementation**:
- K-SVD algorithm for dictionary learning
- Orthogonal Matching Pursuit (OMP) for sparse coding
- Dictionary size: 256 atoms, patch size: 8×8

**Performance**:
- PSNR improvement: ~4-6 dB
- Processing time: ~10-15 seconds per image
- Best for: Structured/periodic patterns (ideal for lattices)

**Advantages**:
- Learns image-specific features
- Excellent for periodic structures
- Better preservation of atomic columns

#### Method C: CNN-Based Denoising

**Architecture**:
- U-Net style encoder-decoder
- Skip connections preserve spatial information
- Trained on paired noisy/clean synthetic data

**Training**:
- 10,000 synthetic image pairs generated
- Augmentation: rotation, translation, noise levels
- Loss: MSE + perceptual loss

**Performance**:
- PSNR improvement: ~5-7 dB
- Processing time: ~0.5 seconds per image (GPU)
- Best for: Speed with good quality

**Advantages**:
- Fastest inference (with GPU)
- Learns complex noise patterns
- Scalable to large datasets

#### Comparative Results

| Method | PSNR Gain | Speed | Memory | Best Use Case |
|--------|-----------|-------|--------|---------------|
| **BM3D** | 3-5 dB | Medium | Low | General purpose |
| **Dictionary Learning** | 4-6 dB | Slow | Medium | Periodic structures |
| **CNN** | 5-7 dB | Fast* | High | Large-scale processing |

*With GPU; slower on CPU than BM3D

**Final Choice**:
- **BM3D** for final pipeline (best quality/speed tradeoff without GPU)
- **Dictionary Learning** for critical high-quality reconstructions
- **CNN** for rapid initial screening

---

## Stage 3: Structure Estimation

### 3.1 Model-Based Estimation

**Theoretical Framework**:

TEM image formation model:
```
I(x,y) = Σ g_i(x,y) + noise

where g_i(x,y) = A_i · exp(-[(x-x_i)²/σ_x² + (y-y_i)²/σ_y²])

Parameters per atom i:
- (x_i, y_i): Position
- A_i: Amplitude (intensity)
- σ_x, σ_y: Spread (related to atomic number, defocus)
```

**Estimation Approach**:

**Step 1: Peak Detection**
- Apply Laplacian of Gaussian (LoG) filter
- Detect local maxima
- Filter by intensity threshold

**Step 2: Gaussian Mixture Model Fitting**
```python
# Simplified algorithm

For each detected peak:
    1. Extract local window (e.g., 15×15 pixels)
    2. Fit 2D Gaussian using:
       - Initial guess from peak detection
       - Non-linear least squares (Levenberg-Marquardt)
       - Bounds on parameters (physical constraints)
    
    3. Refine iteratively:
       - Re-estimate background locally
       - Account for overlapping neighbors
       - Update position estimates
```

**Challenges**:
- **Overlapping peaks**: Atoms closer than PSF width
- **Variable SNR**: Some atoms fainter than others
- **Background variation**: Require local estimation

**Solution**:
- Simultaneous fitting of overlapping Gaussians
- Bayesian priors on lattice positions
- Adaptive background modeling

**Accuracy**:
- Typical precision: 0.05-0.1 Å in x,y
- Depends on SNR, overlap, background

### 3.2 Projected Charge Density (PCD) Method

**Purpose**: Estimate z-coordinates from 2D intensity

**Physical Principle**:
```
I(x,y) ∝ ∫ ρ(x,y,z) dz

For isolated columns:
I(x,y) ≈ N_atoms × f(Z, defocus)

where:
- N_atoms: Number of atoms in column
- f(Z, defocus): Scattering factor
- Z: Atomic number
```

**Implementation**:

**Step 1: Calibration**
- Use known structures to establish I vs. z relationship
- Account for defocus, sample thickness

**Step 2: z-Estimation**
```python
For each atom at (x_i, y_i):
    1. Extract integrated intensity in local window
    2. Subtract background
    3. Apply calibration curve: z_i = f^(-1)(I_i)
    4. Normalize to relative z-heights
```

**Limitations**:
- Only gives relative heights (no absolute z)
- Assumes weak phase object approximation
- Sensitive to thickness variations

**Refinement**:
- Use as initial guess for optimization
- Apply smoothness constraints
- Validate with neighboring atoms

**Accuracy**:
- Initial z-estimate: ±0.5-1.0 Å
- Refined through optimization to <0.5 Å

---

## Quality Control & Validation

### Automated QC Metrics

**For each image processed:**

1. **SNR Assessment**
   - Signal: Mean intensity on atomic columns
   - Noise: Std deviation of background
   - Threshold: SNR > 3 for reliable estimation

2. **Resolution Check**
   - Measure point spread function width
   - Verify lattice spacing resolvable
   - Flag images with drift/vibration

3. **Artifact Detection**
   - Check for scan artifacts (line noise)
   - Detect ice contamination
   - Identify damaged regions

4. **Estimation Quality**
   - Verify expected number of atoms found
   - Check lattice parameter consistency
   - Validate z-height distribution

**Result**:
- Automatic flagging of poor-quality images
- Overall dataset quality score
- Success rate: 96% of images passed QC

---

## Computational Infrastructure

### Processing Statistics

**Hardware**:
- CPU: Multi-core workstation (24 cores)
- RAM: 64 GB
- Storage: 2 TB SSD for data

**Performance**:
- Flat-field correction: ~0.1 sec/image
- Dead pixel removal: One-time statistical analysis (~2 hours total)
- Temporal averaging: ~0.5 sec/frame set
- Denoising (BM3D): ~2 sec/image
- Structure estimation: ~5 sec/image

**Total processing time**:
- 50,000 images × ~8 sec/image = ~110 hours
- With parallelization (24 cores): ~5-6 hours
- Automated overnight processing

### Software Stack

**Languages & Frameworks**:
- Python 3.10 (primary)
- NumPy, SciPy (numerical computing)
- scikit-image (image processing)
- OpenCV (computer vision)
- Joblib (parallel processing)

**Custom Components**:
- Parallel image processor
- Statistical outlier detector
- Model-based estimator
- Quality control pipeline

---

## Key Achievements

**Data Engineering**:
- ✅ Processed 50,000+ images in automated pipeline
- ✅ Robust error handling and logging
- ✅ Efficient memory management for large datasets
- ✅ Parallel processing for computational efficiency

**Computer Vision**:
- ✅ Advanced denoising (3 methods compared)
- ✅ Statistical outlier detection at scale
- ✅ Model-based feature extraction
- ✅ Sub-pixel position estimation

**Statistical Analysis**:
- ✅ Robust statistics (median/MAD) for outlier detection
- ✅ Bayesian estimation framework
- ✅ Uncertainty quantification
- ✅ Quality control metrics

**Software Engineering**:
- ✅ Modular, maintainable code
- ✅ Automated quality checks
- ✅ Comprehensive logging
- ✅ Reproducible pipeline

---

