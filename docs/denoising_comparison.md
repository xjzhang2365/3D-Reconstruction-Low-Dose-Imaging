# Denoising Methods Comparison

**Comprehensive evaluation of three denoising approaches for low-dose TEM imaging**

Applied to 50,000+ experimental images to determine optimal method for reconstruction pipeline.

---

## Overview

Low-dose imaging presents extreme noise challenges (SNR < 5). Three state-of-the-art denoising methods were evaluated:

1. **BM3D** - Block-Matching 3D (classical)
2. **Dictionary Learning** - Sparse representation (adaptive)
3. **CNN** - Deep learning (data-driven)

---

## Method 1: BM3D (Block-Matching 3D)

### Algorithm Description

**Core Idea**: Exploit self-similarity in images

**Three-Stage Process**:

1. **Block Matching**
   - Divide image into overlapping patches (8×8 pixels)
   - For each reference patch, find similar patches across image
   - Group similar patches into 3D arrays (2D patches stacked)

2. **Collaborative Filtering**
   - Apply 3D transform (DCT or wavelets) to patch groups
   - Perform shrinkage in transform domain
   - Inverse transform to get denoised patches

3. **Aggregation**
   - Average overlapping denoised patches
   - Weighted combination based on reliability

**Mathematical Framework**:
```
Stage 1 (Hard Thresholding):
For reference patch P_ref at position x:
  1. Find similar patches: S_x = {P_i : d(P_ref, P_i) < τ}
  2. Stack into 3D group: G_x = stack(S_x)
  3. 3D transform: Ĝ_x = T_3D(G_x)
  4. Hard threshold: Ĝ_x^hard = threshold(Ĝ_x, λ_hard)
  5. Inverse: G_x^est = T_3D^(-1)(Ĝ_x^hard)

Stage 2 (Wiener Filtering):
  - Use Stage 1 result as pilot estimate
  - Wiener filtering in transform domain
  - Better preservation of fine details
```

**Parameters**:
- Patch size: 8×8 pixels
- Search window: 39×39 pixels
- Max similar patches: 16
- Transform: DCT (Discrete Cosine Transform)
- Threshold: Automatically estimated from noise σ

**Advantages**:
- ✅ No training required
- ✅ Works well on diverse images
- ✅ Preserves sharp edges
- ✅ Automatic noise estimation

**Disadvantages**:
- ❌ Computationally expensive
- ❌ Fixed patch-based approach
- ❌ May over-smooth textures

**Performance**:
- PSNR improvement: 3-5 dB
- Processing time: 2-3 seconds/image (CPU)
- Memory: Low (~200 MB)

---

## Method 2: Dictionary Learning

### Algorithm Description

**Core Idea**: Represent image patches as sparse combinations of learned basis functions

**Training Phase** (K-SVD Algorithm):
```
Goal: Learn dictionary D ∈ R^(n×K) where:
  - n = patch dimension (64 for 8×8 patches)
  - K = number of atoms (typically 256-512)

Algorithm:
1. Initialize dictionary D randomly or with DCT basis

2. Iterate until convergence:
   a) Sparse Coding Stage:
      For each patch y_i from training images:
        Find sparse code α_i:
        minimize ||y_i - D·α_i||² + λ||α_i||₀
        
        Solution: Orthogonal Matching Pursuit (OMP)
        - Iteratively add atoms that best correlate
        - Stop when residual small or max atoms reached
   
   b) Dictionary Update Stage:
      For each atom d_k in D:
        - Find patches using this atom
        - Update d_k and corresponding coefficients
        - SVD-based update for optimal reconstruction
```

**Denoising Phase**:
```
For noisy image I:
1. Extract all overlapping patches: {y_i}

2. For each patch y_i:
   a) Sparse coding (OMP):
      α_i = argmin ||y_i - D·α||² + λ||α||₀
   
   b) Reconstruct:
      ŷ_i = D·α_i
   
3. Average overlapping patches → denoised image
```

**Mathematical Details**:

**Sparse Coding (OMP)**:
```python
# Orthogonal Matching Pursuit
Input: Noisy patch y, Dictionary D, Sparsity L

1. Initialize:
   r = y              # Residual
   Λ = {}             # Active set (selected atoms)
   
2. For l = 1 to L:
   # Find best matching atom
   k* = argmax |<r, d_k>|
   
   # Add to active set
   Λ = Λ ∪ {k*}
   
   # Update coefficients (least squares on active atoms)
   α_Λ = (D_Λ^T D_Λ)^(-1) D_Λ^T y
   
   # Update residual
   r = y - D_Λ α_Λ
   
   # Stop if residual small
   if ||r|| < ε: break

3. Return sparse code α
```

**K-SVD Dictionary Update**:
```python
For atom k:
  1. Find patches using this atom:
     ω_k = {i : α_i[k] ≠ 0}
  
  2. Compute residual error (excluding atom k):
     E_k = Y_ω_k - Σ_{j≠k} d_j α^j_ω_k
  
  3. SVD of error matrix:
     E_k = U Σ V^T
  
  4. Update:
     d_k = u_1         # First column of U
     α^k_ω_k = σ_1 v_1  # Scaled first column of V
```

**Implementation Details**:

**Training Data Preparation**:
```python
1. Collect clean/less-noisy patches:
   - From high-dose images
   - From synthetic structures
   - From temporally averaged sequences
   
2. Augmentation:
   - Rotation (90°, 180°, 270°)
   - Flipping (horizontal, vertical)
   - Ensures diverse dictionary
   
3. Dataset size:
   - ~100,000 patches for robust learning
   - Patches normalized to [0, 1]
```

**Parameters**:
- Patch size: 8×8 pixels (64 dimensions)
- Dictionary size: 256 atoms (4× overcomplete)
- Sparsity: L = 5-10 atoms per patch
- λ (regularization): 0.1 * σ_noise
- Training iterations: 20-30

**Advantages**:
- ✅ Learns image-specific features
- ✅ Excellent for structured/periodic patterns (perfect for atomic lattices!)
- ✅ Adaptive to image characteristics
- ✅ Theoretical sparse representation framework

**Disadvantages**:
- ❌ Requires training phase (time-consuming)
- ❌ Slower than BM3D (10-15 sec/image)
- ❌ Performance depends on dictionary quality
- ❌ More memory intensive

**Performance**:
- PSNR improvement: 4-6 dB
- Processing time: 10-15 seconds/image (CPU)
- Training time: 2-3 hours (one-time)
- Memory: Medium (~500 MB)

**Why It Works Well for TEM**:
- Atomic lattices are inherently sparse
- Periodic structures → efficient representation
- Dictionary atoms learn "atomic column patterns"
- Better than BM3D for structured materials

---

## Method 3: CNN (Convolutional Neural Network)

### Architecture: U-Net for Denoising

**Core Idea**: Learn end-to-end mapping from noisy → clean images

**U-Net Architecture**:
```
Input: Noisy Image (H × W × 1)

┌─────────────────────────────────────────────────────────┐
│                    ENCODER (Downsampling)                │
├─────────────────────────────────────────────────────────┤
│ Conv(64) → ReLU → Conv(64) → ReLU → MaxPool ────┐      │
│         ↓                                         │      │
│ Conv(128) → ReLU → Conv(128) → ReLU → MaxPool ──┼──┐   │
│         ↓                                         │  │   │
│ Conv(256) → ReLU → Conv(256) → ReLU → MaxPool ──┼──┼─┐ │
│         ↓                                         │  │ │ │
│ Conv(512) → ReLU → Conv(512) → ReLU  (Bottleneck)│  │ │ │
│                                                   │  │ │ │
├─────────────────────────────────────────────────────────┤
│                    DECODER (Upsampling)                  │
├─────────────────────────────────────────────────────────┤
│         ↓                                         │  │ │ │
│ UpConv(256) → Concat ←──────────────────────────┘  │ │ │
│ Conv(256) → ReLU → Conv(256) → ReLU                │ │ │
│         ↓                                            │ │ │
│ UpConv(128) → Concat ←─────────────────────────────┘ │ │
│ Conv(128) → ReLU → Conv(128) → ReLU                  │ │
│         ↓                                              │ │
│ UpConv(64) → Concat ←────────────────────────────────┘ │
│ Conv(64) → ReLU → Conv(64) → ReLU                      │
│         ↓                                                │
│ Conv(1) → Output: Clean Image (H × W × 1)              │
└─────────────────────────────────────────────────────────┘

Key Features:
- Skip connections (arrows) preserve spatial information
- Encoder: Extract hierarchical features
- Decoder: Reconstruct clean image
- Symmetric architecture
```

**Detailed Layer Configuration**:

**Encoder Block**:
```python
def encoder_block(input, filters):
    conv1 = Conv2D(filters, 3, padding='same', activation='relu')(input)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(filters, 3, padding='same', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    return conv2, pool  # conv2 for skip connection, pool for next layer
```

**Decoder Block**:
```python
def decoder_block(input, skip_connection, filters):
    up = UpSampling2D(size=(2, 2))(input)
    up = Conv2D(filters, 2, padding='same', activation='relu')(up)
    
    concat = Concatenate()([up, skip_connection])
    
    conv1 = Conv2D(filters, 3, padding='same', activation='relu')(concat)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(filters, 3, padding='same', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    
    return conv2
```

**Training Strategy**:

**1. Dataset Generation**:
```python
# Synthetic paired data
For each clean structure:
    1. Generate clean TEM image (forward model)
    2. Add Poisson + Gaussian noise (realistic low-dose)
    3. Store pair: (noisy, clean)

Total pairs: 10,000
- Clean structures: Various atomic configurations
- Noise levels: σ ∈ [10, 50] (covering experimental range)
- Augmentation: Rotation, flipping → 40,000 pairs
```

**2. Loss Function**:
```python
# Combined loss for better results

L = α * L_MSE + β * L_perceptual

Where:
L_MSE = (1/N) Σ ||y_pred - y_true||²

L_perceptual = ||φ(y_pred) - φ(y_true)||²
# φ = features from pre-trained VGG network
# Preserves structural information better

Typical: α = 1.0, β = 0.01
```

**3. Training Configuration**:
```python
Optimizer: Adam
  - Learning rate: 1e-4
  - β1 = 0.9, β2 = 0.999

Learning rate schedule:
  - Reduce by 0.5 when validation loss plateaus
  - Min LR: 1e-6

Batch size: 16 (limited by GPU memory)
Epochs: 100
Early stopping: Patience = 10 epochs

Data split:
  - Training: 70% (28,000 pairs)
  - Validation: 15% (6,000 pairs)
  - Test: 15% (6,000 pairs)
```

**4. Augmentation Pipeline**:
```python
During training (on-the-fly):
  - Random rotation: [0°, 90°, 180°, 270°]
  - Random flip: horizontal, vertical
  - Random crop: 256×256 from 512×512
  - Noise level randomization: σ ~ Uniform(10, 50)
  
Purpose: Prevent overfitting, improve generalization
```

**Inference**:
```python
def denoise_image(noisy_image):
    # Normalize
    normalized = (noisy_image - mean) / std
    
    # Predict
    with torch.no_grad():  # or TensorFlow equivalent
        clean = model(normalized)
    
    # Denormalize
    output = clean * std + mean
    
    return output

# Processing time: ~0.5 seconds (GPU), ~5 seconds (CPU)
```

**Advantages**:
- ✅ Fastest inference (with GPU): 0.5 sec/image
- ✅ Learns complex noise patterns
- ✅ End-to-end optimization
- ✅ Best PSNR improvement: 5-7 dB
- ✅ Scalable to large datasets

**Disadvantages**:
- ❌ Requires large training dataset
- ❌ Training time: ~8-12 hours (GPU)
- ❌ Needs GPU for practical speed
- ❌ May generalize poorly to unseen noise
- ❌ "Black box" - less interpretable

**Performance**:
- PSNR improvement: 5-7 dB
- Processing time: 0.5 sec/image (GPU), ~5 sec (CPU)
- Training time: 8-12 hours (one-time, GPU)
- Memory: High during training (~8 GB GPU)

---

## Comparative Results

### Quantitative Comparison

**Test Dataset**: 1,000 experimental low-dose TEM images

| Metric | Original | BM3D | Dict. Learning | CNN |
|--------|----------|------|----------------|-----|
| **PSNR (dB)** | 18.2 | 22.1 | 23.4 | 24.8 |
| **SSIM** | 0.45 | 0.72 | 0.78 | 0.82 |
| **Processing Time** | - | 2.3 s | 12.1 s | 0.5 s (GPU) |
| **Memory Usage** | - | 200 MB | 500 MB | 2 GB (inference) |
| **Setup Time** | - | None | 3h (training) | 10h (training) |

**PSNR**: Peak Signal-to-Noise Ratio (higher = better)  
**SSIM**: Structural Similarity Index (0-1, higher = better)

### Qualitative Assessment

**Visual Quality**:

| Aspect | BM3D | Dictionary Learning | CNN |
|--------|------|---------------------|-----|
| **Edge Preservation** | ★★★★☆ | ★★★★★ | ★★★★☆ |
| **Texture** | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **Artifacts** | Few | Very few | Occasional smoothing |
| **Atomic Columns** | Clear | Very clear | Clear |
| **Background** | Slight over-smooth | Natural | Very smooth |

### Reconstruction Impact

**Effect on downstream 3D reconstruction**:

| Method | RMSD_x,y (Å) | RMSD_z (Å) | Success Rate |
|--------|--------------|------------|--------------|
| No denoising | 0.15 | 0.82 | 68% |
| BM3D | 0.09 | 0.48 | 94% |
| Dict. Learning | 0.08 | 0.45 | 96% |
| CNN | 0.08 | 0.46 | 95% |

**Key Finding**: All three methods enable high-quality reconstruction, but Dictionary Learning gives slight edge for atomic lattices.

---

## Method Selection Rationale

### For Production Pipeline: **BM3D** Selected

**Reasons**:
1. ✅ No training required (immediate deployment)
2. ✅ Consistent performance across images
3. ✅ Reasonable speed (2-3 sec acceptable for quality)
4. ✅ Low memory footprint (can run on any workstation)
5. ✅ Well-validated in literature
6. ✅ Automatic parameter selection

**Trade-offs accepted**:
- Slightly lower PSNR than CNN (-2 dB acceptable)
- Slower than CNN with GPU (but no GPU required)

### When to Use Dictionary Learning

**Best for**:
- Critical reconstructions requiring absolute best quality
- Periodic/structured materials (graphene, crystals)
- When 10-15 sec/image is acceptable
- Research requiring interpretable features

**Used in our study**:
- Validation dataset (highest quality needed)
- Publication figures
- Challenging samples (defects, edges)

### When to Use CNN

**Best for**:
- Very large datasets (100,000+ images)
- Real-time processing requirements
- GPU resources available
- Diverse noise characteristics (after training)

**Potential applications**:
- High-throughput screening
- Online processing during acquisition
- Batch reprocessing of archives

---

## Implementation Considerations

### Computational Resources

**BM3D**:
```
Hardware: Standard CPU (multi-core)
Memory: 8 GB RAM sufficient
Parallelization: Per-image (trivial)
Scalability: Linear with image count
```

**Dictionary Learning**:
```
Hardware: CPU (multi-core helpful)
Memory: 16 GB RAM recommended
Training: One-time (~3 hours)
Parallelization: Patch-level during sparse coding
Scalability: Linear after training
```

**CNN**:
```
Hardware: GPU strongly recommended (NVIDIA with CUDA)
Memory: 8 GB GPU VRAM, 16 GB system RAM
Training: One-time (~10 hours on GPU)
Parallelization: Batch processing
Scalability: Excellent (batch inference)
```

### Software Stack

**BM3D**:
- Python package: `bm3d` (pip installable)
- Dependencies: NumPy, SciPy
- Platform: Windows/Linux/Mac

**Dictionary Learning**:
- Library: scikit-learn (KMeans), custom K-SVD
- Or: SPAMS toolbox (more efficient)
- Dependencies: NumPy, SciPy, Cython
- Platform: Windows/Linux/Mac

**CNN**:
- Framework: PyTorch or TensorFlow/Keras
- Dependencies: CUDA, cuDNN (for GPU)
- Platform: Linux preferred (better GPU support)

---

## Key Insights

### What We Learned

1. **No Single Best Method**
   - Best choice depends on constraints (time, hardware, quality needs)
   - All three methods enable successful reconstruction

2. **Domain-Specific Matters**
   - Dictionary Learning excels for periodic structures
   - Important for materials science applications
   - CNN best for diverse/complex noise

3. **Practical Trumps Theoretical**
   - BM3D's "good enough + easy" beats "perfect but complex"
   - Training overhead significant barrier for CNN/Dict Learning
   - Reproducibility favors parameter-free methods

4. **Ensemble Potential**
   - Could combine methods for different image regions
   - Use fast CNN for screening, Dict Learning for refinement
   - Hybrid approaches worth exploring

### Recommendations for Similar Projects

**If you have**:
- Limited time/resources → **BM3D**
- Periodic/structured data → **Dictionary Learning**
- Large dataset + GPU → **CNN**
- Critical quality needs → **Dictionary Learning**
- Need immediate deployment → **BM3D**

---

## Future Directions

### Potential Improvements

1. **Self-Supervised CNN**
   - Train on noisy data only (Noise2Noise approach)
   - Eliminates need for clean reference images
   - More practical for real-world data

2. **Adaptive Dictionary Learning**
   - Update dictionary online during processing
   - Adapt to specific sample characteristics
   - Potentially best of both worlds

3. **Hybrid Methods**
   - CNN for initial denoising (fast)
   - Dict Learning for refinement (quality)
   - Leverages strengths of both

4. **Physics-Informed CNN**
   - Incorporate TEM image formation model in loss
   - Could improve generalization
   - More interpretable than pure data-driven

---

## References

**BM3D**:
- Dabov et al., "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering," IEEE TIP, 2007

**Dictionary Learning**:
- Aharon et al., "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation," IEEE TSP, 2006
- Mairal et al., "Online Dictionary Learning for Sparse Coding," ICML, 2009

**CNN Denoising**:
- Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising," IEEE TIP, 2017
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015

**Our Work**:
- Zhang et al., "Revealing 3D Atomic Dynamics of Graphene via High-Speed Low-Dose TEM Imaging," (2025, in preparation)