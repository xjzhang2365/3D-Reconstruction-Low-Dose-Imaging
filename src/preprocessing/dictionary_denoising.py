"""
Dictionary Learning for Image Denoising

Sparse representation-based denoising using K-SVD algorithm.
Particularly effective for periodic/structured images like atomic lattices.

Based on:
- Aharon et al., "K-SVD Algorithm", IEEE TSP, 2006
- Applied in Zhang et al. (2025) for TEM image denoising

Key advantages for TEM imaging:
- Learns image-specific features
- Excellent for periodic atomic structures
- Adaptive to sample characteristics
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.fftpack import dct, idct
from sklearn.linear_model import OrthogonalMatchingPursuit
import warnings


class DictionaryDenoiser:
    """
    Dictionary Learning denoiser using K-SVD + OMP.
    
    This demonstrates:
    - Sparse representation theory
    - Iterative optimization (K-SVD)
    - Orthogonal Matching Pursuit
    - Adaptive feature learning
    
    Examples
    --------
    >>> # Training phase (one-time)
    >>> denoiser = DictionaryDenoiser(n_atoms=256, patch_size=8)
    >>> training_images = [load_image(f) for f in train_files]
    >>> denoiser.train(training_images)
    >>> 
    >>> # Denoising phase
    >>> clean = denoiser.denoise(noisy_image)
    
    Notes
    -----
    Training is computationally expensive but only done once.
    After training, denoising is relatively fast (~10-15 sec/image).
    """
    
    def __init__(self,
                 n_atoms: int = 256,
                 patch_size: int = 8,
                 sparsity: int = 10,
                 n_iterations: int = 20):
        """
        Initialize dictionary denoiser.
        
        Parameters
        ----------
        n_atoms : int
            Dictionary size (number of basis functions)
            Typically 4× overcomplete (256 for 8×8 patches)
        patch_size : int
            Size of image patches (8 or 16 typical)
        sparsity : int
            Maximum non-zero coefficients per patch
        n_iterations : int
            K-SVD training iterations
        """
        self.n_atoms = n_atoms
        self.patch_size = patch_size
        self.sparsity = sparsity
        self.n_iterations = n_iterations
        
        self.dictionary = None
        self.is_trained = False
        
        # Patch dimension
        self.patch_dim = patch_size * patch_size
    
    def _extract_patches(self, 
                        image: np.ndarray,
                        stride: int = 1) -> np.ndarray:
        """
        Extract overlapping patches from image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        stride : int
            Stride between patches
            
        Returns
        -------
        patches : np.ndarray
            Array of patches, shape (n_patches, patch_dim)
        """
        h, w = image.shape
        p = self.patch_size
        
        patches = []
        positions = []
        
        for i in range(0, h - p + 1, stride):
            for j in range(0, w - p + 1, stride):
                patch = image[i:i+p, j:j+p]
                patches.append(patch.flatten())
                positions.append((i, j))
        
        return np.array(patches), positions
    
    def _reconstruct_from_patches(self,
                                  patches: np.ndarray,
                                  positions: list,
                                  image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct image from overlapping patches by averaging.
        
        Parameters
        ----------
        patches : np.ndarray
            Denoised patches, shape (n_patches, patch_dim)
        positions : list
            Patch positions
        image_shape : tuple
            Original image shape
            
        Returns
        -------
        image : np.ndarray
            Reconstructed image
        """
        h, w = image_shape
        p = self.patch_size
        
        # Accumulator for weighted averaging
        reconstructed = np.zeros(image_shape)
        weights = np.zeros(image_shape)
        
        for patch, (i, j) in zip(patches, positions):
            patch_2d = patch.reshape(p, p)
            reconstructed[i:i+p, j:j+p] += patch_2d
            weights[i:i+p, j:j+p] += 1
        
        # Average where overlapping
        reconstructed /= (weights + 1e-10)
        
        return reconstructed
    
    def _initialize_dictionary(self, patches: np.ndarray) -> np.ndarray:
        """
        Initialize dictionary using random patches from training data.
        
        Parameters
        ----------
        patches : np.ndarray
            Training patches
            
        Returns
        -------
        dictionary : np.ndarray
            Initial dictionary, shape (patch_dim, n_atoms)
        """
        # Use random patches from training data
        n_patches = patches.shape[0]
        
        if n_patches < self.n_atoms:
            # Not enough patches, duplicate some
            indices = np.random.choice(n_patches, self.n_atoms, replace=True)
        else:
            # Randomly select n_atoms patches
            indices = np.random.choice(n_patches, self.n_atoms, replace=False)
        
        dictionary = patches[indices].T  # Transpose to get (patch_dim, n_atoms)
        
        # Normalize columns
        norms = np.linalg.norm(dictionary, axis=0, keepdims=True)
        dictionary = dictionary / (norms + 1e-10)
        
        return dictionary
    
    def _sparse_code_omp(self,
                        patches: np.ndarray,
                        dictionary: np.ndarray) -> np.ndarray:
        """
        Sparse coding using Orthogonal Matching Pursuit.
        
        Parameters
        ----------
        patches : np.ndarray
            Image patches, shape (n_patches, patch_dim)
        dictionary : np.ndarray
            Current dictionary, shape (patch_dim, n_atoms)
            
        Returns
        -------
        codes : np.ndarray
            Sparse codes, shape (n_patches, n_atoms)
        """
        n_patches = patches.shape[0]
        codes = np.zeros((n_patches, self.n_atoms))
        
        # Use sklearn's OMP for efficiency
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.sparsity)
        
        for i in range(n_patches):
            omp.fit(dictionary, patches[i])
            codes[i] = omp.coef_
            
            if (i + 1) % 1000 == 0:
                print(f"    Coded {i+1}/{n_patches} patches", end='\r')
        
        print()  # New line after progress
        return codes
    
    def _update_dictionary_ksvd(self,
                               patches: np.ndarray,
                               dictionary: np.ndarray,
                               codes: np.ndarray) -> np.ndarray:
        """
        Update dictionary using K-SVD algorithm.
        
        Parameters
        ----------
        patches : np.ndarray
            Training patches
        dictionary : np.ndarray
            Current dictionary
        codes : np.ndarray
            Sparse codes
            
        Returns
        -------
        dictionary : np.ndarray
            Updated dictionary
        """
        for k in range(self.n_atoms):
            # Find patches using this atom
            using_atom = np.where(codes[:, k] != 0)[0]
            
            if len(using_atom) == 0:
                # Re-initialize unused atom
                random_patch = patches[np.random.randint(len(patches))]
                dictionary[:, k] = random_patch / (np.linalg.norm(random_patch) + 1e-10)
                continue
            
            # Compute error matrix (excluding atom k)
            codes_k = codes[using_atom, k].copy()
            codes[using_atom, k] = 0
            
            error = patches[using_atom] - codes[using_atom] @ dictionary.T
            
            # SVD update
            try:
                U, s, Vt = np.linalg.svd(error, full_matrices=False)
                
                # Update dictionary atom and coefficients
                dictionary[:, k] = U[:, 0]
                codes[using_atom, k] = s[0] * Vt[0, :]
            except np.linalg.LinAlgError:
                # If SVD fails, skip this atom
                codes[using_atom, k] = codes_k
            
            if (k + 1) % 50 == 0:
                print(f"    Updated {k+1}/{self.n_atoms} atoms", end='\r')
        
        print()  # New line
        return dictionary
    
    def train(self,
             training_images: list,
             n_patches_per_image: int = 1000,
             verbose: bool = True):
        """
        Train dictionary using K-SVD algorithm.
        
        Parameters
        ----------
        training_images : list of np.ndarray
            List of clean or less-noisy images for training
        n_patches_per_image : int
            Number of patches to extract per image
        verbose : bool
            Print progress
        """
        if verbose:
            print("Training Dictionary")
            print("="*60)
            print(f"Dictionary size: {self.n_atoms} atoms")
            print(f"Patch size: {self.patch_size}×{self.patch_size}")
            print(f"Sparsity: {self.sparsity}")
            print(f"Training images: {len(training_images)}")
        
        # Extract training patches
        if verbose:
            print("\n1. Extracting training patches...")
        
        all_patches = []
        for img_idx, image in enumerate(training_images):
            # Extract patches with larger stride for efficiency
            stride = max(1, self.patch_size // 2)
            patches, _ = self._extract_patches(image, stride=stride)
            
            # Randomly select subset
            n_select = min(n_patches_per_image, len(patches))
            selected = np.random.choice(len(patches), n_select, replace=False)
            all_patches.append(patches[selected])
            
            if verbose and (img_idx + 1) % 10 == 0:
                print(f"  Processed {img_idx+1}/{len(training_images)} images")
        
        training_patches = np.vstack(all_patches)
        
        if verbose:
            print(f"  ✓ Extracted {len(training_patches)} patches")
        
        # Normalize patches
        training_patches = training_patches - np.mean(training_patches, axis=1, keepdims=True)
        
        # Initialize dictionary
        if verbose:
            print("\n2. Initializing dictionary...")
        
        self.dictionary = self._initialize_dictionary(training_patches)
        
        # K-SVD iterations
        if verbose:
            print("\n3. K-SVD optimization...")
        
        for iteration in range(self.n_iterations):
            if verbose:
                print(f"\n  Iteration {iteration+1}/{self.n_iterations}")
            
            # Sparse coding stage
            if verbose:
                print("  - Sparse coding (OMP)...")
            codes = self._sparse_code_omp(training_patches, self.dictionary)
            
            # Dictionary update stage
            if verbose:
                print("  - Updating dictionary (K-SVD)...")
            self.dictionary = self._update_dictionary_ksvd(
                training_patches, self.dictionary, codes
            )
            
            # Calculate reconstruction error
            reconstructed = codes @ self.dictionary.T
            error = np.mean(np.sum((training_patches - reconstructed)**2, axis=1))
            
            if verbose:
                print(f"  - Reconstruction MSE: {error:.4f}")
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "="*60)
            print("✓ Training complete!")
    
    def denoise(self, 
               noisy_image: np.ndarray,
               verbose: bool = False) -> np.ndarray:
        """
        Denoise image using trained dictionary.
        
        Parameters
        ----------
        noisy_image : np.ndarray
            Noisy input image
        verbose : bool
            Print progress
            
        Returns
        -------
        denoised : np.ndarray
            Denoised image
        """
        if not self.is_trained:
            raise RuntimeError("Dictionary not trained. Call train() first.")
        
        if verbose:
            print("Denoising with Dictionary Learning...")
        
        # Extract patches
        patches, positions = self._extract_patches(noisy_image, stride=1)
        
        # Normalize
        patch_means = np.mean(patches, axis=1, keepdims=True)
        patches_centered = patches - patch_means
        
        # Sparse coding
        if verbose:
            print("  Sparse coding...")
        codes = self._sparse_code_omp(patches_centered, self.dictionary)
        
        # Reconstruct
        if verbose:
            print("  Reconstructing...")
        denoised_patches = codes @ self.dictionary.T
        
        # Add back means
        denoised_patches += patch_means
        
        # Reconstruct image
        denoised = self._reconstruct_from_patches(
            denoised_patches, positions, noisy_image.shape
        )
        
        if verbose:
            print("  ✓ Done!")
        
        return denoised


if __name__ == '__main__':
    """Test dictionary learning denoiser"""
    
    print("Dictionary Learning Denoiser - Test")
    print("="*60)
    
    # Create synthetic data
    size = 128
    n_train_images = 5  # Reduced for demo
    
    print(f"\n1. Creating synthetic training data...")
    print(f"   Images: {n_train_images}")
    print(f"   Size: {size}×{size}")
    
    # Generate clean images (with structure)
    training_images = []
    for i in range(n_train_images):
        y, x = np.mgrid[0:size, 0:size]
        img = np.zeros((size, size))
        
        # Add some Gaussian blobs (simulating atomic columns)
        for _ in range(20):
            cx = np.random.uniform(10, size-10)
            cy = np.random.uniform(10, size-10)
            img += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*3**2))
        
        training_images.append(img)
    
    # Create test image
    print("\n2. Creating test image...")
    test_clean = training_images[0].copy()
    test_noisy = test_clean + np.random.normal(0, 0.1, test_clean.shape)
    
    # Train dictionary
    print("\n3. Training dictionary...")
    denoiser = DictionaryDenoiser(
        n_atoms=64,  # Reduced for demo speed
        patch_size=8,
        sparsity=5,
        n_iterations=5  # Reduced for demo
    )
    
    denoiser.train(training_images, n_patches_per_image=500, verbose=True)
    
    # Denoise
    print("\n4. Testing denoising...")
    denoised = denoiser.denoise(test_noisy, verbose=True)
    
    # Evaluate
    mse_noisy = np.mean((test_noisy - test_clean)**2)
    mse_denoised = np.mean((denoised - test_clean)**2)
    
    improvement = mse_noisy / mse_denoised
    
    print(f"\n5. Results:")
    print(f"   MSE (noisy): {mse_noisy:.6f}")
    print(f"   MSE (denoised): {mse_denoised:.6f}")
    print(f"   Improvement: {improvement:.2f}×")
    
    print("\n" + "="*60)
    print("✓ Dictionary learning test complete!")