# Quick Start Guide

Get up and running with 3D reconstruction in 5 minutes.

---

## Installation (2 minutes)
```bash
# Clone repository
git clone https://github.com/yourusername/3D-Reconstruction-Low-Dose-Imaging.git
cd 3D-Reconstruction-Low-Dose-Imaging

# Create environment
conda create -n reconstruction python=3.10
conda activate reconstruction

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Verify Installation (1 minute)
```bash
# Run tests
python -m pytest tests/ -v

# Should see: All tests passed ✓
```

---

## Your First Reconstruction (2 minutes)

### Option 1: Run Example Script
```bash
cd examples
python basic_usage.py
```

This runs the complete pipeline and generates example outputs.

### Option 2: Interactive Python
```python
from src.preprocessing import PreprocessingPipeline
from src.estimation import GaussianFitter
from src.analysis import StructureAnalyzer
from src.visualization import StructureVisualizer

# 1. Load your images (replace with your data)
import numpy as np
images = [np.load(f'data/frame_{i:03d}.npy') for i in range(10)]

# 2. Preprocess
pipeline = PreprocessingPipeline(window_size=5)
cleaned = pipeline.process(images, target_idx=5)

# 3. Estimate structure
fitter = GaussianFitter()
results = fitter.fit_image(cleaned)
positions_3d = results['positions_3d']

# 4. Analyze
analyzer = StructureAnalyzer()
analysis = analyzer.analyze_full(positions_3d)
print(f"Bond length: {analysis['bonds']['mean']:.3f} Å")

# 5. Visualize
viz = StructureVisualizer()
viz.plot_3d(positions_3d, save_path='my_structure.png')
```

---

## Explore Examples

### Jupyter Notebooks
```bash
jupyter notebook
```

Then open:
- `notebooks/demos/preprocessing_demo.ipynb` - Preprocessing pipeline
- `notebooks/demos/estimation_demo.ipynb` - Structure estimation
- `notebooks/demos/analysis_demo.ipynb` - Structural analysis
- `notebooks/results_showcase/video_showcase.ipynb` - Research results

### Example Scripts
```bash
cd examples
python basic_usage.py 1  # Preprocessing only
python basic_usage.py 2  # Estimation only
python basic_usage.py 3  # Analysis only
python basic_usage.py 4  # Visualization only
python basic_usage.py 5  # Complete workflow (default)
```

---

## Common Tasks

### Denoise a Single Image
```python
from src.preprocessing import BM3DDenoiser

denoiser = BM3DDenoiser()
clean = denoiser.denoise(noisy_image)
```

### Compare Denoising Methods
```python
from src.preprocessing import BM3DDenoiser, DictionaryDenoiser

# BM3D
bm3d = BM3DDenoiser()
result_bm3d = bm3d.denoise(image)

# Dictionary Learning
dict_denoiser = DictionaryDenoiser()
dict_denoiser.train(training_images)
result_dict = dict_denoiser.denoise(image)
```

### Detect Dead Pixels
```python
from src.preprocessing import DeadPixelDetector

detector = DeadPixelDetector()
dead_map = detector.detect_from_stack(image_stack)
corrected = detector.correct_image(image, dead_map)
```

### Analyze Structure
```python
from src.analysis import StructureAnalyzer

analyzer = StructureAnalyzer()

# Full analysis
results = analyzer.analyze_full(structure_3d)

# Or individual metrics
bonds = analyzer.calculate_bond_lengths(structure_3d)
strain = analyzer.calculate_strain(structure_3d)
heights = analyzer.calculate_height_statistics(structure_3d)
```

---

## Next Steps

**Learn More:**
- Read [Methodology Overview](docs/methodology_overview.md)
- Explore [Preprocessing Pipeline](docs/preprocessing_pipeline.md)
- Compare [Denoising Methods](docs/denoising_comparison.md)
- Understand [Medical Applications](docs/medical_imaging_applications.md)

**Customize:**
- Modify preprocessing parameters in `src/preprocessing/`
- Add new analysis metrics in `src/analysis/`
- Create custom visualizations in `src/visualization/`

**Contribute:**
- Report issues on GitHub
- Submit pull requests
- Share your use cases

---

## Need Help?

**Documentation:** See `docs/` folder  
**Examples:** See `examples/` and `notebooks/`  
**Tests:** See `tests/` for usage patterns  
**Contact:** xzhang2365@gmail.com

---

## Common Issues

**Issue:** `ModuleNotFoundError: No module named 'bm3d'`  
**Solution:** `pip install bm3d`

**Issue:** `ImportError: No module named 'src'`  
**Solution:** Run `pip install -e .` from repository root

**Issue:** Tests fail  
**Solution:** Check Python version (need 3.10+), reinstall dependencies

**Issue:** Out of memory  
**Solution:** Reduce image size or use chunked processing

---

Happy reconstructing! 🚀