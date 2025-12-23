"""
Setup directory structure for reconstruction project
Run once to create all necessary folders and files
"""

from pathlib import Path

print("Creating project structure...")
print("=" * 60)

# Define all directories
directories = [
    # Source code
    "src/preprocessing",
    "src/estimation", 
    "src/optimization",
    "src/validation",
    "src/analysis",
    "src/visualization",
    "src/utils",
    
    # Results from research
    "results/figures",
    "results/data/reconstructed_structures",
    "results/data/metrics",
    "results/data/visualizations",
    "results/precomputed",
    
    # Documentation
    "docs/images",
    "docs/methodology",
    
    # Jupyter notebooks
    "notebooks/demos",
    "notebooks/results_showcase",
    
    # Examples and tests
    "examples",
    "tests",
]

# Create all directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created: {directory}")

print("\nCreating Python package files...")

# Create __init__.py files to make Python packages
init_files = [
    "src/__init__.py",
    "src/preprocessing/__init__.py",
    "src/estimation/__init__.py",
    "src/optimization/__init__.py",
    "src/validation/__init__.py",
    "src/analysis/__init__.py",
    "src/visualization/__init__.py",
    "src/utils/__init__.py",
]

for init_file in init_files:
    Path(init_file).touch()
    print(f"  ✓ Created: {init_file}")

print("\n" + "=" * 60)
print("✓ Project structure created successfully!")
print("\nYour project is now organized and ready for development.")