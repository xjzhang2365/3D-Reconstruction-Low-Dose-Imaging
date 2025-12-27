"""
Setup script for 3D Reconstruction from Low-Dose Imaging
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="reconstruction-low-dose",
    version="1.0.0",
    author="Xiaojun Zhang",
    author_email="xzhang2365@gmail.com",
    description="Physics-informed 3D reconstruction from low-dose 2D images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/3D-Reconstruction-Low-Dose-Imaging",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.20.0",
        "scikit-learn>=1.3.0",
        "bm3d>=4.0.0",
        "opencv-python>=4.8.0",
        "plotly>=5.14.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.1.0",
        ],
        "deep-learning": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reconstruct=examples.basic_usage:main",
        ],
    },
)