from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "3D structure reconstruction from low-dose 2D images"

setup(
    name="structure-reconstruction",
    version="1.0.0",
    author="Xiaojun Zhang",
    author_email="xzhang2365@gmail.com",
    description="Physics-informed 3D reconstruction from low-dose 2D images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xjzhang2365/3D-Reconstruction-Low-Dose-Imaging",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-image>=0.21.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "bm3d>=4.0.0",
        "lmfit>=1.2.0",
        "plotly>=5.17.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="electron microscopy, image reconstruction, medical imaging, low-dose imaging",
)