# Copyright (c) 2026 EgoEMG Authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
VERSION = (ROOT / "egoemg" / "__init__.py").read_text().split('__version__ = "')[1].split('"')[0]

setup(
    name="egoemg",
    version=VERSION,
    description="sEMG hand-pose estimation benchmark + EgoEMG vision/fusion models",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ziheng Xi",
    author_email="xige11424@gmail.com",
    url="https://github.com/zhenqis123/EgoEMG",
    project_urls={
        "Source": "https://github.com/zhenqis123/EgoEMG",
        "Issues": "https://github.com/zhenqis123/EgoEMG/issues",
    },
    license="LicenseRef-EgoEMG-Composite",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26,<2",
        "scipy>=1.10",
        "pandas>=2.0",
        "h5py>=3.8",
        "PyYAML>=6.0",
        "hydra-core>=1.3,<1.4",
        "pytorch-lightning>=2.5,<2.6",
        "torch>=2.2,<2.3",
        "torchvision>=0.17,<0.18",
        "torchmetrics>=1.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "train": ["h5py>=3.8", "scipy>=1.10"],
        "vision": ["opencv-python>=4.8", "decord>=0.6", "lmdb>=1.4", "Pillow>=10"],
        "viz": [
            "opencv-python>=4.8", "decord>=0.6", "lmdb>=1.4", "Pillow>=10",
            "matplotlib>=3.7", "plotly>=5.18", "smplx>=0.1.28", "pyrender>=0.1.45",
            "trimesh>=4.0",
        ],
        "realtime": ["pyarrow>=14", "pyserial>=3.5", "pyzmq>=25"],
        "keystroke": ["python-Levenshtein>=0.23"],
        "dev": [
            "pytest>=8", "build>=1.2", "flake8>=7", "mypy>=1.10",
            "plotly>=5.18",
        ],
    },
    package_data={"egoemg.UmeTrack": ["dataset/*.json"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
