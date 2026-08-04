# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="egoemg",
    version="1.0.0",
    description="sEMG hand-pose estimation benchmark + EgoEMG vision/fusion models",
    author="Ziheng Xi",
    author_email="xige11424@gmail.com",
    packages=find_packages(),
    install_requires=[
        # Left empty so you use the conda environment.yml file
    ],
)
