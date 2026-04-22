#!/usr/bin/env python3
"""
Setup script for JNNX - JAGS Neural Network eXtension
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jnnx",
    version="1.0.1",
    author="Joachim Vandekerckhove",
    author_email="joachim@uci.edu",
    description="JAGS Neural Network eXtension - Evaluate ONNX models as deterministic nodes in JAGS",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/joachimvandekerckhove/jnnx",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "jnnx-setup=jnnx.scripts.jnnx_setup:main",
            "validate-jnnx=jnnx.scripts.validate_jnnx:main",
            "generate-module=jnnx.scripts.generate_module:main",
            "validate-module=jnnx.scripts.validate_module:main",
        ],
    },
    include_package_data=True,
    package_data={
        "jnnx": [
            "templates/*.template",
            "models/*.jnnx/*",
        ],
    },
    zip_safe=False,
)
