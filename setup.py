# File: setup.py

"""
Setup configuration for HyperPath-SVM: Intelligent Network Routing Framework
"""

import os
from setuptools import setup, find_packages

# Read long description from README
def read_long_description():
    """Read the README file for long description."""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")
    
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    else:
        return "HyperPath-SVM: Intelligent Network Routing Framework using Hybrid SVMs"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    here = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(here, "requirements.txt")
    
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    else:
        # Fallback to minimal requirements
        requirements = [
            "numpy>=1.21.0",
            "scipy>=1.7.0", 
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "networkx>=2.6.0",
            "matplotlib>=3.5.0",
            "tqdm>=4.62.0"
        ]
    
    return requirements

# Package metadata
PACKAGE_NAME = "hyperpath-svm"
VERSION = "1.0.0"
AUTHOR = "HyperPath-SVM Research Team"
AUTHOR_EMAIL = "hyperpath-svm@research.org"
DESCRIPTION = "Intelligent Network Routing Framework using Hybrid SVMs"
URL = "https://github.com/hyperpath-svm/hyperpath-svm"
LICENSE = "MIT"

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10", 
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: System :: Networking",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]

# Keywords for PyPI search
KEYWORDS = [
    "machine learning", "network routing", "support vector machines", 
    "quantum computing", "graph neural networks", "network optimization",
    "intelligent routing", "svm", "quantum optimization", "temporal graphs",
    "network analysis", "routing protocols", "path optimization"
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "pytest-xdist>=2.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "isort>=5.10.0",
    "mypy>=0.900",
    "pre-commit>=2.15.0",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "nbconvert>=6.0.0",
    "jupyter>=1.0.0",
]

# Visualization dependencies
VISUALIZATION_REQUIREMENTS = [
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0", 
    "plotly>=5.0.0",
    "bokeh>=2.4.0",
    "graphviz>=0.19.0",
    "pydot>=1.4.0",
]

# Quantum computing dependencies
QUANTUM_REQUIREMENTS = [
    "qiskit>=0.39.0",
    "qiskit-aer>=0.11.0",
    "cirq>=1.0.0",
    "pennylane>=0.25.0",
]

# Deep learning dependencies  
DEEP_LEARNING_REQUIREMENTS = [
    "torch>=1.12.0",
    "torch-geometric>=2.1.0",
    "tensorflow>=2.8.0",
    "keras>=2.8.0",
]

# Big data processing dependencies
BIG_DATA_REQUIREMENTS = [
    "dask>=2022.1.0",
    "ray>=2.0.0",
    "apache-beam>=2.40.0",
    "pyarrow>=8.0.0",
]

# Performance optimization dependencies
PERFORMANCE_REQUIREMENTS = [
    "numba>=0.56.0",
    "cython>=0.29.0",
    "bottleneck>=1.3.0",
    "psutil>=5.8.0",
    "memory-profiler>=0.60.0",
]

# All optional dependencies combined
ALL_REQUIREMENTS = (
    DEV_REQUIREMENTS + 
    VISUALIZATION_REQUIREMENTS + 
    QUANTUM_REQUIREMENTS +
    DEEP_LEARNING_REQUIREMENTS +
    BIG_DATA_REQUIREMENTS + 
    PERFORMANCE_REQUIREMENTS
)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Bug Tracker": f"{URL}/issues",
        "Documentation": f"{URL}/docs",
        "Source Code": URL,
        "Research Paper": f"{URL}/paper",
        "Changelog": f"{URL}/releases",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    package_data={
        "hyperpath_svm": [
            "py.typed",
            "data/configs/*.json",
            "data/templates/*.json", 
            "utils/templates/*.txt",
        ],
    },
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    license=LICENSE,
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "test": [
            "pytest>=6.0.0", 
            "pytest-cov>=2.0.0",
            "pytest-xdist>=2.0.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0", 
            "nbconvert>=6.0.0"
        ],
        "visualization": VISUALIZATION_REQUIREMENTS,
        "quantum": QUANTUM_REQUIREMENTS,
        "deep-learning": DEEP_LEARNING_REQUIREMENTS,
        "big-data": BIG_DATA_REQUIREMENTS, 
        "performance": PERFORMANCE_REQUIREMENTS,
        "all": ALL_REQUIREMENTS,
    },
    
    # Console scripts
    entry_points={
        "console_scripts": [
            "hyperpath-train=scripts.train_hyperpath_svm:main",
            "hyperpath-evaluate=scripts.evaluate_all_methods:main",
            "hyperpath-results=scripts.generate_results:main", 
            "hyperpath-reproduce=scripts.reproduce_paper_results:main",
            "hyperpath-simulate=hyperpath_svm.experiments.production_simulation:run_production_simulation",
        ],
    },
    
    # Additional metadata
    zip_safe=False,  # Enable access to package data
    platforms=["any"],
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.0.0"
    ],
    
    # Setuptools options
    options={
        "build_py": {
            "optimize": 2,  # Enable bytecode optimization
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# Post-installation message
def _post_install_message():
    """Display post-installation message."""
    message = """
    
╔══════════════════════════════════════════════════════════════════════════════╗
║                      HyperPath-SVM Installation Complete!                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
                                               ║
║  🚀 Quick Start:                                                             ║
║                                                                              ║
║     # Train a model                                                          ║
║     hyperpath-train --config configs/default.json                           ║
║                                                                              ║
║     # Evaluate all methods                                                   ║  
║     hyperpath-evaluate --methods all --datasets caida,mawi                  ║
║                                                                              ║
║     # Generate paper results                                                 ║
║     hyperpath-results --figures --tables --export-data                      ║
║                                                                              ║
║     # Reproduce paper (one-click)                                           ║
║     hyperpath-reproduce --output-dir results/                               ║
║                                                                              ║
║  📖 Documentation: https://github.com/hyperpath-svm/hyperpath-svm/docs      ║
║  🐛 Issues: https://github.com/hyperpath-svm/hyperpath-svm/issues           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

For detailed usage examples, run: python -c "import hyperpath_svm; help(hyperpath_svm)"
    """
    print(message)

# Display post-install message when setup is run directly
if __name__ == "__main__":
    import sys
    if "install" in sys.argv:
        import atexit
        atexit.register(_post_install_message) 
