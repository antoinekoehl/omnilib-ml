from setuptools import setup, find_packages

setup(
    name="nbfitnessml",
    version="0.1",
    packages=['nabstab'] + find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.2.4',
        'pandas>=1.5.3',
        'scikit-learn', 
        'matplotlib',
        'seaborn',
        'tqdm',
        'biopython',
        'torch>=2.2.0',
        'antpack'
    ],
    extras_require={
        'dev': [
            'notebook',  # Jupyter notebook is typically a development dependency
        ],
        'gpu': [
            'torch==2.2.0',  # Specific CUDA version
        ],
    },
)

# To install the package with basic PyTorch:
# pip install -e .

# To install with CUDA-enabled PyTorch:
# pip install -e .[gpu] --extra-index-url https://download.pytorch.org/whl/cu118

# To install with development dependencies:
# pip install -e .[dev]

# To install with both CUDA support and development dependencies:
# pip install -e .[dev,gpu] --extra-index-url https://download.pytorch.org/whl/cu118
