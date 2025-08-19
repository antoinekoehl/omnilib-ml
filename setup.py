from setuptools import setup, find_packages

setup(
    name="nbfitnessml",
    version="0.1",
    packages=['nabstab'] + find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.2.4',
        'pandas>=1.5.3',
        'scikit-learn>=1.2.2', 
        'matplotlib',
        'seaborn',
        'tqdm',
        'biopython',
        'antpack==0.3.8.6'
    ],
    extras_require={
        'dev': [
            'notebook',  # Jupyter notebook is typically a development dependency
        ],
    },
)