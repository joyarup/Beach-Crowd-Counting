# setup.py
from setuptools import setup, find_packages

setup(
    name="beach_monitoring",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy',
        'matplotlib',
        'pandas',
        'PyYAML',
    ],
)