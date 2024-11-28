# setup.py
from setuptools import setup, find_packages
import os
import re

def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'my_graphgps', '__init__.py')
    with open(version_file, 'r') as vf:
        version_content = vf.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="my_graphgps_package",
    version=read_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for graph-based GPS modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_graphgps_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=1.7.0",
        "torch-geometric",
        "scanpy",
        "networkx",
        "matplotlib",
        "seaborn",
        "umap-learn",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "pygsp",
        "python-igraph",
        "python-louvain",
    ],
    entry_points={
        'console_scripts': [
            'my-graphgps=my_graphgps.main:main',
        ],
    },
    include_package_data=True,
    license="MIT",
)

# Notes for testing:
# >>> import my_graphgps
# >>> from my_graphgps.data import import_pbmc

# >>> adata, sct_data = import_pbmc.create_sc_data()
# >>> exit()