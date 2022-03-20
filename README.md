# Active Sampling for Atomistic Potentials

This codebase introduces an active learning algorithm to dynamically compose a training set during training of a neural network potential (NNP) to prevent over-learning of a specific region of chemical space. The scheme facilitates the further use of data generated from computational studies, which tend to unevenly cover regions of chemical space and are not well-suited to use directly as training sets for NNPs.

Our active learning scheme is applied to the well-known [SchNet NNP](https://github.com/atomistic-machine-learning/schnetpack), with a specific focus on shear processes in materials systems containing several hundred atoms. To accomodate such large systems, several algorithmic enhancements were required, as detailed in the _SchNetPack Expansion Features_ section.

We also demonstrate the use of the actively trained NNP to examine vacancy migration under shear in mixed metal systems. Code is provided for the implementation of the NNP into the [ASE structure optimization](https://wiki.fysik.dtu.dk/ase/ase/optimize.html) module for geometry relaxation, along with the algorithm to simulate shear. We also provide code for the analysis of vacancy position based on [GALAS](https://github.com/pnnl/galas) and the generation and analysis of multiple evolutionary pathways the material can experience under shear.

## Requirements:
- python 3
- ASE
- numpy
- PyTorch (>=2.5.2)
- h5py
- networkx
- Optional: tensorboardX

_**Note: We recommend using a GPU for training neural network potentials.**_

## SchNetPack Expansion Features
- Active learning scheme to dynamically compose the training set during training
- [Automatic Mixed Precision (AMP)](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) for faster and more memory efficient training on NVIDIA GPUs
- Nearest neighbor preprocessing for improved computational efficiency
- Neighbor distance computations for systems with periodic boundary conditions using the [Minimum Image Convention in Non-Cubic MD Cells](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696) to improve speed and reduce memory requirements

## Installation
All code was tested using python 3.9 and cuda 11.0.

#### Create new conda environment with requirements

```
conda create --name active_schnet --file spec-file.txt
```

#### Activate conda environment

```
conda activate active_schnet
```

#### Install SchNetPack with Expansion

```
python setup.py install
```

#### Install Ovito (for post-shear analysis)

```
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito
```



## Documentation

For the original SchNetPack API reference, visit the [documentation](https://schnetpack.readthedocs.io).
