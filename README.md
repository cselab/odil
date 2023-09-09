# ODIL

ODIL (Optimizing a Discrete Loss) is a Python framework for solving inverse and data assimilation problems for partial differential equations.
ODIL formulates the problem through optimization of a loss function including the residuals of a finite-difference and finite-volume discretization
along with data and regularization terms.
ODIL solves the same problems as the popular PINN (Physics-Informed Neural Networks) framework.

Key features:
* automatic differentiation using TensorFlow or JAX
* optimization by gradient-based methods (Adam, L-BFGS) and Newton's method
* orders of magnitude lower computational cost than PINN [[1]](https://arxiv.org/abs/2205.04611)
* multigrid decomposition for faster optimization [[2]](https://doi.org/10.1140/epje/s10189-023-00313-7)
* examples and benchmarks [[slides]](https://cselab.github.io/odil/slides/usc_workshop.pdf)

## Installation

```
pip install --editable .
```

## Using GPU

To enable GPU support, provide a non-empty list of devices in `CUDA_VISIBLE_DEVICES`.
Values `CUDA_VISIBLE_DEVICES=` and `CUDA_VISIBLE_DEVICES=-1` disable GPU support.

## Publications

1. Karnakov P, Litvinov S, Koumoutsakos P. Optimizing a DIscrete Loss
   (ODIL) to solve forward and inverse problems for partial
   differential equations using machine learning tools.
   _arXiv preprint_. 2022
   [arXiv:2205.04611](https://arxiv.org/abs/2205.04611)

2. Karnakov P, Litvinov S, Koumoutsakos P. Flow reconstruction by
   multiresolution optimization of a discrete loss with automatic
   differentiation. Eur. Phys. J. 2023
   [DOI:10.1140/epje/s10189-023-00313-7](https://doi.org/10.1140/epje/s10189-023-00313-7)
   [arXiv:2303.04679](https://arxiv.org/abs/2303.04679)
