# ODIL

ODIL (Optimizing a Discrete Loss) is a Python framework for solving inverse and data assimilation problems for partial differential equations.
ODIL formulates the problem through optimization of a loss function including the residuals of a finite-difference and finite-volume discretization
along with data and regularization terms.
ODIL solves the same problems as the popular PINN (Physics-Informed Neural Networks) framework.

Key features:
* automatic differentiation using TensorFlow or JAX
* optimization by gradient-based methods (Adam, L-BFGS) and Newton's method
* orders of magnitude lower computational cost than PINN [[1]](https://doi.org/10.1093/pnasnexus/pgae005)
* multigrid decomposition for faster optimization [[2]](https://doi.org/10.1140/epje/s10189-023-00313-7)

## Interactive demos

These demos use a C++ implementation of ODIL with [autodiff](https://github.com/pkarnakov/autodiff) and [Emscripten](https://emscripten.org) to run interactively in the web browser.

| [<img src="https://cselab.github.io/odil/media/wasm_poisson.png" width=120>](https://pkarnakov.github.io/autodiff/demos/poisson.html) | [<img src="https://cselab.github.io/odil/media/wasm_wave.png" width=120>](https://pkarnakov.github.io/autodiff/demos/wave.html) | [<img src="https://cselab.github.io/odil/media/wasm_heat.png" width=120>](https://pkarnakov.github.io/autodiff/demos/heat.html) | [<img src="https://cselab.github.io/odil/media/wasm_advection.png" width=120>](https://pkarnakov.github.io/autodiff/demos/advection.html) |
|:---:|:---:|:---:|:---:|
| [Poisson](https://pkarnakov.github.io/autodiff/demos/poisson.html) | [Wave](https://pkarnakov.github.io/autodiff/demos/wave.html) | [Heat](https://pkarnakov.github.io/autodiff/demos/heat.html) | [Advection](https://pkarnakov.github.io/autodiff/demos/advection.html) |

## Installation

```
pip install odil
```

or
```
pip install git+https://github.com/cselab/odil.git
```

## Using GPU

To enable GPU support, provide a non-empty list of devices in `CUDA_VISIBLE_DEVICES`.
Values `CUDA_VISIBLE_DEVICES=` and `CUDA_VISIBLE_DEVICES=-1` disable GPU support.

## Developers

ODIL is developed by researchers at [Harvard University](https://cse-lab.seas.harvard.edu/)

* [Petr Karnakov](https://cse-lab.seas.harvard.edu/people/petr-karnakov)
  [<img src="https://cselab.github.io/odil/media/twitter.png" height=16>](https://twitter.com/pkarnakov)
  [<img src="https://cselab.github.io/odil/media/youtube.png" height=16>](https://www.youtube.com/@pkarnakov)
* [Sergey Litvinov](https://cse-lab.seas.harvard.edu/people/sergey-litvinov)

advised by

* [Prof. Petros Koumoutsakos](https://cse-lab.seas.harvard.edu/people/petros-koumoutsakos)

## Publications

1. Karnakov P, Litvinov S, Koumoutsakos P. Solving inverse problems in physics
   by optimizing a discrete loss: Fast and accurate learning without neural networks. PNAS Nexus, 2024.
   [DOI:10.1093/pnasnexus/pgae005](https://doi.org/10.1093/pnasnexus/pgae005)
   [arXiv:2205.04611](https://arxiv.org/abs/2205.04611)


2. Karnakov P, Litvinov S, Koumoutsakos P. Flow reconstruction by
   multiresolution optimization of a discrete loss with automatic
   differentiation. Eur. Phys. J, 2023.
   [DOI:10.1140/epje/s10189-023-00313-7](https://doi.org/10.1140/epje/s10189-023-00313-7)
   [arXiv:2303.04679](https://arxiv.org/abs/2303.04679)
   [slides](https://cselab.github.io/odil/slides/usc_workshop.pdf)
