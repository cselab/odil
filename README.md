# ODIL

ODIL (Optimizing a Discrete Loss) is a Python framework for solving inverse and data assimilation problems for partial differential equations.
ODIL formulates the problem through optimization of a loss function including the residuals of a finite-difference and finite-volume discretization along with data and regularization terms.
ODIL solves the same problems as PINN (Physics-Informed Neural Networks) but more efficiently.

Key features:
* automatic differentiation using TensorFlow or JAX
* optimization by gradient-based methods (Adam, L-BFGS) and Newton's method
* orders of magnitude lower computational cost than PINN [[1]](https://doi.org/10.1093/pnasnexus/pgae005)
* multigrid decomposition for faster optimization [[2]](https://doi.org/10.1140/epje/s10189-023-00313-7)

## Interactive demos

These demos use a C++ implementation of ODIL with [autodiff](https://github.com/pkarnakov/autodiff) and [Emscripten](https://emscripten.org) to run interactively in the web browser.

| [<img src="https://pkarnakov.github.io/autodiff/media/wasm_poisson.png" width=120>](https://pkarnakov.github.io/autodiff/demos/poisson.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_wave.png" width=120>](https://pkarnakov.github.io/autodiff/demos/wave.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_heat.png" width=120>](https://pkarnakov.github.io/autodiff/demos/heat.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_advection.png" width=120>](https://pkarnakov.github.io/autodiff/demos/advection.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_advection2.png" width=120>](https://pkarnakov.github.io/autodiff/demos/advection2.html) |
|:---:|:---:|:---:|:---:|:---:|
| [Poisson](https://pkarnakov.github.io/autodiff/demos/poisson.html) | [Wave](https://pkarnakov.github.io/autodiff/demos/wave.html) | [Heat](https://pkarnakov.github.io/autodiff/demos/heat.html) | [Advection](https://pkarnakov.github.io/autodiff/demos/advection.html) | [Advection2](https://pkarnakov.github.io/autodiff/demos/advection2.html) |

## Installation

```
pip install odil
```

or
```
pip install git+https://github.com/cselab/odil.git
```

### Using `uv`

```
uv venv --python 3.12
. .venv/bin/activate
uv sync --group dev --extra tensorflow --extra jax
```

## Using GPU

To enable GPU support, provide a non-empty list of devices in `CUDA_VISIBLE_DEVICES`.
Values `CUDA_VISIBLE_DEVICES=` and `CUDA_VISIBLE_DEVICES=-1` disable GPU support.

## Developers

ODIL is developed by researchers at [Harvard University](https://cse-lab.seas.harvard.edu)

* [Petr Karnakov](https://pkarnakov.com)
* [Sergey Litvinov](https://cse-lab.seas.harvard.edu/people/sergey-litvinov)

advised by

* [Prof. Petros Koumoutsakos](https://cse-lab.seas.harvard.edu/people/petros-koumoutsakos)

## Publications

1. Karnakov P, Litvinov S, Koumoutsakos P.
   Solving inverse problems in physics by optimizing a discrete loss: Fast and accurate learning without neural networks.
   PNAS Nexus, 2024.
   [DOI:10.1093/pnasnexus/pgae005](https://doi.org/10.1093/pnasnexus/pgae005)

2. Karnakov P, Litvinov S, Koumoutsakos P.
   Flow reconstruction by multiresolution optimization of a discrete loss with automatic differentiation. Eur. Phys. J, 2023.
   [DOI:10.1140/epje/s10189-023-00313-7](https://doi.org/10.1140/epje/s10189-023-00313-7)
   | [arXiv:2303.04679](https://arxiv.org/abs/2303.04679)
   | [slides](https://cselab.github.io/odil/slides/usc_workshop.pdf)

3. Balcerak M, Amiranashvili T, Wagner A, Weidner J, Karnakov P, Paetzold JC, et al.
   Physics-regularized multi-modal image assimilation for braintumor localization.
   NeurIPS, 2024.
   [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/49fb58cfd482a33619d48a5c5910cf3c-Paper-Conference.pdf)

4. Balcerak M, Weidner J, Karnakov P, Ezhov I, Litvinov S, Koumoutsakos P, et al.
   Individualizing glioma radiotherapy planning by optimization of a data and physics-informed discrete loss.
   Nature Communications, 2025.
   [DOI:10.1038/s41467-025-60366-4](https://doi.org/10.1038/s41467-025-60366-4)

5. Buhendwa B Aaron B., Koumoutsakos P. Data-driven shape inference in three-dimensional steady-state supersonic flows: Optimizing a discrete loss with JAX-fluids.
   Phys Rev Fluids, 2025.
   [DOI:10.1103/9wj9-nmr8](https://doi.org/10.1103/9wj9-nmr8)
   | [arXiv:2408.10094](https://arxiv.org/abs/2408.10094)

6. Karnakov P, Amoudruz L, Koumoutsakos P.
   Optimal navigation in microfluidics via the optimization of a discrete loss.
   Phys Rev Lett, 2025.
   [DOI:PhysRevLett.134.044001](https://link.aps.org/doi/10.1103/PhysRevLett.134.044001)
   | [arXiv:2506.15902](https://arxiv.org/abs/2506.15902)

7. Amoudruz L, Karnakov P, Koumoutsakos P. Contactless precision steering of particles in a fluid inside a cube with rotating walls.
   Journal of Fluid Mechanics, 2025.
   [DOI:10.1017/jfm.2025.10174](https://doi.org/10.1017/jfm.2025.10174)
   | [arXiv:2506.15958](https://arxiv.org/abs/2506.15958)
   | Videos
   [1](https://cselab.github.io/odil/media/odil_hydrocube_1.mp4)
   [2](https://cselab.github.io/odil/media/odil_hydrocube_2.mp4)
   [3](https://cselab.github.io/odil/media/odil_hydrocube_3.mp4)

8. Amoudruz L, Litvinov S, Papadimitriou C, Koumoutsakos P.
   Bayesian Inference for PDE-based Inverse Problems using the Optimization of a Discrete Loss.
   arXiv, 2025.
   [arXiv:2510.15664](https://arxiv.org/abs/2510.15664)
