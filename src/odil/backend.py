'''
Wrappers for backend modules, e.g. TensorFlow, JAX, NumPy, CuPy.
The interface tends to follow NumPy's conventions.
'''

import numpy as np
from argparse import Namespace
from functools import partial


class ModBase:

    def __init__(self, mod=None):
        assert mod is not None
        self.mod = mod
        for name in [
                'int32',
                'float32',
                'float64',
                'linspace',
                'ones',
                'ones_like',
                'roll',
                'reshape',
                'stack',
                'abs',
                'cos',
                'sin',
                'exp',
                'zeros',
                'zeros_like',
                'square',
                'sqrt',
                'transpose',
                'minimum',
                'maximum',
                'meshgrid',
                'where',
        ]:
            setattr(self, name, getattr(mod, name))
        self.flatten = lambda x: mod.reshape(x, [-1])


class ModNumpy(ModBase):

    def __init__(self, mod=None, modsp=None, jax=None):
        mod = mod or np
        super().__init__(mod)
        self.cast = lambda x, dtype: mod.array(x, dtype=dtype)
        self.numpy = mod.array
        self.native = mod.array
        self.spnative = lambda x: x
        self.full = mod.full
        self.sum = mod.sum
        self.mean = mod.mean
        self.cumsum = mod.cumsum
        self.std = mod.std
        self.median = mod.median
        self.min = mod.min
        self.max = mod.max
        self.log = mod.log
        self.tanh = mod.tanh
        self.relu = lambda x: mod.maximum(x, 0)
        self.arctan2 = mod.arctan2
        self.sigmoid = lambda x: 1 / (1 + mod.exp(-x))
        self.arange = mod.arange
        self.norm = mod.linalg.norm
        self.solve = mod.linalg.solve
        self.mod = mod
        self.modsp = modsp
        self.moveaxis = mod.moveaxis
        self.hstack = mod.hstack
        self.concatenate = mod.concatenate
        self.ndarray = mod.ndarray
        self.constant = mod.array
        self.is_tensor = lambda x: isinstance(x, mod.ndarray)
        self.clip = mod.clip
        self.floor = mod.floor
        self.copy = mod.copy
        self.gather_nd = lambda u, idx: u[tuple(mod.moveaxis(idx, -1, 0))]
        self.pad = mod.pad
        self.einsum = mod.einsum
        self.batch_to_space = None
        self.jax = jax
        self.tf = None
        self.array = mod.array
        self.matmul = mod.matmul
        self.variable = mod.array
        if jax:

            def split_by_sizes(array, sizes, axis=0):
                cumsum = np.cumsum(sizes)[:-1]
                return jax.numpy.split(array, cumsum, axis=axis)
        else:

            def split_by_sizes(array, sizes, axis=0):
                cumsum = np.cumsum(sizes)[:-1]
                return mod.split(array, cumsum, axis=axis)

        self.split_by_sizes = split_by_sizes

        if jax:
            self.stop_gradient = jax.lax.stop_gradient
            self.broadcast_to = jax.numpy.broadcast_to
            self.jit_wrap = lambda **kwargs: partial(jax.jit, **kwargs)
        else:
            self.broadcast_to = mod.broadcast_to

        def convolution(input, filters, strides, padding):
            '''
            input: n-dimensional input array
            filters: n-dimiensional kernel
            strides: `int` or sequence of n integers
            '''
            if isinstance(strides, int):
                strides = (strides, ) * len(input.shape)
            input = mod.reshape(input, (1, 1) + input.shape)
            filters = mod.reshape(filters, (1, 1) + filters.shape)
            res = jax.lax.conv(lhs=input,
                               rhs=filters,
                               window_strides=strides,
                               padding=padding)
            res = res[0, 0, ...]
            return res

        self.convolution = convolution

        # Random.
        self.random = Namespace()
        if jax:
            self.random.key = None

            def set_seed(seed):
                self.random.key = jax.random.PRNGKey(seed)

            def uniform(shape, minval, maxval, dtype):
                if self.random.key is None:
                    set_seed(np.random.default_rng().integers(1 << 16))
                self.random.key, subkey = jax.random.split(self.random.key)
                return jax.random.uniform(subkey,
                                          shape=shape,
                                          minval=minval,
                                          maxval=maxval,
                                          dtype=dtype)

            def normal(shape, mean=0, stddev=1, dtype=None):
                if self.random.key is None:
                    set_seed(np.random.default_rng().integers(1 << 16))
                self.random.key, subkey = jax.random.split(self.random.key)
                mean = self.cast(mean, dtype)
                stddev = self.cast(stddev, dtype)
                return mean + stddev * jax.random.normal(
                    subkey, shape=shape, dtype=dtype)
        else:

            def set_seed(seed):
                self.random.set_seed(seed)

            def uniform(shape, minval, maxval, dtype):
                return mod.random.uniform(low=minval, high=maxval,
                                          size=shape).astype(dtype)

            def normal(shape, mean, stddev, dtype):
                return mod.random.normal(loc=mean, scale=stddev,
                                         size=shape).astype(dtype)

        self.random.set_seed = set_seed
        self.random.uniform = uniform
        self.random.normal = normal

        def conv_transpose(input,
                           filters,
                           output_shape=None,
                           strides=None,
                           padding=None):
            dim = len(input.shape)
            if isinstance(strides, int):
                strides = (strides, ) * dim
            res = jax.lax.conv_transpose(lhs=input,
                                         rhs=filters,
                                         strides=strides,
                                         padding=padding)
            return res

        self.conv_transpose = conv_transpose
        if modsp is not None:
            self.csr_matrix = modsp.csr_matrix
            self.diags = modsp.diags
            self.bmat = modsp.bmat
            self.block_diag = modsp.block_diag
            self.tril = modsp.tril
            tmp = self.numpy
            self.numpy = (lambda x: x
                          if isinstance(x, modsp.csr_matrix) else tmp(x))
            self.spnorm = modsp.linalg.norm
            self.spsolve = modsp.linalg.spsolve


class ModTensorflow(ModBase):

    def __init__(self, mod=None, modsp=None):
        super().__init__(mod)
        self.batch_to_space = mod.batch_to_space
        self.cast = mod.cast
        self.numpy = lambda x: x.numpy()
        self.native = lambda x: x
        self.spnative = lambda x: x.numpy() if hasattr(x, 'numpy') else x
        self.sum = mod.reduce_sum
        self.mean = mod.reduce_mean
        self.max = mod.reduce_max
        self.min = mod.reduce_min
        self.std = mod.math.reduce_std
        self.log = mod.math.log
        self.cumsum = mod.math.cumsum
        self.tanh = mod.math.tanh
        self.relu = mod.nn.relu
        self.sigmoid = mod.math.sigmoid
        self.full = mod.fill
        self.arange = mod.range
        self.norm = lambda x: mod.reduce_sum(x**2)**0.5
        self.mod = mod
        self.modsp = modsp
        self.floor = mod.math.floor
        self.arctan2 = mod.math.atan2
        self.concatenate = mod.concat
        self.constant = mod.constant
        self.is_tensor = mod.is_tensor
        self.clip = mod.clip_by_value
        self.copy = mod.identity
        self.gather_nd = mod.gather_nd
        self.batch_to_space = mod.batch_to_space
        self.conv_transpose = mod.nn.conv_transpose
        self.random = mod.random
        self.split_by_sizes = mod.split
        self.einsum = None
        self.array = mod.constant
        self.matmul = mod.linalg.matmul
        self.variable = mod.Variable
        self.tf = mod
        self.stop_gradient = mod.stop_gradient
        self.broadcast_to = mod.broadcast_to

        def jit_wrap(static_argnames=None, **kwargs):
            return partial(mod.function, **kwargs)

        self.jit_wrap = jit_wrap
        self.jax = None

        def pad(array, pad_width, mode):
            return mod.pad(array, paddings=pad_width, mode=mode)

        self.pad = pad

        def convolution(input, filters, strides, padding):
            '''
            input: n-dimensional input array
            filters: n-dimiensional kernel
            strides: `int` or sequence of n integers
            '''
            input = mod.reshape(input, (1, ) + input.shape + (1, ))
            filters = mod.reshape(filters, filters.shape + (1, 1))
            res = mod.nn.convolution(input=input,
                                     filters=filters,
                                     strides=strides,
                                     padding=padding)
            res = res[0, ..., 0]
            return res

        self.convolution = convolution

        # Random.
        self.random = Namespace()

        def set_seed(seed):
            mod.random.set_seed(seed)

        self.random.set_seed = set_seed
        self.random.uniform = mod.random.uniform
        self.random.normal = mod.random.normal

        if modsp is not None:

            def csr_matrix(*args, **kwargs):
                if 'dtype' in kwargs:
                    kwargs['dtype'] = {
                        mod.float32: np.float32,
                        mod.float64: np.float64
                    }[kwargs['dtype']]
                return modsp.csr_matrix(*args, **kwargs)

            self.csr_matrix = csr_matrix
            self.diags = modsp.diags
            self.bmat = modsp.bmat
            self.block_diag = modsp.block_diag
            self.tril = modsp.tril
            tmp = self.numpy
            self.numpy = (lambda x: x
                          if isinstance(x, modsp.csr_matrix) else tmp(x))
            self.spnorm = modsp.linalg.norm
            self.spsolve = modsp.linalg.spsolve


class ModCupy(ModBase):

    def __init__(self, mod=None, modsp=None):
        super().__init__(mod)
        self.cast = lambda x, dtype: mod.array(x, dtype=dtype)
        self.numpy = lambda x: x.get() if isinstance(x, mod.ndarray) else x
        self.full = mod.full
        self.sum = mod.sum
        self.native = mod.array
        self.spnative = lambda x: x
        self.arange = mod.arange
        self.norm = mod.linalg.norm
        self.solve = mod.linalg.solve
        self.mod = mod
        self.modsp = modsp
        self.moveaxis = mod.moveaxis
        self.hstack = mod.hstack
        if modsp is not None:
            self.csr_matrix = modsp.csr_matrix
            self.diags = modsp.diags
            self.bmat = modsp.bmat

            def block_diag(bb):
                n = len(bb)
                res = [[None for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        res[i][i] = bb[i]
                return modsp.bmat(res, format='csr')

            self.block_diag = block_diag
            self.tril = modsp.tril
            tmp = self.numpy
            self.numpy = (lambda x: x.get()
                          if isinstance(x, modsp.csr_matrix) else tmp(x))
            self.spnorm = modsp.linalg.norm
            self.spsolve = modsp.linalg.spsolve
