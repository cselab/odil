import numpy as np
import pickle

from .util import assert_equal, printlog
from .backend import ModTensorflow
from . import core_min


class Domain:

    def __init__(
        self,
        cshape,
        dimnames=None,
        lower=0.,
        upper=1.,
        dtype=None,
        multigrid=False,
        mg_convert_all=True,
        mg_nlvl=None,
        mg_factors=None,
        mg_axes=None,
        mg_interp=None,
        mod=None,
    ):
        '''
        cshape: `tuple`
            Grid size measured in cells.
        multigrid: `bool`
            Generate multigrid hierarchy.
        mg_convert_all: `bool`
            Convert all fields to multigrid fields in `init_state()`.
        mg_nlvl: `int`
            Number of multigrid levels. Defaults to maximum possible.
        mg_factors: `list` of `int`
            Factors of each level. Defaults to ones.
        mg_axes: `list` of `bool`
            Axes in which to use multigrid decomposition. Defaults to all.
        mg_interp: `str`
            Multigrid interpolation method. See `interp_to_finer()`.
        '''
        ndim = len(cshape)
        dimnames = dimnames or ['x', 'y', 'z'][:ndim]
        if mod is None:
            from .runtime import mod
        assert_equal(len(dimnames), ndim, f'with dimnames={dimnames}')
        self.ndim = ndim
        self.cshape = cshape
        self.dimnames = dimnames
        if dtype is None:
            from . import runtime
            dtype = runtime.dtype
        self.dtype = dtype
        self.lower = (np.ones(ndim, dtype=dtype) * lower).astype(dtype)
        self.upper = (np.ones(ndim, dtype=dtype) * upper).astype(dtype)
        self.mod = mod

        # Multigrid decomposition
        self.multigrid = multigrid
        if multigrid:
            self.mg_factors = mg_factors
            mg_axes = mg_axes or [True] * ndim
            nlvl_max = min(
                round(np.log2(n)) if ax else max(cshape)
                for n, ax in zip(cshape, mg_axes))
            if mg_nlvl is not None:
                assert mg_nlvl >= 1
                mg_nlvl = min(mg_nlvl, nlvl_max)
            else:
                mg_nlvl = nlvl_max
            self.mg_nlvl = mg_nlvl
            self.mg_cshapes = [
                tuple(n >> lvl if ax else n for n, ax in zip(cshape, mg_axes))
                for lvl in range(mg_nlvl)
            ]
            check_multigrid_cshapes(self.mg_cshapes, mg_axes)
            self.mg_axes = mg_axes
            self.mg_interp = mg_interp
            self.mg_convert_all = mg_convert_all

    @staticmethod
    def _names_to_indices(dims, dimnames):
        '''
        Converts a list of direction names to indices.
        Examples with `dimnames = ['t', 'x', 'y']`:
            complete_dims(['x']) -> [1]
            complete_dims(['x', 'y']) -> [1, 2]
            complete_dims([]) -> [0, 1, 2]
        '''
        res = dims if dims is not None and len(dims) else range(len(dimnames))
        res = tuple(
            dimnames.index(i) if isinstance(i, str) else i for i in res)
        return res

    def cast(self, value, dtype=None):
        dtype = dtype or self.dtype
        return self.mod.cast(value, dtype)

    def get_minimal(self):
        return core_min.Domain(self)

    def _points_cell_1d(self, d):
        x = np.linspace(self.lower[d],
                        self.upper[d],
                        self.cshape[d],
                        endpoint=False,
                        dtype=self.dtype)
        if len(x) > 1:
            x += (x[1] - x[0]) * 0.5
        return x

    def _points_node_1d(self, d):
        x = np.linspace(self.lower[d],
                        self.upper[d],
                        self.cshape[d] + 1,
                        dtype=self.dtype)
        return x

    def _points_1d(self, d, loc):
        if loc == 'c':
            return self._points_cell_1d(d)
        elif loc == 'n':
            return self._points_node_1d(d)
        else:
            raise ValueError("Unknown loc=" + loc)

    def points_1d(self, *dims, loc=None):
        loc = loc or 'c' * self.ndim
        idims = self._names_to_indices(dims, self.dimnames)
        res = [self._points_1d(idim, c) for idim, c in zip(idims, loc)]
        if len(dims) == 1:
            return res[0]
        return res

    def points(self, *dims, loc=None):
        loc = loc or 'c' * self.ndim
        assert_equal(len(loc), self.ndim, f"with loc={loc}")
        dimnames = [v for v, c in zip(self.dimnames, loc) if c != '.']
        idims = self._names_to_indices(dims, dimnames)
        xx = [
            self._points_1d(d, loc[d]) if loc[d] != '.' else None
            for d in range(self.ndim)
        ]
        xx = [x for x in xx if x is not None]
        data = self.mod.meshgrid(*xx, indexing='ij')
        res = tuple(data[i] for i in idims)
        if len(dims) == 1:
            return res[0]
        return res

    def _indices_cell_1d(self, d):
        return np.arange(self.cshape[d], dtype=int)

    def _indices_node_1d(self, d):
        return np.arange(self.cshape[d] + 1, dtype=int)

    def _indices_1d(self, d, loc):
        if loc == 'c':
            return self._indices_cell_1d(d)
        elif loc == 'n':
            return self._indices_node_1d(d)
        else:
            raise ValueError("Unknown loc=" + loc)

    def indices(self, *dims, loc=None):
        loc = loc or 'c' * self.ndim
        dimnames = [v for v, c in zip(self.dimnames, loc) if c in 'cn']
        idims = self._names_to_indices(dims, dimnames)
        xx = [
            self._indices_1d(d, loc[d]) if loc[d] in 'cn' else None
            for d in range(self.ndim)
        ]
        xx = [x for x in xx if x is not None]
        data = self.mod.meshgrid(*xx, indexing='ij')
        res = tuple(data[i] for i in idims)
        if len(dims) == 1:
            return res[0]
        return res

    @staticmethod
    def _get_field_shape(cshape, loc=None):
        '''
        Returns the size of array with given location of value.

        cshape: `tuple`
            Domain shape measured in cells.
        loc: `str`
            Location of value in cell.
            One character per direction: 'c' for center and 'n' for node.
        '''
        loc = loc or 'c' * len(cshape)
        assert all(c in 'cn' for c in loc)
        return tuple(s + 1 if c == 'n' else s for s, c in zip(cshape, loc))

    def get_field_shape(self, loc=None):
        return _get_field_shape(self.cshape, loc=loc)

    def _size_1d(self, d, loc):
        if loc == 'c':
            return self.cshape[d]
        elif loc == 'n':
            return self.cshape[d] + 1
        else:
            raise ValueError("Unknown loc=" + loc)

    def size(self, *dims, loc=None):
        loc = loc or 'c' * self.ndim
        assert_equal(len(loc), self.ndim, f"with loc={loc}")
        idims = self._names_to_indices(dims, self.dimnames)
        res = [self._size_1d(i, loc[i]) for i in idims]
        if len(dims) == 1:
            return res[0]
        return res

    def step_by_dim(self, i):
        return (self.upper[i] - self.lower[i]) / self.cshape[i]

    def step(self, *dims):
        '''
        Returns grid spacing in directions `dims`.
        Examples:
            dx = ctx.step('x')
            dx, = ctx.step() # With dimnames=['x'].
            dx, dy = ctx.step('x', 'y')
            dx, dy = ctx.step() # With dimnames=['x', 'y'].
        '''
        idims = self._names_to_indices(dims, self.dimnames)
        res = tuple(self.step_by_dim(i) for i in idims)
        if len(dims) == 1:
            return res[0]
        return res

    def random_inner(self, size):
        res = latin_hypercube(self.ndim, size, dtype=self.dtype).T
        for i in range(self.ndim):
            res[i] = self.lower[i] + (self.upper[i] - self.lower[i]) * res[i]
        res = [p for p in res]
        return res

    def random_boundary(self, normal, side, size):
        '''
        Returns random points from boundary (domain face).
        normal: `int`
            Direction of normal to boundary, [0, ndim)
        side: `int`
            Side of boundary (0 or 1).
        size: `int`
            Number of samples.
        '''
        assert normal < self.ndim
        assert side == 0 or side == 1
        dtype = self.dtype
        res = latin_hypercube(self.ndim - 1, size, dtype=dtype).T
        const = np.ones(size, dtype=dtype) * side
        res = np.vstack((res[:normal], const, res[normal:]))
        for i in range(self.ndim):
            res[i] = self.lower[i] + (self.upper[i] - self.lower[i]) * res[i]
        res = [p for p in res]
        return res

    def multigrid_to_regular(self, mgfield):
        '''
        Converts multigrid components to regular field.
        uw: `ndarray`
            Multigrid components on levels `self.mg_cshapes`.
        '''
        mod = self.mod
        factors = mgfield.factors or self.mg_factors or [1] * len(cshapes)
        axes = mgfield.axes or self.mg_axes
        shapes = [
            self._get_field_shape(term.cshape, loc=mgfield.loc)
            for term in mgfield.terms
        ]
        assert_equal(len(factors), len(mgfield.terms))
        assert_equal(len(axes), len(mgfield.terms[0].cshape))
        method = mgfield.method or self.mg_interp
        arrays = [term.array * f for term, f in zip(mgfield.terms, factors)]
        loc = ''.join(l if ax else '.' for l, ax in zip(mgfield.loc, axes))
        res = arrays[-1]
        for array in reversed(arrays[:-1]):
            res = array + interp_to_finer(res, loc, method, mod)
        return Field(res, loc=mgfield.loc)

    def get_regular_array(self, field):
        '''
        Returns array from Field, MultigridField, or Array.
        '''
        if isinstance(field, (Field, Array)):
            return field.array
        elif isinstance(field, MultigridField):
            return self.multigrid_to_regular(field).array
        else:
            raise TypeError("Expected Field or MultigridField got {}".format(
                type(field).__name__))

    def regular_to_multigrid(self,
                             field,
                             cshapes=None,
                             factors=None,
                             method=None):
        '''
        Converts regular field to multigrid field.
        u: `ndarray` or `Field`
            Field on the fine grid.
        cshapes: `list` of `tuple`
            List of grid sizes measured in cells, starting from the fine grid.
        '''
        mod = self.mod
        if isinstance(field, (MultigridField, NeuralNet)):
            raise TypeError("Expected Field or ndarray, got type {}".format(
                type(net).__name__))

        field = self.init_field(field)
        cshapes = cshapes or self.mg_cshapes
        factors = factors or self.mg_factors or [1] * len(cshapes)
        assert_equal(len(cshapes), len(factors))
        method = method or self.mg_interp
        terms = [
            Field(field.array / factors[0], loc=field.loc, cshape=field.cshape)
        ]
        for cshape in cshapes[1:]:
            array = mod.zeros(self._get_field_shape(cshape, loc=field.loc),
                              dtype=self.dtype)
            terms.append(Field(array, loc=field.loc, cshape=cshape))
        return MultigridField(terms=terms,
                              loc=field.loc,
                              factors=factors,
                              method=method)

    def init_field(self, field):
        '''
        Initializes field in the given backend.
        Fills omitted attributes with default values.
        '''
        mod = self.mod
        if field is None:
            return self.init_field(
                Field(None, loc='c' * self.ndim, cshape=self.cshape))
        elif isinstance(field, np.ndarray) or mod.is_tensor(field):
            array = field
            return self.init_field(
                Field(array, loc='c' * len(array.shape), cshape=array.shape))
        elif isinstance(field, Field):
            cshape = field.cshape or self.cshape
            ndim = len(cshape)
            loc = field.loc or 'c' * ndim
            assert_equal(len(loc), ndim)
            array = field.array
            if array is None:
                array = mod.zeros(self._get_field_shape(cshape, loc=loc),
                                  dtype=self.dtype)
            array = mod.variable(array, dtype=self.dtype)
            assert_equal(array.shape, self._get_field_shape(cshape, loc=loc))
            return Field(array, loc=loc, cshape=cshape)
        elif isinstance(field, MultigridField):
            return MultigridField(
                [self.init_field(term) for term in field.terms],
                loc=field.loc,
                factors=field.factors,
                axes=field.axes,
                method=field.method)
        elif isinstance(field, NeuralNet):
            dtype = self.dtype
            weights = [mod.variable(w, dtype=dtype) for w in field.weights]
            biases = [mod.variable(b, dtype=dtype) for b in field.biases]
            return NeuralNet(weights,
                             biases,
                             func_in=field.func_in,
                             func_out=field.func_out,
                             activation=field.activation)
        elif isinstance(field, list):
            u = mod.cast(mod.array(field), self.dtype)
            return self.init_field(Array(u, shape=u.shape))
        elif isinstance(field, Array):
            array = field.array
            if array is None:
                array = mod.zeros(field.shape, dtype=self.dtype)
            array = mod.variable(array, dtype=self.dtype)
            return Array(array, field.shape)
        else:
            raise TypeError("Unknown field type '{}'".format(
                type(field).__name__))

    def init_state(self, state):
        '''
        Initializes state variables in the given backend.
        '''
        fields = dict()
        for key in state.fields:
            field = state.fields[key]
            field = self.init_field(field)
            if self.multigrid and self.mg_convert_all and \
                    not isinstance(field, (MultigridField, NeuralNet, Array)):
                field = self.regular_to_multigrid(state.fields[key])
            fields[key] = field
        return State(fields=fields, initialized=True)

    def arrays_from_field(self, field):
        '''
        Returns list of data arrays of `field`.
        '''
        mod = self.mod
        if isinstance(field, Field):
            return [field.array]
        elif isinstance(field, MultigridField):
            return [term.array for term in field.terms]
        elif isinstance(field, NeuralNet):
            return field.weights + field.biases
        elif isinstance(field, Array):
            return [field.array]
        else:
            raise TypeError("Unknown field type '{}'".format(
                type(field).__name__))

    def arrays_from_state(self, state):
        '''
        Returns list of data arrays from fields in `state`.
        '''
        mod = self.mod
        res = []
        for key in state.fields:
            res += self.arrays_from_field(state.fields[key])
        return res

    @staticmethod
    def arrays_to_field(arrays, field):
        '''
        Replaces data in `field` with `arrays`.
        Returns the number of elements in `arrays` consumed.
        '''
        if isinstance(field, Field):
            field.array = arrays[0]
            return 1
        elif isinstance(field, MultigridField):
            offset = 0
            for term in field.terms:
                term.array = arrays[offset]
                offset += 1
            return offset
        elif isinstance(field, NeuralNet):
            offset = 0
            for i in range(len(field.weights)):
                field.weights[i] = arrays[offset]
                offset += 1
            for i in range(len(field.biases)):
                field.biases[i] = arrays[offset]
                offset += 1
            return offset
        elif isinstance(field, Array):
            field.array = arrays[0]
            return 1
        else:
            raise TypeError("Unknown field type '{}'".format(
                type(field).__name__))

    @staticmethod
    def arrays_to_state(arrays, state):
        '''
        Replaces data in `state` with `arrays`.

        Returns the number of elements in `arrays` consumed.
        '''
        offset = 0
        for key in state.fields:
            offset += Domain.arrays_to_field(arrays[offset:],
                                             state.fields[key])
        return offset

    def pack_field(self, field):
        '''
        Returns a flat array with data from `field`.
        '''
        mod = self.mod
        res = self.arrays_from_field(field)
        res = [mod.flatten(f) for f in res]
        return mod.concatenate(res, axis=0)

    def pack_state(self, state):
        '''
        Returns a flat array with data from `state`.
        '''
        mod = self.mod
        res = self.arrays_from_state(state)
        res = [mod.flatten(f) for f in res]
        return mod.concatenate(res, axis=0)

    def unpack_field(self, packed, field):
        '''
        Unpacks a flat array `packed` into `field`.
        Returns the number of elements in `packed` consumed.
        '''
        mod = self.mod
        arrays = self.arrays_from_field(field)
        sizes = [np.prod(a.shape) for a in arrays]
        split = mod.split_by_sizes(packed[:sum(sizes)], sizes)
        arrays = [mod.reshape(s, a.shape) for s, a in zip(split, arrays)]
        self.arrays_to_field(arrays, field)
        return sum(sizes)

    def unpack_state(self, packed, state):
        '''
        Unpacks a flat array `packed` into `state`.
        Returns the number of elements in `packed` consumed.
        '''
        mod = self.mod
        arrays = self.arrays_from_state(state)
        sizes = [np.prod(a.shape) for a in arrays]
        split = mod.split_by_sizes(packed[:sum(sizes)], sizes)
        arrays = [mod.reshape(s, a.shape) for s, a in zip(split, arrays)]
        self.arrays_to_state(arrays, state)
        return sum(sizes)

    def make_neural_net(self,
                        layers,
                        initializer='lecun',
                        func_in=None,
                        func_out=None,
                        activation=None):
        return make_neural_net(layers, self.dtype, self.mod, initializer,
                               func_in, func_out, activation)

    def field(self, state, key, *shift):
        mod = self.mod
        field = state.fields[key]
        if not isinstance(field, (Field, MultigridField, Array)):
            raise TypeError(
                "Expected Field or MultigridField, got type {} for field '{}'".
                format(type(net).__name__, key))
        if isinstance(field, Array):
            if len(shift):
                raise RuntimeError('Array requires an empty shift')
            return field.array
        shift = shift or (0, ) * self.ndim
        if len(shift) != self.ndim:
            raise RuntimeError(
                "Expected {} shift components, got shift={}".format(
                    self.ndim, shift))
        array = self.get_regular_array(field)
        return mod.roll(array, np.negative(shift), range(self.ndim))

    def neural_net(self, state, key):
        net = state.fields[key]
        if not isinstance(net, NeuralNet):
            raise TypeError(
                "Expected NeuralNet, got type {} for key='{}'".format(
                    type(net).__name__, key))
        res = lambda *inputs: eval_neural_net(net, inputs, self.mod)
        return res

    def get_context(self, state, extra=None, tracers=None):
        ctx = Context(self.domain,
                      state,
                      watch_func=lambda _: None,
                      extra=extra,
                      tracers=tracers,
                      distinct_shift=False)
        return ctx


class Field:

    def __init__(self, array=None, loc=None, cshape=None):
        '''
        array: `ndarray`
            Data array.
        loc: `str`
            Location in cell.
            One character per direction: 'c' for center and 'n' for node.
        cshape: `tuple`
            Grid size measured in cells.
        '''
        self.array = array
        self.loc = loc
        self.cshape = cshape

    def __str__(self):
        return "odil.Field({}, loc={}, cshape={:}>".format(
            str(self.array), self.loc, self.cshape)

    def __repr__(self):
        return "odil.Field({}, loc='{}', cshape={:})".format(
            repr(self.array), self.loc, self.cshape)


class MultigridField:

    def __init__(self,
                 terms=None,
                 loc=None,
                 factors=None,
                 axes=None,
                 method=None):
        '''
        terms: `list` of `Field`
            Terms of the multigrid decomposition.
        loc: `str`
            Location in cell.
            One character per direction: 'c' for center and 'n' for node.
        axes: `list` of `bool`
            Flag for each axis. If False, the axis is unchanged.
        factors: `list` of `float`
            Factor of each term in the decomposition. Defaults to 1.
        method: `str`
            Interpolation method.
        '''
        self.terms = terms
        self.loc = loc
        self.factors = factors
        self.axes = axes
        self.method = method


class NeuralNet:

    def __init__(self,
                 weights=None,
                 biases=None,
                 func_in=None,
                 func_out=None,
                 activation=None):
        '''
        weights: `list` of `ndarray`
            List of weights, matrices of size `(no, ni)` for each
            layer with `ni` inputs and `no` outputs.
        biases: `list` of `ndarray`
            List of biases,  arrays. of size `no` for each
            layer with `no` outputs.
        func_in: `callable`
            Function called on inputs.
        func_out: `callable`
            Function called on outputs.
        activation: `str`
            Name of the activation function.
        '''
        self.weights = weights
        self.biases = biases
        self.func_in = func_in
        self.func_out = func_out
        self.activation = activation or 'tanh'


class Array:

    def __init__(self, array=None, shape=None):
        '''
        array: `ndarray`
            Data array.
        shape: `tuple`
            Array shape.
        '''
        self.array = array
        self.shape = shape

    def __str__(self):
        return "odil.Array({}, shape={:}>".format(str(self.array), self.shape)

    def __repr__(self):
        return "odil.Array({}, shape={:})".format(repr(self.array), self.shape)


class State:

    def __init__(self, fields=None, initialized=False):
        '''
        fields: A dict mapping field names to state variables
            of type Field, MultigridField, or NeuralNet.
        '''
        self.fields = fields if fields is not None else dict()
        self.initialized = initialized


def interp_to_finer(u, loc=None, method=None, mod=None, depth=1):
    '''
    Interpolates a field to a finer grid.

    u: `array`
        Input field.
    loc: `str`
        Location of value in cell, one character per direction.
        'c': cell, new size `2 * n`;
        'n': node, new size `2 * (n - 1) + 1`;
        '.': none, new size `n`.
    method: `str`
        Interpolation method.
        'conv': using the transpose of convolution,
                may be slower than 'stack' with JIT (JAX and TF XLA).
        'stack': using a stack of shifted arrays.
    depth: `int`
        Number of repetitions.
    '''
    if depth == 0:
        return u

    method = method or 'stack'
    if method not in ['conv', 'stack']:
        raise ValueError("Unknown method='{}'".format(method))
    assert_equal(len(loc), len(u.shape))
    for l in loc:
        assert l in 'cn.', "Invalid loc={}".format(loc)
    dim = len(u.shape)

    # Add padding depending on value location:
    # 'c': linear extrapolation,
    # 'n': no padding,
    # '.': no padding.
    pad_width = [(1, 1) if l == 'c' else (0, 0) for l in loc]
    ur = mod.pad(u, pad_width=pad_width, mode='reflect')
    us = mod.pad(u, pad_width=pad_width, mode='symmetric')
    upad = 2 * us - ur

    if method == 'conv':
        # Convolution weights.
        wnode = np.array([1, 2, 1]) * 0.5
        wcell = np.array([1, 3, 3, 1]) * 0.25
        wnone = np.array([1.])
        wloc = {'n': wnode, 'c': wcell, '.': wnone}
        w = wloc[loc[0]]
        for i in range(1, dim):
            w = np.kron(wloc[loc[i]], w[..., None])
        w = mod.cast(mod.reshape(w, w.shape + (1, 1)), u.dtype)

        # Output shape.
        oshape = lambda s: {'n': s * 2 + 1, 'c': s * 2 + 2, '.': s}
        oshape = tuple(oshape(s)[l] for l, s in zip(loc, upad.shape))
        oshape = (1, ) + oshape + (1, )
        strides = tuple(1 if l == '.' else 2 for l in loc)
        upad = mod.reshape(upad, (1, ) + upad.shape + (1, ))
        res = mod.conv_transpose(upad,
                                 filters=w,
                                 output_shape=oshape,
                                 strides=strides,
                                 padding='VALID')
        # Remove edges.
        oslice = {'n': slice(1, -1), 'c': slice(3, -3), '.': slice(0, None)}
        res = res[(0, ) + tuple(oslice[l] for l in loc) + (0, )]
    elif method == 'stack':

        def term(*dd, ww=None):
            dd = [tuple(-v for v in d) for d in dd]
            return sum(w * mod.roll(upad, d, range(dim))
                       for d, w in zip(dd, ww) if w) / sum(ww)

        # Offsets of nodes of a dim-dimensional cube.
        dd = np.meshgrid(*[[0] if l == '.' else [0, 1] for l in loc],
                         indexing='ij')
        dshape = tuple(1 if l == '.' else 2 for l in loc)
        # Example dim=2: dd = [(0,0), (0,1), (1,0), (1,1)].
        dd = np.reshape(dd, (dim, -1)).T
        # Indices with location in node.
        sn = [i for i, l in enumerate(loc) if l == 'n']
        # Indices with location in cell.
        sc = [i for i, l in enumerate(loc) if l == 'c']

        def weight(r, d):
            return (3**(sum(1 - abs(r - d)[sc]))  #
                    if np.all((r - d)[sn] <= 0) else 0)

        uu = [term(*dd, ww=[weight(r, d) for r in dd]) for d in dd]
        res = mod.stack(uu)
        res = mod.reshape(res, dshape + upad.shape)
        for i in range(dim):
            res = [res[i] for i in range(res.shape[0])]
            res = mod.stack(res, axis=dim + i)
        res = mod.reshape(res,
                          tuple(s * d for s, d in zip(upad.shape, dshape)))
        # Remove edges.
        oslice = {'n': slice(0, -1), 'c': slice(1, -3), '.': slice(0, None)}
        res = res[tuple(oslice[l] for l in loc)]
    else:
        raise ValueError('Unknown method=' + method)

    return interp_to_finer(res, loc, method, mod, depth - 1)


def restrict_to_coarser(u, loc=None, method=None, mod=None, depth=1):
    '''
    Restricts a field to a coarser grid.

    u: `array`
        Input field.
    loc: `str`
        Location of value in cell, one character per direction.
        'c': cell, new size `n // 2`;
        'n': node, new size `(n - 1) // 2 + 1`;
        '.': none, new size `n`.
    method: `str`
        Restriction method.
        'conv': using the convolution
    depth: `int`
        Number of repetitions.
    '''
    if depth == 0:
        return u

    method = method or 'conv'
    if method not in ['conv']:
        raise ValueError("Unknown method='{}'".format(method))
    assert_equal(len(loc), len(u.shape))
    for l in loc:
        assert l in 'cn.', "Invalid loc={}".format(loc)
    dim = len(u.shape)

    # Add padding depending on value location:
    # 'c': no padding,
    # 'n': linear extrapolation, combined with the [1,2,1] kernel
    #      implements the identity condition on the boundaries
    # '.': no padding.
    pad_width = [(1, 1) if l == 'n' else (0, 0) for l in loc]
    ur = mod.pad(u, pad_width=pad_width, mode='reflect')
    us = mod.pad(u, pad_width=pad_width, mode='symmetric')
    upad = 2 * us - ur

    if method == 'conv':
        # Convolution weights.
        wnode = np.array([1, 2, 1]) * 0.25
        wcell = np.array([1, 1]) * 0.5
        wnone = np.array([1.])
        wloc = {'n': wnode, 'c': wcell, '.': wnone}
        w = wloc[loc[0]]
        for i in range(1, dim):
            w = np.kron(wloc[loc[i]], w[..., None])
        w = mod.cast(w, u.dtype)
        res = mod.convolution(upad, filters=w, strides=2, padding='VALID')
    else:
        raise ValueError('Unknown method=' + method)

    return restrict_to_coarser(res, loc, method, mod, depth - 1)


def check_multigrid_cshapes(cshapes, axes=None):
    '''
    Checks grid sizes in a multigrid hierarchy.

    cshapes: `list` of `tuple`
        List of grid sizes measured in cells, starting from the fine grid.
    '''
    if not len(cshapes):
        return
    dim = len(cshapes[0])
    axes = axes or [True] * dim
    assert_equal(len(axes), dim)
    for i in range(1, len(cshapes)):
        for j in range(dim):
            if axes[j]:
                n = cshapes[i][j]
                nfine_expect = n * 2
                nfine = cshapes[i - 1][j]
                assert_equal(nfine, nfine_expect,
                             ' with cshapes={:}'.format(cshapes))


def make_neural_net(layers,
                    dtype,
                    mod,
                    initializer='lecun',
                    func_in=None,
                    func_out=None,
                    activation=None):
    '''
    Returns a NeuralNet with random initial weights and zero biases.

    layers: `list` of `int`
        Number of neuron in each layer.
    '''

    def get_scale(ni, no):
        if initializer == 'legacy':
            return np.sqrt(1. / ni)
        elif initializer == 'glorot':
            return np.sqrt(6. / (ni + no))
        elif initializer == 'lecun':
            return np.sqrt(3. / ni)
        elif initializer == 'he':
            return np.sqrt(6. / ni)
        raise ValueError('Unknown initializer=' + initializer)

    weights = []
    biases = []
    for ni, no in zip(layers[:-1], layers[1:]):
        scale = get_scale(ni, no)
        weights.append(  #
            mod.random.uniform(shape=(no, ni),
                               minval=-scale,
                               maxval=scale,
                               dtype=dtype))
        biases.append(mod.zeros(no, dtype=dtype))
    return NeuralNet(weights,
                     biases,
                     func_in=func_in,
                     func_out=func_out,
                     activation=activation)


def eval_neural_net(net: NeuralNet, inputs, mod, frozen=False):
    '''
    Evaluates a fully connected neural network.

    net: `NeuralNet`
        Neural network to evaluate.
    inputs: `list` of `ndarray`
        List of input arrays. All arrays must have the same shape.
    frozen: `bool`
        Apply `stop_gradient()` to the weights.

    Returns: `list` of `ndarray`
        List of output arrays.
        The output arrays have the same shape as the input arrays.
    '''
    weights = net.weights
    biases = net.biases

    # Check shapes of weights and biases.
    assert_equal(len(weights), len(biases), 'Weights and biases do not match')
    assert_equal(weights[0].shape[1], len(inputs),
                 'Weights and inputs do not match')
    for w, b in zip(weights, biases):
        assert_equal(w.shape[0], b.shape[0])

    if frozen:
        weights = [mod.stop_gradient(w) for w in weights]
        biases = [mod.stop_gradient(b) for b in biases]

    func_act = {
        'tanh': mod.tanh,
        'relu': mod.relu,
        'none': lambda x: x,
    }[net.activation]

    n = len(weights)
    if net.func_in is not None:
        inputs = net.func_in(*inputs)
    tmp = mod.stack(inputs, axis=0)
    axes = range(len(tmp.shape))
    perm = np.roll(axes, -1)  # (1, ..., N-1, 0)
    # Let the last two dimensions of `tmp` be columns of inputs.
    tmp = mod.transpose(tmp, perm)[..., None]
    for i in range(n):
        w = weights[i]
        b = biases[i]
        # Multiply matrix `w` by each column of `tmp` and add bias.
        tmp = mod.matmul(w, tmp) + b[:, None]
        if i < n - 1:
            tmp = func_act(tmp)
    perm = np.roll(axes, 1)  # (N-1, 0, ..., N-2)
    # Let the last two dimensions of `tmp` be columns of inputs.
    tmp = mod.transpose(tmp[..., 0], perm)
    outputs = [tmp[i] for i in range(tmp.shape[0])]
    if net.func_out is not None:
        outputs = net.func_out(*outputs)
    return outputs


class Context:

    class Raw:

        def __init__(self, value):
            self.value = value

    def __init__(self,
                 domain,
                 state,
                 watch_func=None,
                 extra=None,
                 tracers=None,
                 distinct_shift=False):
        '''
        watch_func: `callable`
            Callback to add an array to watched variables.
        distinct_shift: `bool`
            If True, fields with different shifts returned by `field()` are
            considered distinct symbols, which is used by `linearize()`
            and `eval_operator_grad()`.
        extra:
            Regular Python value to be available as `ctx.extra`.
            Typically an `argparse.Namespace` object.
            Changing this value does not trigger recompilation.
        tracers: `dict`
            Dictionary of arrays, scalars, or standard Python containers thereof.
            This argument becomes part of the signature of the jitted function.
        '''
        self.domain = domain
        self.state = state
        self.watch_func = watch_func or (lambda _: None)
        self.extra = extra
        self.tracers = tracers
        self.dtype = domain.dtype
        self.mod = domain.mod
        self.distinct_shift = distinct_shift
        # Storage for field arguments. Mapping from (key, shift, loc) to array.
        self.desc_to_array = dict()
        # Storage for field arguments to request full Jacobian.
        self.key_to_array_jac = dict()
        # Aliases for Domain methods.
        self.step = domain.step
        self.size = domain.size
        self.indices = domain.indices
        self.points = domain.points

    def cast(self, value, dtype=None):
        dtype = dtype or self.dtype
        return self.mod.cast(value, dtype)

    def field(self, key, *shift, loc=None, frozen=False):
        domain = self.domain
        mod = domain.mod
        field = self.state.fields[key]
        if not isinstance(field, (Field, MultigridField, Array)):
            raise TypeError(
                "Expected Field or MultigridField, got type {} for key='{}'".
                format(type(net).__name__, key))
        if isinstance(field, Array):
            if len(shift):
                raise RuntimeError('Array requires an empty shift')
            self.watch_func(field.array)
            self.key_to_array_jac[(key, None, None)] = field.array
            array = field.array
            if frozen:
                array = mod.stop_gradient(array)
            return array
        shift_src = (0, ) * domain.ndim
        shift = shift or shift_src
        loc = loc or field.loc
        if len(shift) != domain.ndim:
            raise RuntimeError(
                "Expected {} shift components, got shift={}".format(
                    domain.ndim, shift))
        desc = (key, shift, loc)  # Descriptor of target field.
        desc_src = (key, shift_src, field.loc)  # Descriptor of source field.
        if desc in self.desc_to_array:  # Reuse target field.
            array = self.desc_to_array[desc]
            if self.distinct_shift and isinstance(field, Field):
                self.watch_func(array)
        else:
            if desc_src in self.desc_to_array:  # Reuse source field.
                array_src = self.desc_to_array[desc_src]
            else:  # Create new source field.
                if not self.distinct_shift:
                    if isinstance(field, Field):
                        self.watch_func(field.array)
                    elif isinstance(field, MultigridField):
                        for term in field.terms:
                            self.watch_func(term.array)
                array_src = domain.get_regular_array(field)
                self.desc_to_array[desc_src] = array_src
            if self.distinct_shift and desc != desc_src:
                # If shifts are treated separately,
                # avoid differentiation through the source field.
                array_src = mod.stop_gradient(array_src)
            array = array_src
            # True if needs padding in that direction.
            pad_flag = [
                lf == 'c' and l == 'n' for lf, l in zip(field.loc, loc)
            ]
            # Pad the array to account for change of location.
            if any(pad_flag):
                pad_width = [(1, 0) if f else (0, 0) for f in pad_flag]
                array = mod.pad(array, pad_width=pad_width, mode='constant')
            # Shift the array.
            if shift != shift_src:
                array = mod.roll(array, np.negative(shift), range(domain.ndim))
            # True if needs trimming in that direction.
            trim_flag = [
                lf == 'n' and l == 'c' for lf, l in zip(field.loc, loc)
            ]
            # Trim the array to account for change of location.
            if any(trim_flag):
                trim_slice = [slice(0, -1 if f else None) for f in trim_flag]
                array = array[trim_slice]
            if self.distinct_shift and isinstance(field, Field):
                self.watch_func(array)
            self.desc_to_array[desc] = array
        if frozen:
            array = mod.stop_gradient(array)
        return array

    def neural_net(self, key, frozen=False):
        domain = self.domain
        net = self.state.fields[key]
        if not isinstance(net, NeuralNet):
            raise TypeError(
                "Expected NeuralNet, got type {} for key='{}'".format(
                    type(net).__name__, key))
        arrays = domain.arrays_from_field(net)
        self.watch_func(arrays)
        if self.distinct_shift:
            self.key_to_array_jac[(key, None, None)] = arrays
        res = lambda *inputs: eval_neural_net(
            net, inputs, self.mod, frozen=frozen)
        return res


class Problem:

    def __init__(self, operator, domain, extra=None, tracers=None, jit=None):
        '''
        operator: callable(mod, ctx)
            Discrete operator returning fields on grid.
            Each field corresponds to an equation to be solved.
            Arguments are:
                mod: module with mathematical functions (tensorflow, numpy)
                ctx: instance of Context
        domain: instance of Domain
        '''
        self.domain = domain
        self.operator = operator
        self.extra = extra
        if tracers is None:
            tracers = dict()
        if 'epoch' not in tracers:
            tracers['epoch'] = 0
        self.tracers = tracers
        mod = domain.mod
        if jit is None:
            from .runtime import jit
        self.jit = jit

        self._cache_eval_loss_grad = dict()
        self._cache_eval_operator = dict()
        self._cache_eval_operator_grad = dict()

        if isinstance(mod, ModTensorflow):
            self._eval_loss_grad = self._eval_loss_grad_tf
            self._eval_operator = self._eval_operator_tf
            self._eval_operator_grad = self._eval_operator_grad_tf
        elif mod.jax is not None:
            self._eval_loss_grad = self._eval_loss_grad_jax
            self._eval_operator = self._eval_operator_jax
            self._eval_operator_grad = self._eval_operator_grad_jax
        else:
            raise NotImplementedError('Unsupported mod={:}'.format(mod))

    def _eval_loss_grad_tf(self, state):
        domain = self.domain
        mod = domain.mod
        tf = mod.tf
        cache = self._cache_eval_loss_grad

        if 'state' not in cache:
            cache['state'] = domain.init_state(state)

        def func(arrays):
            with tf.GradientTape(persistent=False,
                                 watch_accessed_variables=False) as tape:
                domain.arrays_to_state(arrays, cache['state'])
                ctx = Context(domain,
                              cache['state'],
                              watch_func=tape.watch,
                              extra=self.extra,
                              tracers=self.tracers)
                ff = self.operator(ctx)
                assert isinstance(ff, (tuple, list)) and len(ff), \
                    'Operator must return a non-empty list'
                names = [f[0] if isinstance(f, tuple) else '' for f in ff]
                values = [f[1] if isinstance(f, tuple) else f for f in ff]
                terms = [
                    mod.mean(v.value) if isinstance(v, Context.Raw)  #
                    else mod.mean(mod.square(v)) for v in values
                ]
                loss = sum(terms)
                norms = [
                    t if isinstance(v, Context.Raw)  #
                    else mod.sqrt(t) for t, v in zip(terms, values)
                ]
            grads = tape.gradient(loss, arrays)
            grads = [
                g if g is not None else mod.zeros_like(u)
                for u, g in zip(arrays, grads)
            ]
            if 'names' not in cache:
                cache['names'] = names
            return loss, grads, terms, norms

        if 'func' not in cache:
            cache['func'] = tf.function(func, jit_compile=self.jit)

        # Evaluate gradients.
        arrays = domain.arrays_from_state(state)
        loss, grads, terms, norms = cache['func'](arrays)
        return loss, grads, terms, cache['names'], norms

    def _eval_loss_grad_jax(self, state):
        domain = self.domain
        mod = domain.mod
        jax = mod.jax
        cache = self._cache_eval_loss_grad

        def eval_loss(arrays, tracers):
            if 'state' not in cache:
                cache['state'] = domain.init_state(state)
            domain.arrays_to_state(arrays, cache['state'])
            ctx = Context(domain,
                          cache['state'],
                          extra=self.extra,
                          tracers=tracers)
            ff = self.operator(ctx)
            assert isinstance(ff, (tuple, list)) and len(ff), \
                'Operator must return a non-empty list'
            names = [f[0] if isinstance(f, tuple) else '' for f in ff]
            values = [f[1] if isinstance(f, tuple) else f for f in ff]
            terms = [
                mod.mean(f.value) if isinstance(f, Context.Raw) \
                else mod.mean(mod.square(f)) for f in values
            ]
            loss = sum(terms)
            norms = [
                t if isinstance(v, Context.Raw)  #
                else mod.sqrt(t) for t, v in zip(terms, values)
            ]
            return loss, (terms, names, norms)

        def func(arrays, tracers):
            # Create gradient function.
            fgrad = jax.value_and_grad(eval_loss, argnums=[0], has_aux=True)
            (loss, (terms, names, norms)), grads = fgrad(arrays, tracers)
            # Use cache to store the names as JAX does not support `str`.
            cache['names'] = names
            return loss, grads[0], terms, norms

        if 'func' not in cache:
            cache['func'] = jax.jit(func)

        arrays = domain.arrays_from_state(state)
        loss, grads, terms, norms = cache['func'](arrays, self.tracers)
        return loss, grads, terms, cache['names'], norms

    def linearize(self, state, modsp=None):
        '''
        Returns a vector V0 and a sparse matrix M that linearize the operator
            operator(V) ~= M @ (V - V0) + V0
        where V is a flattened state.
        '''
        if not state.initialized:
            raise RuntimeError(
                'Uninitialized state, use `state = domain.init_state(state)`')
        domain = self.domain
        mod = domain.mod
        modsp = modsp or mod.modsp
        if modsp is None:
            import scipy
            import scipy.sparse as modsp

        values, grads, names = self.eval_operator_grad(state)

        # Mapping from field key to offset in flattened state.
        key_to_offset = dict()
        # Mapping from field key to size of flattened data.
        key_to_size = dict()
        offset = 0
        # Iterate over unknown fields.
        for key, field in state.fields.items():
            arrays = domain.arrays_from_field(field)
            size = sum(np.prod(array.shape) for array in arrays)
            key_to_offset[key] = offset
            key_to_size[key] = size
            offset += size
        size_all = offset  # Size of the vector of unknowns.
        del offset

        def field_to_matrix(key, shift, loc, field, garray):
            offset = key_to_offset[key]
            size = key_to_size[key]
            cols = offset + mod.reshape(mod.arange(size), field.array.shape)
            # True if needs padding in that direction.
            pad_flag = [
                lf == 'c' and l == 'n' for lf, l in zip(field.loc, loc)
            ]
            # Pad the array to account for change of location.
            if any(pad_flag):
                pad_width = [(1, 0) if f else (0, 0) for f in pad_flag]
                cols = mod.pad(cols, pad_width=pad_width, mode='constant')
            # Shift the array.
            if shift != shift_src:
                cols = mod.roll(cols, np.negative(shift), range(domain.ndim))
            # True if needs trimming in that direction.
            trim_flag = [
                lf == 'n' and l == 'c' for lf, l in zip(field.loc, loc)
            ]
            # Trim the array to account for change of location.
            if any(trim_flag):
                trim_slice = [slice(0, -1 if f else None) for f in trim_flag]
                cols = cols[trim_slice]
            # Indices of rows, i.e. components of operator value.
            rows = mod.arange(np.prod(value.shape))
            # Indices of columns, i.e. components of unknown field.
            cols = mod.flatten(cols)
            garray = mod.flatten(garray)
            matr = modsp.csr_array((garray, (rows, cols)),
                                   dtype=domain.dtype,
                                   shape=(np.prod(value.shape), size_all))
            return matr

        shift_src = (0, ) * domain.ndim
        matrices = []
        vectors = []
        for name, value, grad in zip(names, values, grads):
            # Shape of the resulting matrix.
            mshape = (np.prod(value.shape), size_all)
            # Rows of the resulting matrix corresponding to one operator value.
            matrix = modsp.csr_array(mshape, dtype=domain.dtype)
            for desc, garray in grad.items():
                key, shift, loc = desc
                if garray is None:
                    # Skip empty gradient.
                    continue
                if isinstance(garray, list) and all(a is None for a in garray):
                    # Skip array of empty gradients.
                    continue
                field = state.fields[key]
                if shift is None:  # Full Jacobian.
                    if isinstance(garray, list):
                        garray = [
                            mod.reshape(a, [mshape[0], -1]) for a in garray
                        ]
                        garray = mod.concatenate(garray, axis=1)
                    # Dense array to sparse.
                    m = modsp.csr_array(garray)
                    # Shift the columns to the field's offset.
                    m = modsp.csr_array(
                        (m.data, m.indices + key_to_offset[key], m.indptr),
                        shape=mshape)
                    matrix += m
                else:  # Element-wise gradient.
                    if not isinstance(field, Field):
                        raise TypeError(
                            "Expected Field, got type {} for key='{}'".format(
                                type(field).__name__, key))
                    matrix += field_to_matrix(key, shift, loc, field, garray)
            matrices.append(matrix)
            vectors.append(mod.flatten(value))
        # Append gradients from weights to gradients from fields.
        matrix = modsp.vstack(matrices)
        vector = mod.concatenate(vectors, axis=0)

        return vector, matrix

    def eval_loss_grad(self, state):
        '''
        Evaluates the loss function and its gradient on `state`.

        Returns:
        loss: `float`
            Value of the loss function.
        grads: `list` of `ndarray`
            Gradient of the loss with respect to each array in the state.
        terms: `list` of `float`
            Terms of the loss function (mean squared operator values).
        names: `list` of `str`
            Names of the operator values.
        norms: `list` of `float`
            Norms (root-mean-square) of the operator values.
        '''
        if not state.initialized:
            raise RuntimeError(
                'Uninitialized state, use `state = domain.init_state(state)`')
        loss, grads, terms, names, norms = self._eval_loss_grad(state)
        loss = np.array(loss)
        terms = list(map(np.array, terms))
        norms = list(map(np.array, norms))
        return loss, grads, terms, names, norms

    def _eval_operator_tf(self, state):
        domain = self.domain
        mod = domain.mod
        tf = mod.tf
        cache = self._cache_eval_operator

        if 'state' not in cache:
            cache['state'] = domain.init_state(state)

        def func(arrays, tracers):
            domain.arrays_to_state(arrays, cache['state'])
            ctx = Context(domain,
                          cache['state'],
                          extra=self.extra,
                          tracers=tracers)
            ff = self.operator(ctx)
            assert isinstance(ff, (tuple, list)) and len(ff), \
                'Operator must return a non-empty list'
            names = [f[0] if isinstance(f, tuple) else '' for f in ff]
            values = [f[1] if isinstance(f, tuple) else f for f in ff]
            if 'names' not in cache:
                cache['names'] = names
            return values

        if 'func' not in cache:
            cache['func'] = tf.function(func, jit_compile=self.jit)

        # Evaluate gradients.
        arrays = domain.arrays_from_state(state)
        values = cache['func'](arrays, self.tracers)
        return values, cache['names']

    def _eval_operator_jax(self, state):
        domain = self.domain
        mod = domain.mod
        jax = mod.jax
        cache = self._cache_eval_operator

        def func(arrays, tracers):
            if 'state' not in cache:
                cache['state'] = domain.init_state(state)
            domain.arrays_to_state(arrays, cache['state'])
            ctx = Context(domain,
                          cache['state'],
                          extra=self.extra,
                          tracers=tracers)
            ff = self.operator(ctx)
            assert isinstance(ff, (tuple, list)) and len(ff), \
                'Operator must return a non-empty list'
            names = [f[0] if isinstance(f, tuple) else '' for f in ff]
            values = [f[1] if isinstance(f, tuple) else f for f in ff]
            if 'names' not in cache:
                cache['names'] = names
            return values

        if 'func' not in cache:
            cache['func'] = jax.jit(func)

        arrays = domain.arrays_from_state(state)
        values = cache['func'](arrays, self.tracers)
        return values, cache['names']

    def eval_operator(self, state):
        '''
        Evaluates the operator on `state`.

        Returns:
        values: `list` of `ndarray`
            Arrays containing the operator values.
        names: `list` of `str`
            Names of the operator values.
        '''
        if not state.initialized:
            raise RuntimeError(
                'Uninitialized state, use `state = domain.init_state(state)`')
        values, names = self._eval_operator(state)
        return values, names

    def _eval_operator_grad_tf(self, state):
        domain = self.domain
        mod = domain.mod
        tf = mod.tf
        cache = self._cache_eval_operator_grad

        if 'state' not in cache:
            cache['state'] = domain.init_state(state)

        def func(arrays, tracers):
            with tf.GradientTape(persistent=True,
                                 watch_accessed_variables=False) as tape:

                domain.arrays_to_state(arrays, cache['state'])
                ctx = Context(domain,
                              cache['state'],
                              watch_func=tape.watch,
                              extra=self.extra,
                              tracers=tracers,
                              distinct_shift=True)
                ff = self.operator(ctx)
                assert isinstance(ff, (tuple, list)) and len(ff), \
                    'Operator must return a non-empty list'
                names = [f[0] if isinstance(f, tuple) else '' for f in ff]
                values = [f[1] if isinstance(f, tuple) else f for f in ff]
                sums = [tf.reduce_sum(v) for v in values]
            grads = [tape.gradient(s, ctx.desc_to_array) for s in sums]
            # Having experimental_use_pfor=True leads to excessive memory usage,
            # sufficient to store a dense matrix of size `prod(domain.shape)**2`.
            grads = []
            for v, s in zip(values, sums):
                g = tape.gradient(s, ctx.desc_to_array)
                if ctx.key_to_array_jac:
                    jac = tape.jacobian(v,
                                        ctx.key_to_array_jac,
                                        experimental_use_pfor=False)
                    g.update(jac)
                grads.append(g)
            if 'names' not in cache:
                cache['names'] = names
            return values, grads

        if 'func' not in cache:
            cache['func'] = tf.function(func, jit_compile=self.jit)

        # Evaluate gradients.
        arrays = domain.arrays_from_state(state)
        values, grads = cache['func'](arrays, self.tracers)
        return values, grads, cache['names']

    def _eval_operator_grad_jax(self, state):
        raise NotImplementedError()

    def eval_operator_grad(self, state):
        '''
        Evaluates the operator and its gradient with respect to fields in state.

        Returns:
        values: `list` of `ndarray`
            Arrays containing the operator values.
        grads: `list` of `dict`
            For each operator value, contains a mapping with the gradient
            of that value with respect to stencil fields.
            The mapping is from tuples (key, shift, loc) to array.
        names: `list` of `str`
            Names of the operator values.
        '''
        if not state.initialized:
            raise RuntimeError(
                'Uninitialized state, use `state = domain.init_state(state)`')
        values, grads, names = self._eval_operator_grad(state)
        return values, grads, names

    def get_context(self, state):
        return self.domain.get_context(extra=self.extra, tracers=self.tracers)


def checkpoint_save(domain, state, path):
    '''
    Saves state to a checkpoint file.
    The checkpoint file will contain a mapping with one entry `fields`
    which is itself a mapping from field keys to lists of arrays
    returned by `domain.arrays_from_field()`.

    path: `str`
        Path to the new checkpoint `.pickle` file.
    '''
    res = dict()
    fields = dict()
    for key in state.fields:
        arrays = domain.arrays_from_field(state.fields[key])
        fields[key] = list(map(np.array, arrays))
    res['fields'] = fields

    with open(path, 'wb') as f:
        pickle.dump(res, f)


def checkpoint_load(domain, state, path, skip_missing=True, keys=None):
    '''
    Loads fields from a checkpoint file to state.
    The checkpoint file must contain a mapping with one entry `fields`
    which is itself a mapping from field keys to lists of arrays
    returned by `domain.arrays_from_field()`.

    state: `State`
        A state with defined fields to load.
    skip_missing: `bool`
        If True, ignore fields from `field_to_load` missing in the checkpoint.
    keys: `list` of `str`
        List of field keys to load.
    '''
    with open(path, 'rb') as f:
        s = pickle.load(f)
    data = s.get('fields', dict())
    keys = keys or state.fields.keys()
    for key in keys:
        if key not in data:
            if not skip_missing:
                raise RuntimeError(f"Field {key} not found in {path}")
            continue
        arrays = data[key]
        if not isinstance(arrays, list):
            arrays = [arrays]
        domain.arrays_to_field(arrays, state.fields[key])


def extrap_quadh(u0, u1, u1p):
    '''
    Quadratic extrapolation from points 0, 1, 1.5 to point 2.
    Suffix `h` means half.
    '''
    u2 = (u0 - 6 * u1 + 8 * u1p) / 3
    return u2


def extrap_quad(u0, u1, u2):
    'Quadratic extrapolation from points 0, 1, 2 to point 3.'
    u3 = u0 - 3 * u1 + 3 * u2
    return u3


def extrap_linear(u0, u1):
    'Linear extrapolation from points 0, 1 to point 2.'
    u2 = 2 * u1 - u0
    return u2


def latin_hypercube(ndim, size, dtype):
    '''
    Returns `size` points from the unit cube.
    '''
    # XXX Copied from pyDOE.
    cut = np.linspace(0, 1, size + 1, dtype=dtype)
    u = np.random.rand(size, ndim).astype(dtype)
    a = cut[:size]
    b = cut[1:size + 1]
    rdpoints = np.zeros_like(u, dtype=dtype)
    for j in range(ndim):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    H = np.zeros_like(rdpoints)
    for j in range(ndim):
        order = np.random.permutation(range(size))
        H[:, j] = rdpoints[order, j]
    return H


class Approx():

    def __init__(self, domain):
        self.domain = domain
        self.mod = domain.mod

    def stencil(self, q):
        'Returns: q, qxm, qxp, qym, qyp.'
        mod = self.mod
        st = [None] * 5
        st[0] = q
        st[1] = mod.roll(st[0], shift=1, axis=0)
        st[2] = mod.roll(st[0], shift=-1, axis=0)
        st[3] = mod.roll(st[0], shift=1, axis=1)
        st[4] = mod.roll(st[0], shift=-1, axis=1)
        return st

    def stencil5(self, st):
        'Returns: qxmm, qxpp, qymm, qypp.'
        mod = self.mod
        st5 = [None] * 4
        st5[0] = mod.roll(st[1], shift=1, axis=0)
        st5[1] = mod.roll(st[2], shift=-1, axis=0)
        st5[2] = mod.roll(st[3], shift=1, axis=1)
        st5[3] = mod.roll(st[4], shift=-1, axis=1)
        return st5

    def central(self, st):
        hx, hy = self.domain.step()
        q, qxm, qxp, qym, qyp = st
        q_x = (qxp - qxm) / (2 * hx)
        q_y = (qyp - qym) / (2 * hy)
        return q_x, q_y

    def apply_bc_extrap_linear(self, st):
        'Linear extrapolation from inner cells to halo cells.'
        domain = self.domain
        nx, ny = domain.size()
        ix, iy = domain.indices()
        mod = domain.mod
        extrap = extrap_quad
        st[1] = mod.where(ix == 0, extrap(st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap(st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap(st[3], st[0]), st[4])
        return st

    def apply_bc_extrap_quad(self, st, st5):
        'Linear extrapolation from inner cells to halo cells.'
        domain = self.domain
        nx, ny = domain.size()
        ix, iy = domain.indices()
        mod = domain.mod
        extrap = extrap_quad
        st[1] = mod.where(ix == 0, extrap(st5[1], st[2], st[0]), st[1])
        st[2] = mod.where(ix == nx - 1, extrap(st5[0], st[1], st[0]), st[2])
        st[3] = mod.where(iy == 0, extrap(st5[3], st[4], st[0]), st[3])
        st[4] = mod.where(iy == ny - 1, extrap(st5[2], st[3], st[0]), st[4])
        return st

    def vorticity(self, u, v):
        u_st = self.stencil(u)
        v_st = self.stencil(v)
        self.apply_bc_extrap_quad(u_st, self.stencil5(u_st))
        self.apply_bc_extrap_quad(v_st, self.stencil5(v_st))
        _, u_y = self.central(u_st)
        v_x, _ = self.central(v_st)
        omega = v_x - u_y
        return omega


def struct_to_numpy(mod, d):
    if mod.is_tensor(d):
        return np.array(d)
    if isinstance(d, dict):
        for key in d:
            d[key] = struct_to_numpy(mod, d[key])
        return d
    if isinstance(d, list):
        return [struct_to_numpy(mod, a) for a in d]
    if isinstance(d, tuple):
        return tuple(struct_to_numpy(mod, a) for a in d)
    return d
