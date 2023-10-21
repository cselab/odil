import numpy as np
from argparse import Namespace


class Domain():

    def __init__(self,
                 domain=None,
                 ndim=None,
                 lower=None,
                 upper=None,
                 dimnames=None,
                 dtype=None,
                 cshape=None):
        domain = domain or Namespace(ndim=None,
                                     lower=0.,
                                     upper=1.,
                                     dimnames=None,
                                     dtype=None,
                                     cshape=None)
        dtype = dtype or domain.dtype
        cshape = cshape or domain.cshape
        ndim = ndim or domain.ndim
        dimnames = dimnames or domain.dimnames or ['x', 'y', 'z'][:ndim]
        lower = lower if lower is not None else domain.lower
        upper = upper if upper is not None else domain.upper
        ndim = len(cshape)
        lower = (np.ones(ndim, dtype=dtype) * lower).astype(dtype)
        upper = (np.ones(ndim, dtype=dtype) * upper).astype(dtype)
        self.ndim = ndim
        self.lower = lower
        self.upper = upper
        self.dimnames = dimnames
        self.dtype = dtype
        self.cshape = cshape

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

    def _points_cell_1d(self, d):
        x = np.linspace(self.lower[d],
                        self.upper[d],
                        self.cshape[d],
                        endpoint=False,
                        dtype=self.dtype)
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
