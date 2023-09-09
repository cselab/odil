#!/usr/bin/env python3

import numpy as np
import odil
from odil.runtime import mod


def test(method, dim, loc):
    '''
    Checks that `restrict_to_coarser()` is exact
    on linear functions with discontinuous boundary values.
    '''
    cshapeh = 3 + np.array(range(dim))
    cshape = cshapeh * 2

    dimnames = ['x', 'y', 'z', 'w'][:dim]
    domain = odil.Domain(cshape=cshape, dimnames=dimnames)
    domainh = odil.Domain(cshape=cshapeh, dimnames=dimnames)
    dtype = domain.dtype
    xx = domain.points(loc=loc)
    xxh = domainh.points(loc=loc)

    def func(xx):
        # Linear function with discontinuous boundary values.
        res = mod.zeros_like(xx[0])
        for i in range(len(xx)):
            res += xx[i] * (i + 1)
            res += mod.cast(mod.where(xx[i] == 0, 10, 0), dtype)
            res += mod.cast(mod.where(xx[i] == 1, 10, 0), dtype)
        return res

    u = func(xx)
    uh = func(xxh)
    uhr = odil.core.restrict_to_coarser(u, loc=loc, mod=mod, method=method)
    error = mod.max(abs(uhr - uh))
    if error > np.finfo(dtype).eps * 100:
        return 'error={:.8g}'.format(error)
    return None


def main():
    failed = 0
    for method in ['conv']:
        for dim in [1, 2, 3, 4]:
            for loc in {'c' * dim, 'n' * dim, 'cnnn'[:dim], 'nccc'[:dim]}:
                # Skip unsupported combinations.
                if (isinstance(mod, odil.backend.ModTensorflow)  #
                        and dim == 4 and method == 'conv'):
                    continue
                msg = f'method={method:>4}, dim={dim}, loc={loc:>4}:'
                try:
                    error = test(method, dim, loc)
                except Exception as e:
                    error = e
                failed += bool(error)
                print(msg, f"FAIL {error}" if error else "PASS")
    exit(failed)


if __name__ == "__main__":
    main()
