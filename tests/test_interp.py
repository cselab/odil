#!/usr/bin/env python3

import numpy as np
import odil
from odil.runtime import mod


def test(method, dim, loc):
    cshapeh = 3 + np.array(range(dim))
    cshape = cshapeh * 2

    dimnames = ['x', 'y', 'z', 'w'][:dim]
    domain = odil.Domain(cshape=cshape, dimnames=dimnames)
    domainh = odil.Domain(cshape=cshapeh, dimnames=dimnames)
    dtype = domain.dtype
    xx = domain.points(loc=loc)
    xxh = domainh.points(loc=loc)

    def func(xx):
        # Linear function in space.
        return sum(x * np.sqrt(i + 1) for i, x in enumerate(xx))

    u = func(xx)
    uh = func(xxh)
    ui = odil.core.interp_to_finer(uh, loc=loc, mod=mod, method=method)
    error = mod.max(abs(ui - u))
    if error > np.finfo(dtype).eps * 100:
        return 'error={:.8g}'.format(error)
    return None


def main():
    failed = 0
    for method in ['conv', 'stack']:
        for dim in [1, 2, 3, 4]:
            for loc in {'c' * dim, 'n' * dim, 'cnnn'[:dim], 'nccc'[:dim]}:
                # Skip unsupported combinations.
                if dim == 4 and method == 'conv':
                    continue
                try:
                    error = test(method, dim, loc)
                except Exception as e:
                    error = e
                failed += bool(error)
                print(f'method={method:>5}, dim={dim}, loc={loc:>4}:',
                      f"FAIL {error}" if error else "PASS")
    exit(failed)


if __name__ == "__main__":
    main()
