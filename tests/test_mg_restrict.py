#!/usr/bin/env python3

import argparse
import numpy as np
import odil
from odil.runtime import mod


def test(method, ndim, loc):
    '''
    Checks that `restrict_to_coarser()` is exact
    on linear functions with discontinuous boundary values.
    '''
    cshapeh = 3 + np.array(range(ndim))
    cshape = cshapeh * 2

    dimnames = ['x', 'y', 'z', 'w'][:ndim]
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
    msg = '{:.6e}'.format(error)
    failed = error > np.finfo(dtype).eps * 100
    return failed, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(  #
        '--method', type=str, nargs='+', default=['conv'])
    parser.add_argument('--ndim', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--loc',
                        type=str,
                        nargs='+',
                        default=['cccc', 'nnnn', 'cnnn', 'nccc'])
    args = parser.parse_args()
    count_failed = 0
    for method in args.method:
        for ndim in args.ndim:
            for loc in {s[:ndim] for s in args.loc}:
                # Skip unsupported combinations.
                if (isinstance(mod, odil.backend.ModTensorflow)  #
                        and ndim == 4 and method == 'conv'):
                    continue
                try:
                    failed, msg = test(method, ndim, loc)
                except Exception as e:
                    msg = e
                count_failed += bool(failed)
                print(f'method={method:>5}, ndim={ndim}, loc={loc:>4}:',
                      "{} {}".format(msg, 'FAIL' if failed else 'PASS'))
    exit(count_failed)


if __name__ == "__main__":
    main()
