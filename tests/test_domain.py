#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
import odil
from odil.runtime import mod

cases = ['pack', 'arrays']


def test(case, dim):
    cshape = (1 + np.array(range(dim))) * 2
    dimnames = ['x', 'y', 'z', 'w'][:dim]
    domain = odil.Domain(cshape=cshape,
                         dimnames=dimnames,
                         multigrid=1,
                         mg_convert_all=False)

    state = odil.State(
        fields={
            'field': np.random.rand(*cshape),
            'mgfield': domain.regular_to_multigrid(  #
                np.random.rand(*cshape)),
            'net': domain.make_neural_net([3, 3]),
            'array': [1, 2, 3],
        })
    state = domain.init_state(state)
    state2 = deepcopy(state)

    def upd(u):
        return u + 1

    if case == cases[0]:
        # Modify state through packed array.
        packed = domain.pack_state(state)
        packed = upd(packed)
        domain.unpack_state(packed, state)
    elif case == cases[1]:
        # Modify state through raw arrays.
        arrays = domain.arrays_from_state(state)
        for i in range(len(arrays)):
            arrays[i] = upd(arrays[i])
        domain.arrays_to_state(arrays, state)

    # Modify state directly.
    for f in state2.fields.values():
        if isinstance(f, odil.core.Field):
            f.array = upd(f.array)
        elif isinstance(f, odil.core.MultigridField):
            for t in f.terms:
                t.array = upd(t.array)
        elif isinstance(f, odil.core.NeuralNet):
            for i in range(len(f.weights)):
                f.weights[i] = upd(f.weights[i])
                f.biases[i] = upd(f.biases[i])
        elif isinstance(f, odil.core.Array):
            f.array = upd(f.array)

    # Check that both coincide.
    error = max(domain.pack_state(state) - domain.pack_state(state2))
    return error


def main():
    failed = 0
    for case in cases:
        for dim in [1, 2]:
            try:
                error = test(case, dim)
            except Exception as e:
                error = e
                raise e
            failed += bool(error)
            print(f'case={case:>6} dim={dim}:',
                  f"FAIL {error}" if error else "PASS")
    exit(failed)


if __name__ == "__main__":
    main()
