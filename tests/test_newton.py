#!/usr/bin/env python3

import argparse
import numpy as np
import scipy
import scipy.sparse as sp

import odil
from odil import printlog


def operator(ctx):
    mod = ctx.mod
    extra = ctx.extra
    args = extra.args

    res = []

    # Value in cell (i, j) taken from face (i-0.5, j).
    u_xm = ctx.field('ufx', 0, 0, loc='cc')
    # Value in cell (i, j) taken from face (i+0.5, j).
    u_xp = ctx.field('ufx', 1, 0, loc='cc')
    # Equation in cells, given derivative in x.
    hx = ctx.step('x')
    res += [(u_xp - u_xm) / hx - extra.ref['dudx']]

    # Boundary conditions at x=0.
    ufx = ctx.field('ufx')
    ixfx = ctx.indices('x', loc='nc')
    mask = mod.where(ixfx == 0, ctx.cast(1), ctx.cast(0))
    res += [(ufx - extra.ref['ufx']) * mask]

    # Average over two faces.
    uc = ctx.field('uc')
    res += [(u_xp + u_xm) * 0.5 - uc]

    # Non-grid array, will generate a full Jacobian.
    a = ctx.field('a')
    res += [a - extra.ref['a']]

    # Neural network.
    net_out = ctx.neural_net('net')(*extra.ref['net_in'])
    for i in range(args.Nnet):
        res += [(f'net{i}', net_out[i] - extra.ref['net_out'][i])]
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nx', type=int, default=3, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=2, help="Grid size in y")
    parser.add_argument('--Na', type=int, default=5, help="Size of array a")
    parser.add_argument('--Nnet',
                        type=int,
                        default=5,
                        help="Number of inputs and output in neural net")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_newton')
    parser.set_defaults(multigrid=0)
    parser.set_defaults(seed=1000)
    return parser.parse_args()


def make_problem(args):
    domain = odil.Domain(cshape=(args.Nx, args.Ny),
                         dimnames=['x', 'y'],
                         lower=(0, 0),
                         dtype=np.float64,
                         upper=(args.Nx, args.Ny),
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_axes=[True, True],
                         mg_nlvl=args.nlvl)
    dtype = domain.dtype

    from odil import Field, Array
    state = odil.State(
        fields={
            'uc':
            Field(np.ones(domain.size(loc='cc')), loc='cc'),
            'ufx':
            Field(np.ones(domain.size(loc='nc')), loc='nc'),
            'a':
            Array(np.zeros(args.Na, dtype=dtype)),
            'net':
            domain.make_neural_net([args.Nnet, args.Nnet], activation='none'),
        })
    state = domain.init_state(state)

    def func(x, y):
        return 0.25 * x * y

    def func_x(x, y):
        return 0.25 * y

    extra = argparse.Namespace()
    xc, yc = domain.points(loc='cc')  # Cells.
    xn, yn = domain.points(loc='nn')  # Nodes.
    xfx, yfx = domain.points(loc='nc')  # Faces in x.
    xfy, yfy = domain.points(loc='cn')  # Faces in y.
    extra.ref = {
        'uc': func(xc, yc),
        'ufx': func(xfx, yfx),
        'dudx': func_x(xc, yc),
        'a': np.linspace(0, 1, state.fields['a'].array.shape[0], dtype=dtype),
    }
    # A linear neural net will transform one random matrix into another.
    # The number of degrees of freedom is `Nnet * Nnet + 1`
    # which includes the weights and biases.
    extra.ref['net_in'] = np.random.rand(args.Nnet, args.Nnet + 1)
    extra.ref['net_out'] = np.random.rand(args.Nnet, args.Nnet + 1)
    extra.args = args

    problem = odil.Problem(operator, domain, extra)
    return problem, state


def main():
    if odil.runtime.backend == 'jax':
        print('Skip test_newton.py with jax backend. Not implemented.')
        exit(0)
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    domain = problem.domain
    extra = problem.extra

    # Linearize the operator.
    vector, matrix = problem.linearize(state)
    vector = np.array(vector)

    # Solve the normal equations.
    delta = sp.linalg.spsolve(matrix.T @ matrix, -matrix.T @ vector)
    packed = domain.pack_state(state)
    domain.unpack_state(packed + delta, state)

    failed = 0
    for key in ['ufx', 'uc', 'a', 'net_out']:
        if key == 'net_out':
            value = domain.neural_net(state, 'net')(*extra.ref['net_in'])
        else:
            value = domain.field(state, key)
        error = value - extra.ref[key]
        error = np.sqrt(np.mean(np.square(error)))
        if error < 1e-6:
            status = 'PASS'
        else:
            status = 'FAIL'
            failed += 1
        print(f'{key:<4} {error:.6e} {status}')
    exit(failed)


if __name__ == "__main__":
    main()
