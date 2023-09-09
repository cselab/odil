#!/usr/bin/env python3

import argparse
import numpy as np
import scipy
import scipy.sparse as sp

import odil
from odil import printlog


def operator(mod, ctx):
    extra = ctx.extra

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

    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nx', type=int, default=3, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=2, help="Grid size in y")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_newton')
    parser.set_defaults(multigrid=0)
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
    mod = domain.mod

    from odil import Field, Array
    state = odil.State(
        fields={
            'uc': Field(np.ones(domain.size(loc='cc')), loc='cc'),
            'ufx': Field(np.ones(domain.size(loc='nc')), loc='nc'),
            'a': Array(np.zeros(5)),
            #'net': domain.make_neural_net([1, 7, 1]),
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
        'a': np.arange(state.fields['a'].array.shape[0]),
        #'net_a': np.arange(state.fields['a'].array.shape[0]),
    }

    problem = odil.Problem(operator, domain, extra)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    domain = problem.domain
    extra = problem.extra

    '''
    values, grads, names = problem.eval_operator_grad(state)
    for i in range(len(grads)):
        printlog(f'\ni={i}')
        printlog(f'value\n', values[i])
        printlog(f'grad\n', grads[i])
    exit()
    '''

    # Linearize the operator.
    vector, matrix = problem.linearize(state)
    vector = np.array(vector)

    # Solve the normal equations.
    delta = sp.linalg.spsolve(matrix.T @ matrix, -matrix.T @ vector)
    packed = domain.pack_state(state)
    domain.unpack_state(packed + delta, state)

    failed = 0
    for key in ['ufx', 'uc', 'a']:
        error = domain.field(state, key) - extra.ref[key]
        error = np.sqrt(np.mean(np.square(error)))
        if error < 1e-6:
            status = 'PASS'
        else:
            status = 'FAIL'
            failed += 1
        print(f'{key:<4} {status} {error:.8g}')
    exit(failed)


if __name__ == "__main__":
    main()
