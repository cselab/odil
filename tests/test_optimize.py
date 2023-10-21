#!/usr/bin/env python3

import argparse
import numpy as np

import odil
from odil import printlog


def operator(ctx):
    extra = ctx.extra

    res = []

    for key in ['uc', 'un', 'ufx', 'ufy']:
        res += [(key, ctx.field(key) - extra.ref[key])]
    res += [('a', ctx.field('a') - extra.ref['a'])]
    net_a = ctx.neural_net('net')(ctx.field('a'))[0]
    res += [('net_a', net_a - extra.ref['net_a'])]
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nx', type=int, default=8, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=4, help="Grid size in y")
    parser.add_argument('--optimizers',
                        type=str,
                        default='lbfgsb,adamn',
                        help="Optimizers to test, comma-separated list")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_optimize')
    parser.set_defaults(multigrid=1)
    parser.set_defaults(lr=0.1)
    parser.set_defaults(seed=1)
    parser.set_defaults(epochs=1000, report_every=100, history_every=100)
    return parser.parse_args()


def make_problem(args):
    domain = odil.Domain(cshape=(args.Nx, args.Ny),
                         dimnames=['x', 'y'],
                         lower=(0, 0),
                         upper=(2, 1),
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_axes=[True, True],
                         mg_nlvl=args.nlvl)
    mod = domain.mod
    dtype = domain.dtype

    xx, yy = domain.points('x', 'y', loc='cn')
    ixx, iyy = domain.indices('x', 'y', loc='cc')

    from odil import Field, Array

    state = odil.State(
        fields={
            'uc': Field(np.zeros(domain.size(loc='cc')), loc='cc'),
            'un': Field(np.zeros(domain.size(loc='nn')), loc='nn'),
            'ufx': Field(np.zeros(domain.size(loc='nc')), loc='nc'),
            'ufy': Field(np.zeros(domain.size(loc='cn')), loc='cn'),
            'a': Array(np.zeros(5)),
            'net': domain.make_neural_net([1, 7, 1]),
        })
    state = domain.init_state(state)

    def func(x, y):
        return x * 0.25 + y * 0.5

    extra = argparse.Namespace()
    xc, yc = domain.points(loc='cc')  # Cells.
    xn, yn = domain.points(loc='nn')  # Nodes.
    xfx, yfx = domain.points(loc='nc')  # Faces in x.
    xfy, yfy = domain.points(loc='cn')  # Faces in y.
    extra.ref = {
        'uc': func(xc, yc),
        'un': func(xn, yn),
        'ufx': func(xfx, yfx),
        'ufy': func(xfy, yfy),
        'a': np.arange(state.fields['a'].array.shape[0], dtype=dtype),
    }
    extra.ref['net_a'] = extra.ref['a'] * 0.5

    problem = odil.Problem(operator, domain, extra)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args)
    failed = 0
    for opt in args.optimizers.split(','):
        problem, state = make_problem(args)
        domain = problem.domain
        extra = problem.extra
        callback = odil.make_callback(problem, args)
        odil.util.optimize_grad(args, opt, problem, state, callback)
        error = [
            domain.field(state, key) - extra.ref[key]
            for key in ['uc', 'un', 'ufx', 'ufy', 'a']
        ]
        error.append(
            domain.neural_net(state, 'net')(domain.field(state, 'a')) -
            extra.ref['net_a'])
        error = np.sqrt(sum(np.mean(np.square(e)) for e in error))
        if error < 1e-2:
            status = 'PASS'
        else:
            status = 'FAIL'
            failed += 1
        print(f'opt={opt:>6} {status} {error:.8g}')
    exit(failed)


if __name__ == "__main__":
    main()
