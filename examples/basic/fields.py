#!/usr/bin/env python3

import argparse
import numpy as np
import odil
from odil import printlog
from odil import plotutil
import matplotlib.pyplot as plt
"""
Demonstrates the use of fields with values centered in cell, faces, and nodes.
"""


def operator(ctx):
    res = []

    def func(x, y):
        return x * 0.25 + y * 0.5

    # Cells.
    xc, yc = ctx.points(loc='cc')
    uc = ctx.field('uc')
    res += [('uc', uc - func(xc, yc))]

    # Nodes.
    xn, yn = ctx.points(loc='nn')
    un = ctx.field('un')
    res += [('un', un - func(xn, yn))]

    # Faces in x.
    xfx, yfx = ctx.points(loc='nc')
    ufx = ctx.field('ufx')
    res += [('ufx', ufx - func(xfx, yfx))]

    # Faces in y.
    xfy, yfy = ctx.points(loc='cn')
    ufy = ctx.field('ufy')
    res += [('ufy', ufy - func(xfy, yfy))]
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nx', type=int, default=8, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=4, help="Grid size in y")
    parser.add_argument('--plot', type=int, default=1, help="Plot fields")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_fields')
    parser.set_defaults(echo=1)
    parser.set_defaults(frames=1,
                        plot_every=100,
                        report_every=50,
                        history_every=10)
    parser.set_defaults(optimizer='adam', lr=1e-2)
    parser.set_defaults(multigrid=1)
    return parser.parse_args()


def plot(problem, state, epoch, frame, cbinfo=None):
    from odil import plotutil
    import matplotlib.pyplot as plt
    domain = problem.domain
    fig, ax = plt.subplots(figsize=(4, 2))
    kw = dict(vmin=0, vmax=1, cmap='Greys', clip_on=False, lw=0.5)

    # Cells.
    xc, yc = domain.points(loc='cc')
    uc = domain.field(state, 'uc')
    ax.scatter(xc, yc, s=10, c=uc, edgecolor='C0', label='uc', **kw)

    # Nodes.
    xn, yn = domain.points(loc='nn')
    un = domain.field(state, 'un')
    ax.scatter(xn, yn, s=10, c=un, edgecolor='C1', label='un', **kw)

    # Faces in x.
    xfx, yfx = domain.points(loc='nc')
    ufx = domain.field(state, 'ufx')
    ax.scatter(xfx, yfx, s=10, c=ufx, edgecolor='C2', label='ufx', **kw)

    # Faces in y.
    xfy, yfy = domain.points(loc='cn')
    ufy = domain.field(state, 'ufy')
    ax.scatter(xfy, yfy, s=10, c=ufy, edgecolor='C3', label='ufy', **kw)

    ax.legend(loc='lower left',
              bbox_to_anchor=(0.1, 1),
              ncol=4,
              handletextpad=0)

    ax.pcolormesh(xn,
                  yn,
                  uc,
                  edgecolor='k',
                  shading='flat',
                  zorder=0,
                  **dict(kw, lw=0.5))

    ax.set_aspect('equal')
    ax.set_axis_off()
    plotutil.savefig(fig, "grid_{:05d}".format(frame), printf=printlog)
    plt.close(fig)


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nx, args.Ny),
                         dimnames=['x', 'y'],
                         lower=(0, 0),
                         upper=(2, 1),
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_axes=[True, True],
                         mg_nlvl=args.nlvl)
    xx, yy = domain.points('x', 'y', loc='cn')
    ixx, iyy = domain.indices('x', 'y', loc='cc')

    from odil import Field

    state = odil.State(
        fields={
            'uc': Field(np.zeros(domain.size(loc='cc')), loc='cc'),
            'un': Field(np.zeros(domain.size(loc='nn')), loc='nn'),
            'ufx': Field(np.zeros(domain.size(loc='nc')), loc='nc'),
            'ufy': Field(np.zeros(domain.size(loc='cn')), loc='cn'),
            'net': domain.make_neural_net([2, 4, 2]),
        })
    state = domain.init_state(state)
    problem = odil.Problem(operator, domain)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(problem,
                                  args,
                                  plot_func=plot if args.plot else None)
    odil.util.optimize_grad(args, args.optimizer, problem, state, callback)


if __name__ == "__main__":
    main()
